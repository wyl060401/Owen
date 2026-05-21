import math
import time
from typing import Dict, FrozenSet, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from rgis_shapley import (
    PROJ_NAMES_DEFAULT,
    RGISShapleyEstimator,
    enumerate_menus,
    fake_quantize_weight,
    menu_to_key,
)


def get_molmo_expert_linear_modules(model, layer_id: int, expert_id: int):
    try:
        expert = model.model.transformer.blocks[layer_id].mlp.experts[expert_id]
        linears = {}
        for name, module in expert.named_modules():
            if isinstance(module, nn.Linear):
                linears[name] = module
        return linears
    except (AttributeError, IndexError):
        return {}


class RGISShapleyEstimatorMolmo(RGISShapleyEstimator):
    @staticmethod
    def _finite_attention_bias_(x: torch.Tensor) -> torch.Tensor:
        x.masked_fill_(x == float("-inf"), torch.finfo(x.dtype).min)
        return x

    @staticmethod
    def _causal_attention_bias(seq_len: int, device: torch.device) -> torch.Tensor:
        att_bias = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.float),
            diagonal=1,
        )
        att_bias.masked_fill_(att_bias == 1, torch.finfo(att_bias.dtype).min)
        return att_bias.view(1, 1, seq_len, seq_len)

    @staticmethod
    def _prepare_decoder_inputs(model, batch: Dict[str, torch.Tensor]):
        core_model = model.model
        config = model.config
        input_ids = batch.get("input_ids")
        inputs_embeds = batch["inputs_embeds"]
        attention_mask = batch.get("attention_mask")

        batch_size, seq_len = inputs_embeds.shape[:2]
        if config.use_position_ids and attention_mask is None and input_ids is not None:
            attention_mask = input_ids != -1

        position_ids = batch.get("position_ids")
        if config.use_position_ids and attention_mask is not None and position_ids is None:
            position_ids = torch.clamp(
                torch.cumsum(attention_mask.to(torch.int32), dim=-1) - 1,
                min=0,
            ).broadcast_to((batch_size, attention_mask.shape[-1]))

        x = core_model.transformer.emb_drop(inputs_embeds)
        if config.normalize_input_embeds:
            x = x * (config.d_model ** 0.5)

        attention_bias = batch.get("attention_bias")
        if attention_mask is not None:
            if len(attention_mask.shape) == 2:
                attention_mask_4d = attention_mask.to(dtype=torch.float).view(batch_size, -1)[:, None, None, :]
            else:
                attention_mask_4d = attention_mask.unsqueeze(1).to(dtype=torch.float)
            attention_mask_4d = (1.0 - attention_mask_4d) * torch.finfo(attention_mask_4d.dtype).min
        else:
            attention_mask_4d = None

        if attention_bias is not None or attention_mask_4d is not None:
            if attention_bias is None:
                attention_bias = RGISShapleyEstimatorMolmo._causal_attention_bias(seq_len, x.device)
            elif attention_bias.dtype in (torch.int8, torch.bool):
                attention_bias = attention_bias.to(dtype=torch.float)
                attention_bias.masked_fill_(attention_bias == 0.0, torch.finfo(attention_bias.dtype).min)

            mask_len = attention_mask_4d.shape[-1] if attention_mask_4d is not None else seq_len
            attention_bias = attention_bias[:, :, :mask_len, :mask_len].to(dtype=torch.float)
            if attention_mask_4d is not None:
                attention_bias = attention_bias + attention_mask_4d
                RGISShapleyEstimatorMolmo._finite_attention_bias_(attention_bias)

        return x, attention_bias, position_ids, batch["labels"]

    @staticmethod
    def _forward_prefix(model, hidden_states, attention_bias, position_ids, layer_id: int):
        blocks = model.model.transformer.blocks
        for idx in range(layer_id):
            hidden_states, _ = blocks[idx](
                hidden_states,
                attention_bias=attention_bias,
                position_ids=position_ids,
                drop_mask=None,
                layer_past=None,
                use_cache=False,
            )
        return hidden_states

    @staticmethod
    def _forward_layer_attn_only(layer, hidden_states, attention_bias, position_ids):
        att, _ = layer.attn(
            hidden_states,
            attention_bias=attention_bias,
            position_ids=position_ids,
            drop_mask=None,
            layer_past=None,
            use_cache=False,
        )
        return hidden_states + layer.dropout(att, drop_mask=None)

    @staticmethod
    def _forward_moe_only(layer, mlp_input):
        moe_out, _ = layer.mlp(mlp_input)
        return mlp_input + layer.dropout(moe_out, drop_mask=None)

    @staticmethod
    def _forward_suffix_loss(model, hidden_states, attention_bias, position_ids, start_layer: int, labels):
        core_model = model.model
        for idx in range(start_layer, len(core_model.transformer.blocks)):
            hidden_states, _ = core_model.transformer.blocks[idx](
                hidden_states,
                attention_bias=attention_bias,
                position_ids=position_ids,
                drop_mask=None,
                layer_past=None,
                use_cache=False,
            )

        hidden_states = core_model.transformer.ln_f(hidden_states)
        if model.config.weight_tying:
            logits = F.linear(hidden_states, core_model.transformer.wte.weight, None)
        else:
            logits = model.lm_head(hidden_states)

        if model.config.scale_logits:
            logits.mul_(1 / math.sqrt(model.config.d_model))
        if model.config.final_logit_softcapping is not None:
            logits = logits / model.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * model.config.final_logit_softcapping

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)
        n_tokens = int((shift_labels != -100).sum().item())
        if n_tokens <= 0:
            return 0.0, 0

        loss_sum = torch.nn.functional.cross_entropy(
            shift_logits.float().view(-1, model.config.embedding_size),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="sum",
        )
        return float(loss_sum.item()), n_tokens

    def _build_molmo_attn_cache(self, model, inputs: List[Dict[str, torch.Tensor]], layer_id: int):
        cached_batches = []
        layer = model.model.transformer.blocks[layer_id]
        with torch.no_grad():
            for batch in inputs:
                hidden, attention_bias, position_ids, labels = self._prepare_decoder_inputs(model, batch)
                hidden = self._forward_prefix(model, hidden, attention_bias, position_ids, layer_id)
                mlp_input = self._forward_layer_attn_only(layer, hidden, attention_bias, position_ids)
                cached_batches.append((mlp_input, labels, position_ids, attention_bias))
        return cached_batches

    def _eval_from_molmo_attn_cache(self, model, cached_batches, layer_id: int) -> float:
        total_loss = 0.0
        total_tokens = 0
        layer = model.model.transformer.blocks[layer_id]
        with torch.no_grad():
            for mlp_input, labels, position_ids, attention_bias in cached_batches:
                hidden = self._forward_moe_only(layer, mlp_input)
                loss_sum, n_tokens = self._forward_suffix_loss(
                    model,
                    hidden,
                    attention_bias,
                    position_ids,
                    start_layer=layer_id + 1,
                    labels=labels,
                )
                total_loss += loss_sum
                total_tokens += n_tokens
        avg_loss = total_loss / max(total_tokens, 1)
        return math.exp(-min(avg_loss, 100))

    def eval_cached_full_forward_value(self, model, inputs: List[Dict[str, torch.Tensor]], layer_id: int) -> float:
        cached_batches = self._build_molmo_attn_cache(model, inputs, layer_id)
        return self._eval_from_molmo_attn_cache(model, cached_batches, layer_id)

    def estimate_menu_value(
        self,
        model,
        inputs: List,
        permutations: torch.Tensor,
        p_tilde: torch.Tensor,
        num_experts_per_layer: int,
        start_layer: int,
        value_fn,
        proj_names: List[str] = None,
    ) -> Dict[str, Dict[str, float]]:
        if proj_names is None:
            proj_names = PROJ_NAMES_DEFAULT

        menus = enumerate_menus(proj_names)
        empty_menu = frozenset()
        nonempty_menus = [m for m in menus if m != empty_menu]

        M = permutations.shape[0]
        N = self.num_experts
        progress_interval = max(1, math.ceil(N / 8))

        menu_sum = {}
        for gid in range(N):
            for menu in nonempty_menus:
                menu_sum[(gid, menu_to_key(menu))] = 0.0

        print("Caching original FP weights...", flush=True)
        orig_weights = {}
        proj_module_cache = {}
        current_bits_state = {}
        for global_id in range(N):
            layer_id = global_id // num_experts_per_layer + start_layer
            expert_id = global_id % num_experts_per_layer
            linears = get_molmo_expert_linear_modules(model, layer_id, expert_id)
            orig_weights[global_id] = {}
            proj_module_cache[global_id] = {}
            current_bits_state[global_id] = {}
            for name, module in linears.items():
                matched_proj = None
                for pn in proj_names:
                    if pn in name:
                        matched_proj = pn
                        break
                if matched_proj is None:
                    matched_proj = name
                orig_weights[global_id][matched_proj] = module.weight.detach().cpu().clone()
                proj_module_cache[global_id][matched_proj] = module
                current_bits_state[global_id][matched_proj] = None
            if (global_id + 1) % progress_interval == 0 or global_id + 1 == N:
                print(
                    f"  cached FP weights for {global_id + 1}/{N} experts",
                    flush=True,
                )

        cached_orig_layer_id = None
        cached_orig_layer_weights = {}

        def ensure_layer_orig_weights_on_device(layer_id: int):
            nonlocal cached_orig_layer_id, cached_orig_layer_weights
            if cached_orig_layer_id == layer_id:
                return

            cached_orig_layer_weights = {}
            layer_offset = layer_id - start_layer
            layer_base_gid = layer_offset * num_experts_per_layer
            for expert_id in range(num_experts_per_layer):
                global_id = layer_base_gid + expert_id
                if global_id not in orig_weights:
                    continue
                cached_orig_layer_weights[global_id] = {}
                for proj_name, cpu_weight in orig_weights[global_id].items():
                    module = proj_module_cache[global_id][proj_name]
                    cached_orig_layer_weights[global_id][proj_name] = cpu_weight.to(
                        device=module.weight.device,
                        dtype=module.weight.dtype,
                        non_blocking=True,
                    )
            cached_orig_layer_id = layer_id

        def set_proj_bits(global_id: int, proj_name: str, bits: int):
            if current_bits_state[global_id][proj_name] == bits:
                return
            module = proj_module_cache[global_id][proj_name]
            layer_id = global_id // num_experts_per_layer + start_layer
            ensure_layer_orig_weights_on_device(layer_id)
            orig_weight = cached_orig_layer_weights[global_id][proj_name]
            module.weight.data = fake_quantize_weight(
                orig_weight, bits, self.group_size
            )
            current_bits_state[global_id][proj_name] = bits

        def set_expert_all_bits(global_id: int, bits: int):
            for pn in proj_names:
                if pn in proj_module_cache[global_id]:
                    set_proj_bits(global_id, pn, bits)

        def set_expert_menu(global_id: int, menu: FrozenSet[str]):
            for pn in proj_names:
                if pn in proj_module_cache[global_id]:
                    bits = self.bits_high if pn in menu else self.bits_low
                    set_proj_bits(global_id, pn, bits)

        def restore_expert(global_id: int):
            for pn in proj_names:
                if pn in proj_module_cache[global_id]:
                    module = proj_module_cache[global_id][pn]
                    layer_id = global_id // num_experts_per_layer + start_layer
                    ensure_layer_orig_weights_on_device(layer_id)
                    module.weight.data = cached_orig_layer_weights[global_id][pn].clone()
                    current_bits_state[global_id][pn] = None

        print(f"Initializing all experts to {self.bits_high}-bit fake quant...", flush=True)
        for global_id in range(N):
            set_expert_all_bits(global_id, self.bits_high)
            if (global_id + 1) % progress_interval == 0 or global_id + 1 == N:
                print(
                    f"  initialized {global_id + 1}/{N} experts to {self.bits_high}-bit",
                    flush=True,
                )

        print(f"Computing all-{self.bits_high}bit baseline value...", flush=True)
        with torch.no_grad():
            value_full = value_fn(model, inputs)
        print(f"All-{self.bits_high}bit baseline value: {value_full:.6f}", flush=True)

        max_expert_fraction = getattr(self, "max_expert_fraction_per_permutation", 1.0)
        max_experts_per_perm = N
        if max_expert_fraction < 1.0:
            max_experts_per_perm = max(
                1, math.ceil(N * max_expert_fraction)
            )
            print(
                "Expert prefix truncation enabled: "
                f"evaluating {max_experts_per_perm}/{N} experts per permutation "
                f"({max_expert_fraction:.2%})",
                flush=True,
            )

        total_start_time = time.time()
        dirty_experts = set()
        for m in range(M):
            perm_start_time = time.time()
            elapsed_total = perm_start_time - total_start_time
            print(f"\nPermutation {m + 1}/{M} (elapsed {elapsed_total:.1f}s)", flush=True)
            permutation = permutations[m]

            perm_prob = self.compute_plackett_luce_probability(permutation, p_tilde)
            weight = self.compute_importance_weights(torch.tensor([perm_prob]))[0].item()
            print(f"  permutation prob: {perm_prob:.2e}, IS weight: {weight:.2e}", flush=True)

            if m == 0 and not dirty_experts:
                print("  reusing initialized all-high state for first permutation", flush=True)
            else:
                for global_id in dirty_experts:
                    set_expert_all_bits(global_id, self.bits_high)
                dirty_experts.clear()

            value_before_expert = value_full
            experts_evaluated = 0
            for n in range(max_experts_per_perm):
                if self.early_stop_threshold is not None:
                    if value_before_expert < self.early_stop_threshold * value_full:
                        print(
                            f"  early stop at n={n}, value ratio "
                            f"{value_before_expert / max(value_full, 1e-12):.2%} "
                            f"< threshold {self.early_stop_threshold:.2%}",
                            flush=True,
                        )
                        break

                expert_start = time.time()
                expert_gid = permutation[n].item()
                layer_id = expert_gid // num_experts_per_layer + start_layer
                expert_in_layer = expert_gid % num_experts_per_layer

                set_expert_all_bits(expert_gid, self.bits_low)
                with torch.no_grad():
                    v_base = value_fn(model, inputs)

                for menu in nonempty_menus:
                    set_expert_menu(expert_gid, menu)
                    with torch.no_grad():
                        v_menu = value_fn(model, inputs)
                    menu_sum[(expert_gid, menu_to_key(menu))] += weight * (v_menu - v_base)

                set_expert_all_bits(expert_gid, self.bits_low)
                value_before_expert = v_base
                dirty_experts.add(expert_gid)
                experts_evaluated += 1

                expert_elapsed = time.time() - expert_start
                if n < 5 or n % 20 == 0:
                    perm_elapsed = time.time() - perm_start_time
                    avg_per_expert = perm_elapsed / experts_evaluated
                    remaining_this_perm = avg_per_expert * (max_experts_per_perm - n - 1)
                    remaining_future_perms = avg_per_expert * max_experts_per_perm * (M - m - 1)
                    eta_min = (remaining_this_perm + remaining_future_perms) / 60
                    delta_from_full = value_before_expert - value_full
                    print(
                        f"  n={n}, expert={expert_gid} (L{layer_id}.E{expert_in_layer}), "
                        f"V={value_before_expert:.8f}, dV={delta_from_full:+.8f}, "
                        f"cost={expert_elapsed:.2f}s, ETA~{eta_min:.1f}min",
                        flush=True,
                    )

            perm_elapsed = time.time() - perm_start_time
            print(
                f"  permutation {m + 1} done, evaluated {experts_evaluated}/{N} experts in {perm_elapsed:.1f}s",
                flush=True,
            )

        total_elapsed = time.time() - total_start_time
        print(f"\nAll permutations done in {total_elapsed:.1f}s ({total_elapsed / 60:.1f}min)", flush=True)
        print("Restoring original FP weights...", flush=True)
        for global_id in range(N):
            restore_expert(global_id)

        menu_values = {}
        for global_id in range(N):
            layer_id = global_id // num_experts_per_layer + start_layer
            expert_id = global_id % num_experts_per_layer
            expert_key = f"L{layer_id}.E{expert_id}"
            menu_values[expert_key] = {}
            for menu in nonempty_menus:
                mk = menu_to_key(menu)
                menu_values[expert_key][mk] = menu_sum[(global_id, mk)] / M

        return menu_values

    def estimate_menu_value_cached_full_forward(
        self,
        model,
        inputs: List,
        permutations: torch.Tensor,
        p_tilde: torch.Tensor,
        num_experts_per_layer: int,
        start_layer: int,
        value_fn,
        proj_names: List[str] = None,
    ) -> Dict[str, Dict[str, float]]:
        if proj_names is None:
            proj_names = PROJ_NAMES_DEFAULT
        if not inputs or not isinstance(inputs[0], dict) or "inputs_embeds" not in inputs[0]:
            raise ValueError("cached full-forward MolMoE path requires precomputed inputs_embeds eval batches")

        menus = enumerate_menus(proj_names)
        empty_menu = frozenset()
        nonempty_menus = [m for m in menus if m != empty_menu]

        M = permutations.shape[0]
        N = self.num_experts
        progress_interval = max(1, math.ceil(N / 8))

        menu_sum = {}
        for gid in range(N):
            for menu in nonempty_menus:
                menu_sum[(gid, menu_to_key(menu))] = 0.0

        print("Caching original FP weights...", flush=True)
        orig_weights = {}
        proj_module_cache = {}
        current_bits_state = {}
        for global_id in range(N):
            layer_id = global_id // num_experts_per_layer + start_layer
            expert_id = global_id % num_experts_per_layer
            linears = get_molmo_expert_linear_modules(model, layer_id, expert_id)
            orig_weights[global_id] = {}
            proj_module_cache[global_id] = {}
            current_bits_state[global_id] = {}
            for name, module in linears.items():
                matched_proj = None
                for pn in proj_names:
                    if pn in name:
                        matched_proj = pn
                        break
                if matched_proj is None:
                    matched_proj = name
                orig_weights[global_id][matched_proj] = module.weight.detach().cpu().clone()
                proj_module_cache[global_id][matched_proj] = module
                current_bits_state[global_id][matched_proj] = None
            if (global_id + 1) % progress_interval == 0 or global_id + 1 == N:
                print(f"  cached FP weights for {global_id + 1}/{N} experts", flush=True)

        cached_orig_layer_id = None
        cached_orig_layer_weights = {}

        def ensure_layer_orig_weights_on_device(layer_id: int):
            nonlocal cached_orig_layer_id, cached_orig_layer_weights
            if cached_orig_layer_id == layer_id:
                return
            cached_orig_layer_weights = {}
            layer_offset = layer_id - start_layer
            layer_base_gid = layer_offset * num_experts_per_layer
            for expert_id in range(num_experts_per_layer):
                global_id = layer_base_gid + expert_id
                if global_id not in orig_weights:
                    continue
                cached_orig_layer_weights[global_id] = {}
                for proj_name, cpu_weight in orig_weights[global_id].items():
                    module = proj_module_cache[global_id][proj_name]
                    cached_orig_layer_weights[global_id][proj_name] = cpu_weight.to(
                        device=module.weight.device,
                        dtype=module.weight.dtype,
                        non_blocking=True,
                    )
            cached_orig_layer_id = layer_id

        def set_proj_bits(global_id: int, proj_name: str, bits: int):
            if current_bits_state[global_id][proj_name] == bits:
                return
            module = proj_module_cache[global_id][proj_name]
            layer_id = global_id // num_experts_per_layer + start_layer
            ensure_layer_orig_weights_on_device(layer_id)
            orig_weight = cached_orig_layer_weights[global_id][proj_name]
            module.weight.data = fake_quantize_weight(orig_weight, bits, self.group_size)
            current_bits_state[global_id][proj_name] = bits

        def set_expert_all_bits(global_id: int, bits: int):
            for pn in proj_names:
                if pn in proj_module_cache[global_id]:
                    set_proj_bits(global_id, pn, bits)

        def set_expert_menu(global_id: int, menu: FrozenSet[str]):
            for pn in proj_names:
                if pn in proj_module_cache[global_id]:
                    bits = self.bits_high if pn in menu else self.bits_low
                    set_proj_bits(global_id, pn, bits)

        def restore_expert(global_id: int):
            for pn in proj_names:
                if pn in proj_module_cache[global_id]:
                    module = proj_module_cache[global_id][pn]
                    layer_id = global_id // num_experts_per_layer + start_layer
                    ensure_layer_orig_weights_on_device(layer_id)
                    module.weight.data = cached_orig_layer_weights[global_id][pn].clone()
                    current_bits_state[global_id][pn] = None

        print(f"Initializing all experts to {self.bits_high}-bit fake quant...", flush=True)
        for global_id in range(N):
            set_expert_all_bits(global_id, self.bits_high)
            if (global_id + 1) % progress_interval == 0 or global_id + 1 == N:
                print(f"  initialized {global_id + 1}/{N} experts to {self.bits_high}-bit", flush=True)

        print(f"Computing all-{self.bits_high}bit baseline value...", flush=True)
        with torch.no_grad():
            value_full = value_fn(model, inputs)
        print(f"All-{self.bits_high}bit baseline value: {value_full:.6f}", flush=True)

        max_expert_fraction = getattr(self, "max_expert_fraction_per_permutation", 1.0)
        max_experts_per_perm = N
        if max_expert_fraction < 1.0:
            max_experts_per_perm = max(1, math.ceil(N * max_expert_fraction))
            print(
                "Expert prefix truncation enabled: "
                f"evaluating {max_experts_per_perm}/{N} experts per permutation "
                f"({max_expert_fraction:.2%})",
                flush=True,
            )

        all_menus = [empty_menu] + nonempty_menus
        total_start_time = time.time()
        dirty_experts = set()
        for m in range(M):
            perm_start_time = time.time()
            elapsed_total = perm_start_time - total_start_time
            print(f"\nPermutation {m + 1}/{M} (elapsed {elapsed_total:.1f}s) [cached-full]", flush=True)
            permutation = permutations[m]

            perm_prob = self.compute_plackett_luce_probability(permutation, p_tilde)
            weight = self.compute_importance_weights(torch.tensor([perm_prob]))[0].item()
            print(f"  permutation prob: {perm_prob:.2e}, IS weight: {weight:.2e}", flush=True)

            for global_id in dirty_experts:
                set_expert_all_bits(global_id, self.bits_high)
            dirty_experts.clear()

            cached_layer_id: Optional[int] = None
            cached_attn_batches = None
            value_before_expert = value_full
            experts_evaluated = 0
            for n in range(max_experts_per_perm):
                if self.early_stop_threshold is not None:
                    if value_before_expert < self.early_stop_threshold * value_full:
                        print(
                            f"  early stop at n={n}, value ratio "
                            f"{value_before_expert / max(value_full, 1e-12):.2%} "
                            f"< threshold {self.early_stop_threshold:.2%}",
                            flush=True,
                        )
                        break

                expert_start = time.time()
                expert_gid = permutation[n].item()
                layer_id = expert_gid // num_experts_per_layer + start_layer
                expert_in_layer = expert_gid % num_experts_per_layer

                if cached_layer_id != layer_id:
                    cached_attn_batches = self._build_molmo_attn_cache(model, inputs, layer_id)
                    cached_layer_id = layer_id

                values = []
                for menu in all_menus:
                    if menu:
                        set_expert_menu(expert_gid, menu)
                    else:
                        set_expert_all_bits(expert_gid, self.bits_low)
                    values.append(self._eval_from_molmo_attn_cache(model, cached_attn_batches, layer_id))

                v_base = values[0]
                for menu, v_menu in zip(nonempty_menus, values[1:]):
                    menu_sum[(expert_gid, menu_to_key(menu))] += weight * (v_menu - v_base)

                set_expert_all_bits(expert_gid, self.bits_low)
                value_before_expert = v_base
                dirty_experts.add(expert_gid)
                experts_evaluated += 1

                expert_elapsed = time.time() - expert_start
                if n < 5 or n % 20 == 0:
                    perm_elapsed = time.time() - perm_start_time
                    avg_per_expert = perm_elapsed / experts_evaluated
                    remaining_this_perm = avg_per_expert * (max_experts_per_perm - n - 1)
                    remaining_future_perms = avg_per_expert * max_experts_per_perm * (M - m - 1)
                    eta_min = (remaining_this_perm + remaining_future_perms) / 60
                    delta_from_full = value_before_expert - value_full
                    print(
                        f"  n={n}, expert={expert_gid} (L{layer_id}.E{expert_in_layer}), "
                        f"V={value_before_expert:.8f}, dV={delta_from_full:+.8f}, "
                        f"cost={expert_elapsed:.2f}s, ETA~{eta_min:.1f}min [cached-full]",
                        flush=True,
                    )

            perm_elapsed = time.time() - perm_start_time
            print(
                f"  permutation {m + 1} done, evaluated {experts_evaluated}/{N} experts in {perm_elapsed:.1f}s",
                flush=True,
            )

        total_elapsed = time.time() - total_start_time
        print(f"\nAll permutations done in {total_elapsed:.1f}s ({total_elapsed / 60:.1f}min)", flush=True)
        print("Restoring original FP weights...", flush=True)
        for global_id in range(N):
            restore_expert(global_id)

        menu_values = {}
        for global_id in range(N):
            layer_id = global_id // num_experts_per_layer + start_layer
            expert_id = global_id % num_experts_per_layer
            expert_key = f"L{layer_id}.E{expert_id}"
            menu_values[expert_key] = {}
            for menu in nonempty_menus:
                mk = menu_to_key(menu)
                menu_values[expert_key][mk] = menu_sum[(global_id, mk)] / M

        return menu_values
