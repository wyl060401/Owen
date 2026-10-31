"""Single cached/batched configuration-utility estimator; no online quantization."""
import math
import time
from itertools import combinations
from typing import List, Dict, Tuple, Optional, FrozenSet
import torch
import torch.nn as nn
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from .bank import candidate_weight, bind_model
from .sampling import permutation_weights
PROJ_NAMES_DEFAULT = ['gate_proj', 'up_proj', 'down_proj']

def enumerate_menus(proj_names=None):
    """
    枚举所有 2^n 个 menu（投影层子集）。
    每个 menu 是一个 frozenset，表示保留高精度的投影层集合。
    n=3 时共 8 个 menu：∅, {g}, {u}, {d}, {g,u}, {g,d}, {u,d}, {g,u,d}
    """
    if proj_names is None:
        proj_names = PROJ_NAMES_DEFAULT
    menus = []
    for r in range(len(proj_names) + 1):
        for combo in combinations(proj_names, r):
            menus.append(frozenset(combo))
    return menus

def menu_to_key(menu: FrozenSet[str]) -> str:
    """将 menu (frozenset) 转为 JSON 可序列化的字符串 key。空集返回空字符串。"""
    return ','.join(sorted(menu)) if menu else ''

def key_to_menu(key: str) -> FrozenSet[str]:
    """将字符串 key 转回 frozenset。"""
    return frozenset(key.split(',')) if key else frozenset()

def get_expert_linear_modules(model, layer_id: int, expert_id: int):
    """
    获取指定专家的所有 Linear 层（gate_proj, up_proj, down_proj）。

    返回：
        dict: {子模块名: nn.Linear}
    """
    bind_model(model)
    try:
        expert = model.language.model.layers[layer_id].mlp.experts[expert_id]
        linears = {}
        for name, module in expert.named_modules():
            if isinstance(module, nn.Linear):
                linears[name] = module
        return linears
    except (AttributeError, IndexError):
        return {}

class ConfigOwenEstimator:

    def __init__(self, num_experts, bits_high=4, bits_low=2, group_size=128, early_stop_threshold=None, device=torch.device('cuda:0')):
        self.num_experts = num_experts
        self.bits_high = bits_high
        self.bits_low = bits_low
        self.group_size = group_size
        self.early_stop_threshold = early_stop_threshold
        self.device = device
        self._loss_fct = torch.nn.CrossEntropyLoss()

    @staticmethod
    def _prepare_attention_mask(lang_model, batch_size, seq_length, inputs_embeds):
        """
        复刻 DeepseekV2Model.forward 中的 attention_mask 准备逻辑。
        - flash_attention_2 且无 padding → None
        - 标准 attention → 4D causal mask

        参数：
            lang_model: DeepseekV2ForCausalLM（即 model.language）
            batch_size: int
            seq_length: int
            inputs_embeds: [B, seq_len, hidden_dim]，用于推断 dtype/device

        返回：
            attention_mask: None 或 [B, 1, seq_len, seq_len] 的 4D causal mask
        """
        use_flash = getattr(lang_model.model, '_use_flash_attention_2', False)
        if use_flash:
            return None
        else:
            return _prepare_4d_causal_attention_mask(None, (batch_size, seq_length), inputs_embeds, past_key_values_length=0)

    @staticmethod
    def _forward_prefix(model, inputs_embeds, attention_mask, position_ids, end_layer: int):
        """
        运行 transformer 的前 end_layer 层（层 0 到 end_layer-1），返回中间 hidden_states。
        不修改模型，不计算 loss。严格等价于完整前向的前半段。

        参数：
            model: DeepseekV2ForCausalLM（即 model.language）
            inputs_embeds: [B, seq_len, hidden_dim]
            attention_mask: 预计算的 4D causal mask 或 None
            position_ids: [1, seq_len]
            end_layer: 前缀终止层（不含），即运行层 0..end_layer-1

        返回：
            hidden_states: [B, seq_len, hidden_dim]，层 end_layer 的输入
        """
        hidden_states = inputs_embeds
        for idx in range(end_layer):
            layer = model.model.layers[idx]
            layer_outputs = layer(hidden_states, attention_mask=attention_mask, position_ids=position_ids, past_key_value=None, output_attentions=False, use_cache=False)
            hidden_states = layer_outputs[0]
        return hidden_states

    @staticmethod
    def _forward_moe_only(layer, mlp_input, residual):
        """
        只运行 decoder layer 的 MoE/MLP 部分并加回残差。
        输入为 _forward_layer_attn_only 的输出。

        返回：
            hidden_states: MoE 输出 + 残差，即该层的完整输出
        """
        moe_out = layer.mlp(mlp_input)
        return residual + moe_out

    @staticmethod
    def _forward_layer_attn_only(layer, hidden_states, attention_mask, position_ids):
        """
        只运行 decoder layer 的 attention + residual + post_attention_layernorm 部分。
        返回 MoE/MLP 的输入（layernorm 后）和残差（用于后续加回）。

        返回：
            mlp_input: post_attention_layernorm 的输出，即 MoE 的输入
            residual: attention 输出 + 原始残差，用于 MoE 输出后加回
        """
        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)
        hidden_states, _, _ = layer.self_attn(hidden_states=hidden_states, attention_mask=attention_mask, position_ids=position_ids, past_key_value=None, output_attentions=False, use_cache=False)
        hidden_states = residual + hidden_states
        residual = hidden_states
        mlp_input = layer.post_attention_layernorm(hidden_states)
        return (mlp_input, residual)

    def _eval_all_menus_batched(self, lang_model, cached_attn_batches, layer_id, expert_gid, all_menus, set_expert_menu_fn, set_expert_all_bits_fn):
        """
        将 baseline(∅) + 7 个非空 menu 的 MoE 输出沿 batch 维拼接，
        suffix 只跑一次，返回每个 menu 的 1/PPL。

        all_menus: [frozenset(), menu1, menu2, ...] 共 8 个，第 0 个是 ∅
        cached_attn_batches: list of (mlp_input, residual, labels, position_ids, attn_mask)
        返回: list of float, 长度 = len(all_menus)
        """
        n_menus = len(all_menus)
        layer = lang_model.model.layers[layer_id]
        total_loss = [0.0] * n_menus
        total_tokens = [0] * n_menus
        menu_chunk_size = n_menus
        for mlp_input, residual, labels, position_ids, attn_mask in cached_attn_batches:
            B = labels.shape[0]
            for chunk_start in range(0, n_menus, menu_chunk_size):
                menu_chunk = all_menus[chunk_start:chunk_start + menu_chunk_size]
                chunk_n = len(menu_chunk)
                moe_hiddens = []
                for menu in menu_chunk:
                    if menu:
                        set_expert_menu_fn(expert_gid, menu)
                    else:
                        set_expert_all_bits_fn(expert_gid, self.bits_low)
                    h = self._forward_moe_only(layer, mlp_input, residual)
                    moe_hiddens.append(h)
                stacked_hidden = torch.cat(moe_hiddens, dim=0)
                stacked_labels = labels.repeat(chunk_n, 1)
                stacked_pos = position_ids
                if attn_mask is not None:
                    stacked_attn_mask = attn_mask.repeat(chunk_n, 1, 1, 1)
                else:
                    stacked_attn_mask = None
                h = stacked_hidden
                num_layers = len(lang_model.model.layers)
                for idx in range(layer_id + 1, num_layers):
                    ly = lang_model.model.layers[idx]
                    h = ly(h, attention_mask=stacked_attn_mask, position_ids=stacked_pos, past_key_value=None, output_attentions=False, use_cache=False)[0]
                h = lang_model.model.norm(h)
                logits = lang_model.lm_head(h)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = stacked_labels[..., 1:].contiguous()
                for local_idx in range(chunk_n):
                    mi = chunk_start + local_idx
                    sl = shift_logits[local_idx * B:(local_idx + 1) * B]
                    slb = shift_labels[local_idx * B:(local_idx + 1) * B]
                    mask = slb != -100
                    n_tok = mask.sum().item()
                    if n_tok > 0:
                        loss = self._loss_fct(sl.float().reshape(-1, sl.shape[-1]), slb.reshape(-1).to(sl.device))
                        total_loss[mi] += loss.item() * n_tok
                        total_tokens[mi] += n_tok
                del moe_hiddens, stacked_hidden, stacked_labels, stacked_attn_mask
                del h, logits, shift_logits, shift_labels
        values = []
        for mi in range(n_menus):
            avg = total_loss[mi] / max(total_tokens[mi], 1)
            ppl = math.exp(min(avg, 100))
            values.append(1.0 / ppl)
        return values

    def estimate_configuration_values(self, model, inputs: List, permutations: torch.Tensor, p_tilde: torch.Tensor, num_experts_per_layer: int, start_layer: int, value_fn, proj_names: List[str]=None, batch_size: int=4) -> Dict[str, Dict[str, float]]:
        """
        前缀共享加速版 menu value 估计。
        对 expert i (layer L) 的 8 个 menu，prefix (layers 0..L-1) 和
        layer L attention 只算一次，仅重跑 MoE + suffix。
        同层连续 expert 共享 prefix + attention 缓存。

        唯一的配置效用估计入口：缓存前缀并批处理配置。
        """
        if proj_names is None:
            proj_names = PROJ_NAMES_DEFAULT
        menus = enumerate_menus(proj_names)
        empty_menu = frozenset()
        nonempty_menus = [m for m in menus if m != empty_menu]
        M = permutations.shape[0]
        N = self.num_experts
        device = self.device
        lang_model = model.language
        menu_sum = {}
        for gid in range(N):
            for menu in nonempty_menus:
                menu_sum[gid, menu_to_key(menu)] = 0.0
        print('缓存原始原始精度权重...')
        orig_weights = {}
        proj_module_cache = {}
        current_bits_state = {}
        for global_id in range(N):
            layer_id = global_id // num_experts_per_layer + start_layer
            expert_id = global_id % num_experts_per_layer
            linears = get_expert_linear_modules(model, layer_id, expert_id)
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
                orig_weights[global_id][matched_proj] = module.weight.data.detach().cpu().clone()
                proj_module_cache[global_id][matched_proj] = module
                current_bits_state[global_id][matched_proj] = None
        cached_orig_layer_id = None
        cached_orig_layer_weights = {}

        def ensure_layer_orig_weights_on_device(layer_id):
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
                    cached_orig_layer_weights[global_id][proj_name] = cpu_weight.to(device=module.weight.device, dtype=module.weight.dtype, non_blocking=True)
            cached_orig_layer_id = layer_id

        def set_proj_bits(global_id, proj_name, bits):
            if current_bits_state[global_id][proj_name] == bits:
                return
            module = proj_module_cache[global_id][proj_name]
            module.weight.data = candidate_weight(module, bits, self.group_size)
            current_bits_state[global_id][proj_name] = bits

        def set_expert_all_bits(global_id, bits):
            for pn in proj_names:
                if pn in proj_module_cache[global_id]:
                    set_proj_bits(global_id, pn, bits)

        def set_expert_menu(global_id, menu):
            for pn in proj_names:
                if pn in proj_module_cache[global_id]:
                    bits = self.bits_high if pn in menu else self.bits_low
                    set_proj_bits(global_id, pn, bits)

        def restore_expert(global_id):
            for pn in proj_names:
                if pn in proj_module_cache[global_id]:
                    module = proj_module_cache[global_id][pn]
                    layer_id = global_id // num_experts_per_layer + start_layer
                    ensure_layer_orig_weights_on_device(layer_id)
                    module.weight.data = cached_orig_layer_weights[global_id][pn]
                    current_bits_state[global_id][pn] = None
        is_embeds = inputs[0][0].dim() == 3
        batches = []
        for i in range(0, len(inputs), batch_size):
            be = torch.cat([inputs[j][0] for j in range(i, min(i + batch_size, len(inputs)))], dim=0).contiguous()
            bl = torch.cat([inputs[j][1] for j in range(i, min(i + batch_size, len(inputs)))], dim=0).contiguous()
            batches.append((be, bl))
        print(f'初始化：所有专家施加 {self.bits_high}-bit fixed GPTQ candidates...')
        for global_id in range(N):
            set_expert_all_bits(global_id, self.bits_high)
        print('开始计算全 high-bit 基准价值...', flush=True)
        with torch.no_grad():
            value_full = value_fn(model, inputs)
        print(f'全 {self.bits_high}-bit 基准价值: {value_full:.6f}', flush=True)
        max_experts_per_perm = N
        self._permutation_weights = permutation_weights(permutations, p_tilde)
        total_start_time = time.time()
        dirty_experts = set()
        processed_count = 0
        for m in range(M):
            perm_start = time.time()
            elapsed_total = perm_start - total_start_time
            print(f'\n排列 {m + 1}/{M} (已用时 {elapsed_total:.1f}s)', flush=True)
            permutation = permutations[m]
            weight = self._permutation_weights[m]
            if m == 0:
                print('  复用初始化后的 all-high 状态，跳过首次全量重置', flush=True)
            else:
                for global_id in dirty_experts:
                    set_expert_all_bits(global_id, self.bits_high)
                dirty_experts.clear()
            value_before_expert = value_full
            cached_layer_id = -1
            cached_attn_batches = None
            experts_evaluated = 0
            for n in range(max_experts_per_perm):
                if self.early_stop_threshold is not None:
                    if value_before_expert < self.early_stop_threshold * value_full:
                        print(f'  n={n} 早停（价值降至 {value_before_expert / value_full:.2%}）', flush=True)
                        break
                expert_start = time.time()
                expert_gid = permutation[n].item()
                layer_id = expert_gid // num_experts_per_layer + start_layer
                expert_in_layer = expert_gid % num_experts_per_layer
                if layer_id != cached_layer_id:
                    cached_attn_batches = []
                    with torch.no_grad():
                        for be_cpu, bl_cpu in batches:
                            be = be_cpu.to(device, non_blocking=True)
                            bl = bl_cpu.to(device, non_blocking=True)
                            if is_embeds:
                                embeds = be
                            else:
                                embeds = lang_model.model.embed_tokens(be)
                            B_cur = embeds.shape[0]
                            seq_len = embeds.shape[1]
                            pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
                            attn_mask = self._prepare_attention_mask(lang_model, B_cur, seq_len, embeds)
                            hidden = self._forward_prefix(lang_model, embeds, attn_mask, pos_ids, layer_id)
                            layer_module = lang_model.model.layers[layer_id]
                            mlp_inp, res = self._forward_layer_attn_only(layer_module, hidden, attn_mask, pos_ids)
                            cached_attn_batches.append((mlp_inp, res, bl, pos_ids, attn_mask))
                    cached_layer_id = layer_id
                full_menu = frozenset(proj_names)
                all_menus_with_empty = [empty_menu] + [menu for menu in nonempty_menus if n == 0 or menu != full_menu]
                with torch.no_grad():
                    all_values = self._eval_all_menus_batched(lang_model, cached_attn_batches, layer_id, expert_gid, all_menus_with_empty, set_expert_menu, set_expert_all_bits)
                v_base = all_values[0]
                values_by_menu = dict(zip(all_menus_with_empty, all_values))
                if n > 0:
                    values_by_menu[full_menu] = value_before_expert
                for menu in nonempty_menus:
                    H = values_by_menu[menu] - v_base
                    mk = menu_to_key(menu)
                    weighted_h = weight * H
                    menu_sum[expert_gid, mk] += weighted_h
                set_expert_all_bits(expert_gid, self.bits_low)
                value_before_expert = v_base
                dirty_experts.add(expert_gid)
                experts_evaluated += 1
                if n + 1 < max_experts_per_perm:
                    next_gid = permutation[n + 1].item()
                    next_layer = next_gid // num_experts_per_layer + start_layer
                    if next_layer != layer_id:
                        cached_layer_id = -1
                expert_elapsed = time.time() - expert_start
                if n < 5 or n % 20 == 0:
                    perm_elapsed = time.time() - perm_start
                    avg_per = perm_elapsed / experts_evaluated
                    eta = avg_per * (max_experts_per_perm - n - 1) + avg_per * max_experts_per_perm * (M - m - 1)
                    print(f'  n={n}, expert={expert_gid} (L{layer_id}.E{expert_in_layer}), V={value_before_expert:.6f}, 耗时={expert_elapsed:.2f}s, ETA≈{eta / 60:.1f}min ', flush=True)
            perm_elapsed = time.time() - perm_start
            print(f'  排列 {m + 1} 完成, 评估了 {experts_evaluated}/{N} 个专家, 耗时 {perm_elapsed:.1f}s', flush=True)
            processed_count += 1
        total_elapsed = time.time() - total_start_time
        print(f'\n全部排列完成, 总耗时 {total_elapsed:.1f}s ({total_elapsed / 60:.1f}min)', flush=True)
        print('恢复原始原始精度权重...', flush=True)
        for global_id in range(N):
            restore_expert(global_id)
        del cached_attn_batches
        menu_values = {}
        denom = max(processed_count, 1)
        for global_id in range(N):
            layer_id = global_id // num_experts_per_layer + start_layer
            expert_id = global_id % num_experts_per_layer
            expert_key = f'L{layer_id}.E{expert_id}'
            menu_values[expert_key] = {}
            for menu in nonempty_menus:
                mk = menu_to_key(menu)
                menu_values[expert_key][mk] = menu_sum[global_id, mk] / denom
        return menu_values
