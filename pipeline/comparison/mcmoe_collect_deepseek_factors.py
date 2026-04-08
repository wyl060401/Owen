"""
Collect MC-MoE-style expert factors for DeepSeek-VL2-Tiny.

Outputs:
  - experts_act_frequency.pkl : {layer_id: Tensor[num_experts]}
  - experts_act_weight.pkl    : {layer_id: Tensor[num_experts]}
  - experts_quant_loss.pkl    : {layer_id: {expert_id: {bit: loss}}}

This script is intentionally scoped to routed experts. Attention/shared experts
are treated as fixed-background layers later by the solver / quantization config.
"""

import argparse
import json
import math
import os
import gc
import pickle
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from cal_owen_value import get_c4, get_multimodal_calib, _PROJECT_ROOT, _VLM_DIR


def fake_quantize_weight(weight: torch.Tensor, bits: int, group_size: int = 128) -> torch.Tensor:
    """
    Symmetric RTN fake quantization.
    This is the same style of cheap quant surrogate we use elsewhere in the repo
    for routing / value estimation work.
    """
    w = weight
    qmax = 2 ** (bits - 1) - 1
    out_features, in_features = w.shape

    if group_size > 0 and in_features >= group_size:
        num_groups = in_features // group_size
        w_grouped = w[:, : num_groups * group_size].reshape(out_features * num_groups, group_size)
        max_val = w_grouped.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
        scale = max_val / qmax
        w_q = (w_grouped / scale).round().clamp(-qmax, qmax)
        w_dq = (w_q * scale).reshape(out_features, num_groups * group_size)
        if in_features > num_groups * group_size:
            tail = w[:, num_groups * group_size :]
            max_tail = tail.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
            scale_tail = max_tail / qmax
            tail_q = (tail / scale_tail).round().clamp(-qmax, qmax)
            w_dq = torch.cat([w_dq, tail_q * scale_tail], dim=1)
    else:
        max_val = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
        scale = max_val / qmax
        w_q = (w / scale).round().clamp(-qmax, qmax)
        w_dq = w_q * scale

    return w_dq


def parse_bits(bits_text: str) -> List[int]:
    return [int(x.strip()) for x in bits_text.split(",") if x.strip()]


def iter_moe_modules(model: nn.Module, start_layer: int) -> Iterable[Tuple[int, nn.Module, str]]:
    for name, module in model.named_modules():
        if "DeepseekV2MoE" not in type(module).__name__:
            continue
        layer_id = getattr(module, "layer_id", None)
        if layer_id is None:
            parts = name.split(".")
            for i, part in enumerate(parts):
                if part == "layers" and i + 1 < len(parts):
                    layer_id = int(parts[i + 1])
                    setattr(module, "layer_id", layer_id)
                    break
        if layer_id is None or layer_id < start_layer:
            continue
        yield layer_id, module, name


def install_monitors(model: nn.Module, start_layer: int, cache_batches: int):
    records: Dict[int, Dict[str, object]] = {}

    for layer_id, module, _ in iter_moe_modules(model, start_layer):
        records[layer_id] = {
            "module": module,
            "counts": torch.zeros(module.config.n_routed_experts, dtype=torch.float32),
            "weights": torch.zeros(module.config.n_routed_experts, dtype=torch.float32),
            "inputs": [],
            "outputs": [],
        }

        gate = module.gate
        gate._mcmoe_original_forward = gate.forward

        def gate_forward_wrapper(hidden_states, _orig=gate.forward, _rec=records[layer_id]):
            topk_idx, topk_weight, aux_loss = _orig(hidden_states)
            flat_idx = topk_idx.reshape(-1)
            flat_weight = topk_weight.reshape(-1).float()
            counts = torch.zeros_like(_rec["counts"], device=flat_idx.device)
            counts.scatter_add_(0, flat_idx, torch.ones_like(flat_weight))
            weights = torch.zeros_like(_rec["weights"], device=flat_idx.device)
            weights.scatter_add_(0, flat_idx, flat_weight)
            _rec["counts"] += counts.cpu()
            _rec["weights"] += weights.cpu()
            return topk_idx, topk_weight, aux_loss

        gate.forward = gate_forward_wrapper

        def moe_hook(_mod, inputs, output, _rec=records[layer_id]):
            if len(_rec["inputs"]) < cache_batches:
                _rec["inputs"].append(inputs[0].detach().cpu())
                _rec["outputs"].append(output.detach().cpu())

        hook = module.register_forward_hook(moe_hook)
        records[layer_id]["hook"] = hook

    return records


def remove_monitors(records: Dict[int, Dict[str, object]]) -> None:
    for rec in records.values():
        hook = rec.get("hook")
        if hook is not None:
            hook.remove()
        gate = rec["module"].gate
        if hasattr(gate, "_mcmoe_original_forward"):
            gate.forward = gate._mcmoe_original_forward
            delattr(gate, "_mcmoe_original_forward")


@torch.no_grad()
def forward_calibration(model, inputs, device, batch_size: int = 1):
    is_embeds = inputs[0][0].dim() == 3
    num_samples = len(inputs)
    model.eval()

    if is_embeds:
        for i in range(num_samples):
            batch_embeds = inputs[i][0].to(device)
            batch_labels = inputs[i][1].to(device)
            _ = model.language(inputs_embeds=batch_embeds, labels=batch_labels)
    else:
        for i in range(0, num_samples, batch_size):
            batch_inputs = []
            for j in range(i, min(i + batch_size, num_samples)):
                batch_inputs.append(inputs[j][0])
            batch_inputs = torch.cat(batch_inputs, dim=0).to(device)
            _ = model(input_ids=batch_inputs, labels=batch_inputs)


def collect_quant_loss_for_layer(
    module: nn.Module,
    cached_inputs: List[torch.Tensor],
    cached_outputs: List[torch.Tensor],
    candidate_bits: List[int],
    group_size: int,
    device: torch.device,
) -> Dict[int, Dict[int, float]]:
    loss_history: Dict[int, Dict[int, float]] = {}
    proj_names = ["gate_proj", "up_proj", "down_proj"]

    for expert_id, expert in enumerate(module.experts):
        per_bit: Dict[int, float] = {}
        orig_weights = {
            proj: getattr(expert, proj).weight.data.detach().clone()
            for proj in proj_names
        }
        try:
            for bit in candidate_bits:
                for proj in proj_names:
                    proj_mod = getattr(expert, proj)
                    proj_mod.weight.data = fake_quantize_weight(orig_weights[proj], bit, group_size).to(
                        proj_mod.weight.data.dtype
                    )

                total_loss = 0.0
                for x_cpu, y_cpu in zip(cached_inputs, cached_outputs):
                    x = x_cpu.to(device)
                    y_ref = y_cpu.to(device, dtype=torch.float64)
                    y_hat = module(x).to(torch.float64)
                    total_loss += torch.norm(y_ref - y_hat).item()
                per_bit[bit] = float(total_loss)
        finally:
            for proj in proj_names:
                getattr(expert, proj).weight.data = orig_weights[proj]

        loss_history[expert_id] = per_bit

    return loss_history


def pick_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DeepSeek factor collection.")
    return torch.device("cuda:0")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect MC-MoE-style DeepSeek factors")
    parser.add_argument("--model", type=str, required=True, help="DeepSeek-VL2-Tiny local model path")
    parser.add_argument("--output_dir", type=str, default=os.path.join(_PROJECT_ROOT, "mcmoe_factors_deepseek"))
    parser.add_argument("--calib_data", type=str, default="c4", choices=["c4", "multimodal"])
    parser.add_argument("--calib_data_path", type=str, default=None)
    parser.add_argument("--nsamples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seqlen", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--cache_batches", type=int, default=8)
    parser.add_argument("--candidate_bits", type=str, default="2,3,4")
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--start_layer", type=int, default=1)
    parser.add_argument("--cache_dir", type=str, default=os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))
    args = parser.parse_args()

    candidate_bits = parse_bits(args.candidate_bits)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        cache_dir=args.cache_dir,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    sys_path_added = False
    if _VLM_DIR not in os.sys.path:
        os.sys.path.insert(0, _VLM_DIR)
        sys_path_added = True

    try:
        from DeepSeek_VL2.modeling_deepseek_vl_v2 import DeepseekVLV2ForCausalLM
    except ImportError:
        from DeepSeek_VL2_quant.modeling_deepseek_vl_v2 import DeepseekVLV2ForCausalLM

    device = pick_device()
    torch.cuda.set_device(device)
    model = DeepseekVLV2ForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        cache_dir=args.cache_dir,
        local_files_only=True,
    )
    model.eval()

    if args.calib_data == "c4":
        inputs = get_c4(args.nsamples, args.seed, args.seqlen, tokenizer, data_path=args.calib_data_path)
    else:
        inputs = get_multimodal_calib(
            args.nsamples,
            args.seed,
            model,
            device,
            data_path=args.calib_data_path,
            model_path=args.model,
            seqlen=args.seqlen,
        )

    records = install_monitors(model, args.start_layer, args.cache_batches)

    print(f"Collecting activation factors on {len(inputs)} calibration samples...")
    forward_calibration(model, inputs, device, batch_size=args.batch_size)

    act_frequency: Dict[int, torch.Tensor] = {}
    act_weight: Dict[int, torch.Tensor] = {}
    for layer_id, rec in records.items():
        act_frequency[layer_id] = rec["counts"].clone()
        act_weight[layer_id] = rec["weights"].clone()

    print("Collecting per-expert quantization loss...")
    quant_loss: Dict[int, Dict[int, Dict[int, float]]] = {}
    for layer_id in sorted(records.keys()):
        rec = records[layer_id]
        print(f"  Layer {layer_id}: {len(rec['inputs'])} cached batches")
        quant_loss[layer_id] = collect_quant_loss_for_layer(
            rec["module"],
            rec["inputs"],
            rec["outputs"],
            candidate_bits,
            args.group_size,
            device,
        )

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "experts_act_frequency.pkl"), "wb") as f:
        pickle.dump(act_frequency, f)
    with open(os.path.join(args.output_dir, "experts_act_weight.pkl"), "wb") as f:
        pickle.dump(act_weight, f)
    with open(os.path.join(args.output_dir, "experts_quant_loss.pkl"), "wb") as f:
        pickle.dump(quant_loss, f)
    with open(os.path.join(args.output_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": args.model,
                "calib_data": args.calib_data,
                "calib_data_path": args.calib_data_path,
                "nsamples": args.nsamples,
                "seqlen": args.seqlen,
                "batch_size": args.batch_size,
                "cache_batches": args.cache_batches,
                "candidate_bits": candidate_bits,
                "group_size": args.group_size,
                "start_layer": args.start_layer,
            },
            f,
            indent=4,
        )

    remove_monitors(records)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    print(f"Saved factors to: {args.output_dir}")
    if sys_path_added:
        os.sys.path.pop(0)


if __name__ == "__main__":
    main()
