"""
MC-MoE-style expert-level bit allocation for DeepSeek-VL2-Tiny.

Inputs:
  - experts_act_frequency.pkl
  - experts_act_weight.pkl
  - experts_quant_loss.pkl

Outputs:
  - expert-level assignment JSON
  - projection-level quant config JSON compatible with our DeepSeek quantizer
  - summary JSON

This mirrors the factor-based allocation idea from MC-MoE, but adapts it to
DeepSeek's routed experts and our existing quantization/evaluation stack.
"""

import argparse
import json
import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
import torch


PROJ_NAMES = ["gate_proj", "up_proj", "down_proj"]


def dominance_elimination(items: List[Tuple[int, float, int]]) -> List[Tuple[int, float, int]]:
    items = sorted(items, key=lambda x: (x[0], x[1], x[2]))
    kept: List[Tuple[int, float, int]] = []
    best_value = float("inf")
    for cost, value, bit in items:
        if value < best_value:
            kept.append((cost, value, bit))
            best_value = value
    return kept


def get_fixed_quant_layers(
    num_layers: int = 12,
    start_layer: int = 1,
    hidden_size: int = 1280,
    moe_intermediate_size: int = 896,
    dense_intermediate_size: int = 6848,
    n_shared_experts: int = 2,
    quant_attention: bool = True,
    quant_shared_experts: bool = True,
    quant_dense_ffn: bool = True,
) -> Dict[str, int]:
    fixed_layers: Dict[str, int] = {}

    for layer_id in range(num_layers):
        if quant_attention:
            attn_params = hidden_size * hidden_size
            base = f"language.model.layers.{layer_id}.self_attn"
            for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                fixed_layers[f"{base}.{proj}"] = attn_params

        if quant_dense_ffn and layer_id < start_layer:
            base = f"language.model.layers.{layer_id}.mlp"
            fixed_layers[f"{base}.gate_proj"] = dense_intermediate_size * hidden_size
            fixed_layers[f"{base}.up_proj"] = dense_intermediate_size * hidden_size
            fixed_layers[f"{base}.down_proj"] = hidden_size * dense_intermediate_size

        if quant_shared_experts and layer_id >= start_layer:
            shared_intermediate = moe_intermediate_size * n_shared_experts
            base = f"language.model.layers.{layer_id}.mlp.shared_experts"
            fixed_layers[f"{base}.gate_proj"] = shared_intermediate * hidden_size
            fixed_layers[f"{base}.up_proj"] = shared_intermediate * hidden_size
            fixed_layers[f"{base}.down_proj"] = hidden_size * shared_intermediate

    return fixed_layers


def format_tag(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def solve_block_dp(items: List[List[Tuple[int, float, int]]], budget: int) -> Tuple[float, List[int]]:
    """
    Exact DP over experts for a single MoE block.
    Each expert chooses one candidate bit option.
    Cost is the actual bit level (or a remapped surrogate if desired).
    """
    n = len(items)
    dp = np.full((n + 1, budget + 1), np.inf, dtype=np.float64)
    choice = np.full((n + 1, budget + 1), -1, dtype=np.int32)
    dp[0, 0] = 0.0

    for i in range(1, n + 1):
        for prev_cost in range(budget + 1):
            if not np.isfinite(dp[i - 1, prev_cost]):
                continue
            for item_idx, (cost, value, bit) in enumerate(items[i - 1]):
                new_cost = prev_cost + cost
                if new_cost > budget:
                    continue
                new_value = dp[i - 1, prev_cost] + value
                if new_value < dp[i, new_cost]:
                    dp[i, new_cost] = new_value
                    choice[i, new_cost] = item_idx

    best_cost = int(np.argmin(dp[n]))
    best_value = float(dp[n, best_cost])
    selected_bits = [0] * n
    cur_cost = best_cost
    for i in range(n, 0, -1):
        item_idx = int(choice[i, cur_cost])
        if item_idx < 0:
            raise RuntimeError(f"DP backtrace failed at expert index {i - 1}.")
        cost, _, bit = items[i - 1][item_idx]
        selected_bits[i - 1] = bit
        cur_cost -= cost
    return best_value, selected_bits


def main() -> None:
    parser = argparse.ArgumentParser(description="MC-MoE-style solver for DeepSeek-VL2-Tiny")
    parser.add_argument("--actnum_path", type=str, required=True)
    parser.add_argument("--weight_path", type=str, required=True)
    parser.add_argument("--quant_loss_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./mcmoe_configs_deepseek")
    parser.add_argument("--output_prefix", type=str, default="deepseek_vl2_tiny_mcmoe")
    parser.add_argument("--target_avg_bits", type=float, nargs="+", default=[2.59])
    parser.add_argument("--candidate_bits", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--normalize_experts", action="store_true")
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_experts", type=int, default=64)
    parser.add_argument("--start_layer", type=int, default=1)
    parser.add_argument("--fixed_bits", type=int, default=4)
    parser.add_argument("--quant_attention", action="store_true")
    parser.add_argument("--quant_shared_experts", action="store_true")
    parser.add_argument("--quant_dense_ffn", action="store_true")
    parser.add_argument("--hidden_size", type=int, default=1280)
    parser.add_argument("--moe_intermediate_size", type=int, default=896)
    parser.add_argument("--dense_intermediate_size", type=int, default=6848)
    parser.add_argument("--n_shared_experts", type=int, default=2)
    args = parser.parse_args()

    with open(args.actnum_path, "rb") as f:
        actnum = pickle.load(f)
    with open(args.weight_path, "rb") as f:
        weight = pickle.load(f)
    with open(args.quant_loss_path, "rb") as f:
        quant_loss = pickle.load(f)

    blocks = sorted(int(k) for k in actnum.keys())
    if args.normalize_experts:
        for block in blocks:
            actnum[block] = actnum[block].float() / max(float(actnum[block].sum()), 1e-8)
            weight[block] = weight[block].float() / max(float(weight[block].sum()), 1e-8)

    proj_params = {
        "gate_proj": args.hidden_size * args.moe_intermediate_size,
        "up_proj": args.hidden_size * args.moe_intermediate_size,
        "down_proj": args.hidden_size * args.moe_intermediate_size,
    }
    params_per_expert = sum(proj_params.values())

    os.makedirs(args.output_dir, exist_ok=True)
    fixed_layers = get_fixed_quant_layers(
        num_layers=args.num_layers,
        start_layer=args.start_layer,
        hidden_size=args.hidden_size,
        moe_intermediate_size=args.moe_intermediate_size,
        dense_intermediate_size=args.dense_intermediate_size,
        n_shared_experts=args.n_shared_experts,
        quant_attention=args.quant_attention,
        quant_shared_experts=args.quant_shared_experts,
        quant_dense_ffn=args.quant_dense_ffn,
    )

    for target_avg_bits in args.target_avg_bits:
        target_block_budget = int(round(target_avg_bits * args.num_experts))
        layer_assignments: Dict[int, Dict[int, int]] = {}
        total_objective = 0.0

        for block in blocks:
            expert_items: List[List[Tuple[int, float, int]]] = []
            for expert_id in range(args.num_experts):
                significance = float(actnum[block][expert_id]) ** args.alpha
                significance *= float(weight[block][expert_id]) ** args.beta
                items = []
                qloss_dict = quant_loss[block][expert_id]
                for bit in args.candidate_bits:
                    if bit not in qloss_dict:
                        continue
                    value = significance * (float(qloss_dict[bit]) ** args.gamma)
                    items.append((bit, value, bit))
                if not items:
                    raise RuntimeError(f"No quant-loss entries for layer {block} expert {expert_id}.")
                expert_items.append(dominance_elimination(items))

            obj, chosen_bits = solve_block_dp(expert_items, target_block_budget)
            total_objective += obj
            layer_assignments[block] = {eid: bit for eid, bit in enumerate(chosen_bits)}

        config: Dict[str, int] = {}
        routed_bits_total = 0
        routed_counts: Dict[int, int] = {bit: 0 for bit in args.candidate_bits}
        for block in blocks:
            for expert_id, bit in layer_assignments[block].items():
                base = f"language.model.layers.{block}.mlp.experts.{expert_id}"
                for proj in PROJ_NAMES:
                    config[f"{base}.{proj}"] = bit
                routed_bits_total += params_per_expert * bit
                routed_counts[bit] += len(PROJ_NAMES)

        for layer_key in fixed_layers:
            config[layer_key] = args.fixed_bits

        total_routed_params = len(blocks) * args.num_experts * params_per_expert
        fixed_params_total = sum(fixed_layers.values())
        quantized_subset_avg_bits = (
            (routed_bits_total + fixed_params_total * args.fixed_bits) / (total_routed_params + fixed_params_total)
            if (total_routed_params + fixed_params_total) > 0
            else 0.0
        )

        tag = format_tag(target_avg_bits)
        assign_path = os.path.join(args.output_dir, f"{args.output_prefix}_assignments_avg{tag}.json")
        config_path = os.path.join(args.output_dir, f"{args.output_prefix}_avg{tag}.json")
        summary_path = os.path.join(args.output_dir, f"{args.output_prefix}_summary_avg{tag}.json")

        with open(assign_path, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in layer_assignments.items()}, f, indent=4)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "target_routed_avg_bits": target_avg_bits,
                    "block_budget_bits": target_block_budget,
                    "candidate_bits": args.candidate_bits,
                    "total_objective": total_objective,
                    "routed_projection_count_by_bit": routed_counts,
                    "quantized_subset_avg_bits": quantized_subset_avg_bits,
                    "fixed_layer_count": len(fixed_layers),
                    "assignments_path": assign_path,
                    "config_path": config_path,
                },
                f,
                indent=4,
            )

        print("=" * 60)
        print(f"Target routed avg bits: {target_avg_bits:.3f}")
        print(f"Per-block bit budget: {target_block_budget}")
        print(f"Total objective: {total_objective:.6f}")
        print(f"Quantized-subset avg bits: {quantized_subset_avg_bits:.4f}")
        print(f"Saved assignments: {assign_path}")
        print(f"Saved config:      {config_path}")
        print(f"Saved summary:     {summary_path}")


if __name__ == "__main__":
    main()
