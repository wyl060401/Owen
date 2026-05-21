"""
Phase II: MCKP solver for DeepSeek-VL2.

Input:
  - Phase I menu values JSON produced by cal_owen_value.py

Output:
  - projection-level mixed-precision config JSON
  - menu assignment JSON
  - summary JSON

Assumptions:
  - routed experts use 2/4-bit menu choices over {gate_proj, up_proj, down_proj}
  - optional attention/shared-expert/dense-FFN layers are quantized to fixed_bits
  - target_avg_bits applies to routed-expert projections only
"""

import argparse
import json
import os
import re
import sys
from functools import reduce
from math import gcd
from typing import Dict, List, Tuple

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from deepseek_vl2_model import (
    get_attention_proj_names,
    get_model_spec,
    get_output_prefix,
    infer_model_config,
    resolve_model_path,
)


PROJ_NAMES = ["gate_proj", "up_proj", "down_proj"]
EXPERT_KEY_RE = re.compile(r"^language\.model\.layers\.(\d+)\.mlp\.experts\.(\d+)$")


def validate_menu_values(
    menu_values: Dict[str, Dict[str, float]],
    num_layers: int,
    num_experts: int,
    start_layer: int,
    allow_incomplete: bool = False,
) -> None:
    invalid_keys = []
    seen = set()
    for expert_key in menu_values:
        match = EXPERT_KEY_RE.match(expert_key)
        if not match:
            invalid_keys.append(expert_key)
            continue
        layer_id = int(match.group(1))
        expert_id = int(match.group(2))
        if layer_id < start_layer or layer_id >= num_layers or expert_id < 0 or expert_id >= num_experts:
            invalid_keys.append(expert_key)
            continue
        seen.add((layer_id, expert_id))

    if invalid_keys:
        preview = ", ".join(invalid_keys[:5])
        raise RuntimeError(
            f"Menu values contain invalid DeepSeek expert keys for "
            f"layers [{start_layer}, {num_layers - 1}] and experts [0, {num_experts - 1}]. "
            f"Examples: {preview}"
        )

    expected = (num_layers - start_layer) * num_experts
    if not allow_incomplete and len(seen) != expected:
        raise RuntimeError(
            f"Menu values cover {len(seen)} experts, expected {expected} "
            f"for num_layers={num_layers}, start_layer={start_layer}, num_experts={num_experts}. "
            "Refusing to emit a partial mixed-precision config. "
            "Pass --allow_incomplete_menu only for debugging subsets."
        )


def compute_menu_cost(
    menu_key: str,
    proj_params: Dict[str, int],
    bits_high: int,
    bits_low: int,
) -> int:
    if not menu_key:
        return 0
    cost = 0
    for proj in menu_key.split(","):
        proj = proj.strip()
        if proj in proj_params:
            cost += proj_params[proj] * (bits_high - bits_low)
    return cost


def dominance_elimination(items: List[Tuple[str, float, int]]) -> List[Tuple[str, float, int]]:
    items = sorted(items, key=lambda x: (x[2], -x[1]))
    non_dominated: List[Tuple[str, float, int]] = []
    for item in items:
        dominated = False
        for kept in non_dominated:
            if kept[2] <= item[2] and kept[1] >= item[1]:
                if kept[2] < item[2] or kept[1] > item[1]:
                    dominated = True
                    break
        if dominated:
            continue
        non_dominated = [
            kept
            for kept in non_dominated
            if not (
                item[2] <= kept[2]
                and item[1] >= kept[1]
                and (item[2] < kept[2] or item[1] > kept[1])
            )
        ]
        non_dominated.append(item)
    return non_dominated


def solve_mckp_lagrangian(
    experts: List[Dict[str, object]],
    budget_bits: int,
    tol: float = 0.001,
    max_iter: int = 100,
) -> Tuple[float, Dict[str, str]]:
    def select_at_lambda(lam: float) -> Tuple[Dict[str, str], float, float]:
        assignments: Dict[str, str] = {}
        total_cost = 0.0
        total_val = 0.0
        for expert in experts:
            best_mk, best_score, best_cost, best_val = "", 0.0, 0.0, 0.0
            for mk, val, cost in expert["items"]:
                score = val - lam * cost
                if score > best_score or (score == best_score and cost < best_cost):
                    best_mk, best_score, best_cost, best_val = mk, score, cost, val
            assignments[expert["key"]] = best_mk
            total_cost += best_cost
            total_val += best_val
        return assignments, total_cost, total_val

    positive_costs = [c for expert in experts for _, _, c in expert["items"] if c > 0]
    if not positive_costs:
        assignments = {}
        total_val = 0.0
        for expert in experts:
            best = max(expert["items"], key=lambda x: x[1])
            assignments[expert["key"]] = best[0]
            total_val += best[1]
        return total_val, assignments

    max_val = max(val for expert in experts for _, val, _ in expert["items"])
    min_cost = min(positive_costs)
    lam_lo, lam_hi = 0.0, max_val / min_cost + 1e-6

    for _ in range(max_iter):
        lam_mid = (lam_lo + lam_hi) / 2.0
        _, total_cost, _ = select_at_lambda(lam_mid)
        if abs(total_cost - budget_bits) / max(budget_bits, 1) < tol:
            break
        if total_cost > budget_bits:
            lam_lo = lam_mid
        else:
            lam_hi = lam_mid

    assignments, _, total_val = select_at_lambda((lam_lo + lam_hi) / 2.0)
    return total_val, assignments


def solve_mckp_dp(
    experts: List[Dict[str, object]],
    budget_bits: int,
) -> Tuple[float, Dict[str, str]]:
    if not experts:
        return 0.0, {}

    all_costs = [c for expert in experts for _, _, c in expert["items"] if c > 0]
    if not all_costs:
        assignments = {}
        total_val = 0.0
        for expert in experts:
            best = max(expert["items"], key=lambda x: x[1])
            assignments[expert["key"]] = best[0]
            total_val += best[1]
        return total_val, assignments

    cost_gcd = reduce(gcd, [int(c) for c in all_costs])
    cost_gcd = cost_gcd or 1
    capacity = int(budget_bits) // cost_gcd

    if len(experts) * capacity > 100_000_000:
        print(f"  DP table too large ({len(experts)} x {capacity}), fallback to lagrangian")
        return solve_mckp_lagrangian(experts, budget_bits)

    dp = np.full(capacity + 1, -np.inf, dtype=np.float64)
    dp[0] = 0.0
    choice = np.full((len(experts), capacity + 1), -1, dtype=np.int32)

    for idx, expert in enumerate(experts):
        new_dp = np.full(capacity + 1, -np.inf, dtype=np.float64)
        new_choice = np.full(capacity + 1, -1, dtype=np.int32)
        for item_idx, (menu_key, value, cost) in enumerate(expert["items"]):
            scaled_cost = int(cost) // cost_gcd
            if scaled_cost > capacity:
                continue
            for w in range(scaled_cost, capacity + 1):
                candidate = dp[w - scaled_cost] + value
                if candidate > new_dp[w]:
                    new_dp[w] = candidate
                    new_choice[w] = item_idx
        dp = new_dp
        choice[idx] = new_choice

    best_w = int(np.argmax(dp))
    total_val = float(dp[best_w])
    assignments: Dict[str, str] = {}
    w = best_w
    for idx in range(len(experts) - 1, -1, -1):
        chosen = int(choice[idx][w])
        if chosen >= 0:
            menu_key, _, cost = experts[idx]["items"][chosen]
            assignments[experts[idx]["key"]] = menu_key
            w -= int(cost) // cost_gcd
        else:
            assignments[experts[idx]["key"]] = ""
    return total_val, assignments


def get_deepseek_proj_params(hidden_size: int = 1280, moe_intermediate_size: int = 896) -> Dict[str, int]:
    return {
        "gate_proj": moe_intermediate_size * hidden_size,
        "up_proj": moe_intermediate_size * hidden_size,
        "down_proj": hidden_size * moe_intermediate_size,
    }


def get_fixed_quant_layers(
    num_layers: int = 12,
    start_layer: int = 1,
    hidden_size: int = 1280,
    moe_intermediate_size: int = 896,
    dense_intermediate_size: int = 6848,
    n_shared_experts: int = 2,
    attn_proj_names: List[str] = None,
    quant_attention: bool = True,
    quant_shared_experts: bool = True,
    quant_dense_ffn: bool = True,
) -> Dict[str, int]:
    fixed_layers: Dict[str, int] = {}
    attn_proj_names = attn_proj_names or ["q_proj", "k_proj", "v_proj", "o_proj"]

    for layer_id in range(num_layers):
        if quant_attention:
            attn_params = hidden_size * hidden_size
            base = f"language.model.layers.{layer_id}.self_attn"
            for proj in attn_proj_names:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase II: DeepSeek-VL2 MCKP solver")
    parser.add_argument("--menu_values_path", type=str, required=True, help="Phase I menu values JSON")
    parser.add_argument("--target_avg_bits", type=float, nargs="+", default=[2.6], help="Target routed-expert average bits")
    parser.add_argument("--bits_high", type=int, default=4)
    parser.add_argument("--bits_low", type=int, default=2)
    parser.add_argument("--model_id", type=str, default="deepseek_vl2_tiny")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--hidden_size", type=int, default=1280)
    parser.add_argument("--moe_intermediate_size", type=int, default=896)
    parser.add_argument("--dense_intermediate_size", type=int, default=6848)
    parser.add_argument("--n_shared_experts", type=int, default=2)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_experts", type=int, default=64)
    parser.add_argument("--start_layer", type=int, default=1)
    parser.add_argument("--fixed_bits", type=int, default=4)
    parser.add_argument("--quant_attention", action="store_true")
    parser.add_argument("--quant_shared_experts", action="store_true")
    parser.add_argument("--quant_dense_ffn", action="store_true")
    parser.add_argument("--solver", type=str, default="dp", choices=["dp", "lagrangian"])
    parser.add_argument("--output_dir", type=str, default="./mckp_configs_deepseek")
    parser.add_argument("--output_prefix", type=str, default=None)
    parser.add_argument(
        "--allow_incomplete_menu",
        action="store_true",
        help="Allow debug subset menu files that do not cover every routed expert.",
    )
    args = parser.parse_args()

    model_spec = get_model_spec(args.model_id)
    resolved_model_path = resolve_model_path(args.model_path or args.model_id, _PROJECT_ROOT)
    model_cfg = infer_model_config(resolved_model_path)
    if args.output_prefix is None:
        args.output_prefix = get_output_prefix(model_spec.model_id, "menu_mckp")
    if model_cfg.get("hidden_size") is not None:
        args.hidden_size = model_cfg["hidden_size"]
    if model_cfg.get("moe_intermediate_size") is not None:
        args.moe_intermediate_size = model_cfg["moe_intermediate_size"]
    if model_cfg.get("intermediate_size") is not None:
        args.dense_intermediate_size = model_cfg["intermediate_size"]
    if model_cfg.get("num_layers") is not None:
        args.num_layers = model_cfg["num_layers"]
    if model_cfg.get("num_experts") is not None:
        args.num_experts = model_cfg["num_experts"]
    if model_cfg.get("n_shared_experts") is not None:
        args.n_shared_experts = model_cfg["n_shared_experts"]
    attn_proj_names = get_attention_proj_names(model_cfg, args.model_id)

    with open(args.menu_values_path, "r", encoding="utf-8") as f:
        menu_values = json.load(f)
    validate_menu_values(
        menu_values,
        num_layers=args.num_layers,
        num_experts=args.num_experts,
        start_layer=args.start_layer,
        allow_incomplete=args.allow_incomplete_menu,
    )

    proj_params = get_deepseek_proj_params(args.hidden_size, args.moe_intermediate_size)
    params_per_expert = sum(proj_params.values())
    total_experts = (args.num_layers - args.start_layer) * args.num_experts
    total_routed_params = total_experts * params_per_expert
    baseline_bits = total_routed_params * args.bits_low
    max_bits = total_routed_params * args.bits_high
    max_extra_bits = max_bits - baseline_bits

    print("\n============================================================")
    print("Phase II: DeepSeek-VL2 MCKP")
    print("============================================================")
    print(f"Model: {model_spec.model_id}")
    print(f"Resolved model path: {resolved_model_path}")
    print(f"Input menu values: {args.menu_values_path}")
    print(f"Loaded experts: {len(menu_values)}")
    print(f"Solver: {args.solver}")
    print(f"Routed experts: {total_experts}")
    print(f"Per-expert params: {params_per_expert:,}")
    print(f"All-{args.bits_low}bit routed size: {baseline_bits / 8 / 1024**3:.3f} GB")
    print(f"All-{args.bits_high}bit routed size: {max_bits / 8 / 1024**3:.3f} GB")

    all_menu_keys = {menu_key for menus in menu_values.values() for menu_key in menus.keys()}
    menu_costs = {
        menu_key: compute_menu_cost(menu_key, proj_params, args.bits_high, args.bits_low)
        for menu_key in all_menu_keys
    }
    menu_costs[""] = 0

    print("\nBuilding MCKP items with dominance elimination...")
    experts: List[Dict[str, object]] = []
    total_items_before = 0
    total_items_after = 0
    for expert_key, menus in menu_values.items():
        items: List[Tuple[str, float, int]] = [("", 0.0, 0)]
        for menu_key, value in menus.items():
            items.append((menu_key, float(value), menu_costs.get(menu_key, 0)))
        total_items_before += len(items)
        pruned = dominance_elimination(items)
        total_items_after += len(pruned)
        experts.append({"key": expert_key, "items": pruned})
    print(f"  Before: {total_items_before} items")
    print(f"  After:  {total_items_after} items")
    print(f"  Avg per expert: {total_items_after / max(len(experts), 1):.2f}")

    fixed_layers = get_fixed_quant_layers(
        num_layers=args.num_layers,
        start_layer=args.start_layer,
        hidden_size=args.hidden_size,
        moe_intermediate_size=args.moe_intermediate_size,
        dense_intermediate_size=args.dense_intermediate_size,
        n_shared_experts=args.n_shared_experts,
        attn_proj_names=attn_proj_names,
        quant_attention=args.quant_attention,
        quant_shared_experts=args.quant_shared_experts,
        quant_dense_ffn=args.quant_dense_ffn,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    for target_avg in args.target_avg_bits:
        ratio = (target_avg - args.bits_low) / float(args.bits_high - args.bits_low)
        ratio = min(max(ratio, 0.0), 1.0)
        budget_bits = int(round(ratio * max_extra_bits))

        print("\n------------------------------------------------------------")
        print(f"Target routed avg bits: {target_avg:.3f}")
        print(f"Equivalent high-bit ratio: {ratio:.4f}")
        print(f"Extra bit budget: {budget_bits:,}")

        if args.solver == "dp":
            total_value, assignments = solve_mckp_dp(experts, budget_bits)
        else:
            total_value, assignments = solve_mckp_lagrangian(experts, budget_bits)

        menu_counts: Dict[str, int] = {}
        actual_cost = 0
        for _, menu_key in assignments.items():
            tag = menu_key if menu_key else "(empty)"
            menu_counts[tag] = menu_counts.get(tag, 0) + 1
            actual_cost += menu_costs.get(menu_key, 0)

        actual_ratio = actual_cost / max_extra_bits if max_extra_bits > 0 else 0.0
        actual_avg_bits = args.bits_low + actual_ratio * (args.bits_high - args.bits_low)

        config: Dict[str, int] = {}
        for expert_key, menu_key in assignments.items():
            high_proj = {x.strip() for x in menu_key.split(",")} if menu_key else set()
            for proj in PROJ_NAMES:
                config[f"{expert_key}.{proj}"] = args.bits_high if proj in high_proj else args.bits_low
        for layer_key in fixed_layers:
            config[layer_key] = args.fixed_bits

        routed_high = sum(1 for key, bit in config.items() if "experts" in key and "shared_experts" not in key and bit == args.bits_high)
        routed_low = sum(1 for key, bit in config.items() if "experts" in key and "shared_experts" not in key and bit == args.bits_low)
        routed_bits_total = sum(
            proj_params[proj] * config[f"{expert_key}.{proj}"]
            for expert_key in menu_values.keys()
            for proj in PROJ_NAMES
        )
        fixed_params_total = sum(fixed_layers.values())
        fixed_bits_total = fixed_params_total * args.fixed_bits
        quantized_params_total = total_routed_params + fixed_params_total
        quantized_avg_bits = (
            (routed_bits_total + fixed_bits_total) / quantized_params_total if quantized_params_total > 0 else 0.0
        )

        print(f"  Total value: {total_value:.6f}")
        print(f"  Actual routed avg bits: {actual_avg_bits:.4f}")
        print(f"  Routed projections: {routed_high} high / {routed_low} low")
        if fixed_layers:
            print(f"  Fixed {args.fixed_bits}-bit layers: {len(fixed_layers)}")
            print(f"  Quantized-subset avg bits (routed + fixed): {quantized_avg_bits:.4f}")
        print("  Menu distribution:")
        for menu_key, count in sorted(menu_counts.items(), key=lambda x: (-x[1], x[0])):
            print(f"    {menu_key}: {count}")

        tag = format_tag(target_avg)
        config_path = os.path.join(args.output_dir, f"{args.output_prefix}_avg{tag}.json")
        menus_path = os.path.join(args.output_dir, f"{args.output_prefix}_menus_avg{tag}.json")
        summary_path = os.path.join(args.output_dir, f"{args.output_prefix}_summary_avg{tag}.json")

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)
        with open(menus_path, "w", encoding="utf-8") as f:
            json.dump(assignments, f, indent=4)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "target_routed_avg_bits": target_avg,
                    "actual_routed_avg_bits": actual_avg_bits,
                    "quantized_subset_avg_bits": quantized_avg_bits,
                    "solver": args.solver,
                    "total_value": total_value,
                    "routed_high": routed_high,
                    "routed_low": routed_low,
                    "fixed_layer_count": len(fixed_layers),
                    "menu_distribution": menu_counts,
                    "config_path": config_path,
                },
                f,
                indent=4,
            )

        print(f"  Saved config:   {config_path}")
        print(f"  Saved menus:    {menus_path}")
        print(f"  Saved summary:  {summary_path}")


if __name__ == "__main__":
    main()
