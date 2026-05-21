"""
Phase II: MCKP solver for MolMoE.

Input:
  - Phase I menu values JSON produced by cal_owen_value_molmo.py

Output:
  - projection-level mixed-precision config JSON
  - menu assignment JSON
  - summary JSON
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

from molmo_model import (
    get_attention_proj_names,
    get_model_spec,
    get_output_prefix,
    infer_model_config,
    resolve_model_path,
)


PROJ_NAMES = ["gate_proj", "up_proj", "down_proj"]
EXPERT_KEY_RE = re.compile(r"^model\.transformer\.blocks\.(\d+)\.mlp\.experts\.(\d+)$")


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
            "Menu values contain invalid MolMoE expert keys. "
            f"Examples: {preview}"
        )

    expected = (num_layers - start_layer) * num_experts
    if not allow_incomplete and len(seen) != expected:
        raise RuntimeError(
            f"Menu values cover {len(seen)} experts, expected {expected} "
            f"for num_layers={num_layers}, start_layer={start_layer}, num_experts={num_experts}."
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


def get_molmo_proj_params(hidden_size: int, mlp_hidden_size: int) -> Dict[str, int]:
    expert_hidden_size = mlp_hidden_size // 2
    return {
        "gate_proj": hidden_size * expert_hidden_size,
        "up_proj": hidden_size * expert_hidden_size,
        "down_proj": hidden_size * expert_hidden_size,
    }


def get_fixed_quant_layers(
    num_layers: int,
    start_layer: int,
    hidden_size: int,
    attn_proj_names: List[str] | None = None,
    quant_attention: bool = True,
) -> Dict[str, int]:
    fixed_layers: Dict[str, int] = {}
    attn_proj_names = attn_proj_names or ["att_proj"]
    for layer_id in range(num_layers):
        if layer_id < start_layer:
            continue
        if quant_attention:
            base = f"model.transformer.blocks.{layer_id}.attn"
            for proj in attn_proj_names:
                fixed_layers[f"{base}.{proj}"] = hidden_size * hidden_size
    return fixed_layers


def format_tag(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase II: MolMoE MCKP solver")
    parser.add_argument("--menu_values_path", type=str, required=True, help="Phase I menu values JSON")
    parser.add_argument("--target_avg_bits", type=float, nargs="+", default=[2.6], help="Target routed-expert average bits")
    parser.add_argument("--bits_high", type=int, default=4)
    parser.add_argument("--bits_low", type=int, default=2)
    parser.add_argument("--model_id", type=str, default="molmoE-1B-0924")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--hidden_size", type=int, default=2048)
    parser.add_argument("--mlp_hidden_size", type=int, default=1024)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_experts", type=int, default=64)
    parser.add_argument("--start_layer", type=int, default=0)
    parser.add_argument("--fixed_bits", type=int, default=4)
    parser.add_argument("--quant_attention", action="store_true")
    parser.add_argument("--solver", type=str, default="dp", choices=["dp", "lagrangian"])
    parser.add_argument("--output_dir", type=str, default="./mckp_configs_molmo")
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
    if model_cfg.get("mlp_hidden_size") is not None:
        args.mlp_hidden_size = model_cfg["mlp_hidden_size"]
    if model_cfg.get("num_layers") is not None:
        args.num_layers = model_cfg["num_layers"]
    if model_cfg.get("num_experts") is not None:
        args.num_experts = model_cfg["num_experts"]
    attn_proj_names = get_attention_proj_names()

    with open(args.menu_values_path, "r", encoding="utf-8") as f:
        menu_values = json.load(f)
    validate_menu_values(
        menu_values,
        num_layers=args.num_layers,
        num_experts=args.num_experts,
        start_layer=args.start_layer,
        allow_incomplete=args.allow_incomplete_menu,
    )

    proj_params = get_molmo_proj_params(args.hidden_size, args.mlp_hidden_size)
    params_per_expert = sum(proj_params.values())
    total_experts = (args.num_layers - args.start_layer) * args.num_experts
    total_routed_params = total_experts * params_per_expert
    baseline_bits = total_routed_params * args.bits_low
    max_bits = total_routed_params * args.bits_high
    max_extra_bits = max_bits - baseline_bits

    print("\n============================================================")
    print("Phase II: MolMoE MCKP")
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

    experts: List[Dict[str, object]] = []
    for expert_key, menus in menu_values.items():
        items: List[Tuple[str, float, int]] = [("", 0.0, 0)]
        for menu_key, value in menus.items():
            items.append((menu_key, float(value), menu_costs.get(menu_key, 0)))
        experts.append({"key": expert_key, "items": dominance_elimination(items)})

    fixed_layers = get_fixed_quant_layers(
        num_layers=args.num_layers,
        start_layer=args.start_layer,
        hidden_size=args.hidden_size,
        attn_proj_names=attn_proj_names,
        quant_attention=args.quant_attention,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    for target_avg in args.target_avg_bits:
        ratio = (target_avg - args.bits_low) / float(args.bits_high - args.bits_low)
        ratio = min(max(ratio, 0.0), 1.0)
        budget_bits = int(round(ratio * max_extra_bits))

        if args.solver == "dp":
            total_value, assignments = solve_mckp_dp(experts, budget_bits)
        else:
            total_value, assignments = solve_mckp_lagrangian(experts, budget_bits)

        config: Dict[str, int] = {}
        menu_counts: Dict[str, int] = {}
        actual_cost = 0
        for expert_key, menu_key in assignments.items():
            tag = menu_key if menu_key else "(empty)"
            menu_counts[tag] = menu_counts.get(tag, 0) + 1
            actual_cost += menu_costs.get(menu_key, 0)
            high_proj = {x.strip() for x in menu_key.split(",")} if menu_key else set()
            for proj in PROJ_NAMES:
                config[f"{expert_key}.{proj}"] = args.bits_high if proj in high_proj else args.bits_low
        for layer_key in fixed_layers:
            config[layer_key] = args.fixed_bits

        actual_ratio = actual_cost / max_extra_bits if max_extra_bits > 0 else 0.0
        actual_avg_bits = args.bits_low + actual_ratio * (args.bits_high - args.bits_low)
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
                    "solver": args.solver,
                    "total_value": total_value,
                    "menu_distribution": menu_counts,
                    "fixed_layer_count": len(fixed_layers),
                    "config_path": config_path,
                },
                f,
                indent=4,
            )

        print(f"Target {target_avg:.3f} -> actual {actual_avg_bits:.4f}, total value {total_value:.6f}")
        print(f"Saved config:  {config_path}")
        print(f"Saved menus:   {menus_path}")
        print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
