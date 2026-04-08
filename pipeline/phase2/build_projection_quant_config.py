"""
将 Phase III 投影层最优配置合并到最终量化配置文件。

用途：
1) 读取 local swap 的 final_s_hi（专家级结果）
2) 读取一个或多个 projection_shapley 结果文件（含 optimal_config）
3) 生成可直接用于量化/评测的 JSON（专家外 low-bit，S_hi 内按投影层细分）
"""

import argparse
import json
import os
from typing import Dict, List, Set, Tuple


PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def parse_expert_key(key: str) -> Tuple[int, int]:
    parts = key.split(".")
    return int(parts[3]), int(parts[6])


def load_final_s_hi(swap_result_path: str) -> Set[Tuple[int, int]]:
    with open(swap_result_path, "r") as f:
        data = json.load(f)
    return {parse_expert_key(k) for k in data["final_s_hi"]}


def load_projection_results(projection_files: List[str]) -> Dict[Tuple[int, int], List[str]]:
    merged: Dict[Tuple[int, int], List[str]] = {}
    for path in projection_files:
        with open(path, "r") as f:
            data = json.load(f)
        for item in data.get("experts", []):
            key = (int(item["layer_id"]), int(item["expert_id"]))
            merged[key] = list(item.get("optimal_config", []))
    return merged


def build_config(
    s_hi_set: Set[Tuple[int, int]],
    projection_optimal: Dict[Tuple[int, int], List[str]],
    start_layer: int,
    num_layers: int,
    num_experts_per_layer: int,
    bits_high: int,
    bits_low: int,
) -> Tuple[Dict[str, int], int, int]:
    config: Dict[str, int] = {}
    applied_projection = 0
    missing_projection = 0

    for layer_id in range(start_layer, num_layers):
        for expert_id in range(num_experts_per_layer):
            expert_key = f"language.model.layers.{layer_id}.mlp.experts.{expert_id}"
            eid = (layer_id, expert_id)

            if eid in s_hi_set:
                optimal = projection_optimal.get(eid)
                if optimal is None:
                    missing_projection += 1
                    for proj in PROJECTIONS:
                        config[f"{expert_key}.{proj}"] = bits_high
                else:
                    optimal_set = set(optimal)
                    for proj in PROJECTIONS:
                        config[f"{expert_key}.{proj}"] = bits_high if proj in optimal_set else bits_low
                    applied_projection += 1
            else:
                for proj in PROJECTIONS:
                    config[f"{expert_key}.{proj}"] = bits_low

    return config, applied_projection, missing_projection


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--swap_result_path", type=str, required=True, help="local_swap_result.json 路径")
    parser.add_argument(
        "--projection_files",
        type=str,
        nargs="+",
        required=True,
        help="一个或多个 projection_shapley*.json 文件路径",
    )
    parser.add_argument("--output_config_path", type=str, required=True, help="输出量化配置 JSON 路径")
    parser.add_argument("--bits_high", type=int, default=4)
    parser.add_argument("--bits_low", type=int, default=2)
    parser.add_argument("--start_layer", type=int, default=1)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_experts_per_layer", type=int, default=64)
    args = parser.parse_args()

    s_hi_set = load_final_s_hi(args.swap_result_path)
    projection_optimal = load_projection_results(args.projection_files)

    config, applied, missing = build_config(
        s_hi_set=s_hi_set,
        projection_optimal=projection_optimal,
        start_layer=args.start_layer,
        num_layers=args.num_layers,
        num_experts_per_layer=args.num_experts_per_layer,
        bits_high=args.bits_high,
        bits_low=args.bits_low,
    )

    os.makedirs(os.path.dirname(args.output_config_path), exist_ok=True)
    with open(args.output_config_path, "w") as f:
        json.dump(config, f, indent=4)

    print("=" * 60)
    print("已生成融合 Phase III 结果的最终量化配置")
    print(f"S_hi 专家数: {len(s_hi_set)}")
    print(f"应用投影层细分的 S_hi 专家数: {applied}")
    print(f"未命中投影层结果、回退全 high-bit 的 S_hi 专家数: {missing}")
    print(f"输出: {args.output_config_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
