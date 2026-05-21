import argparse
import json
import math
from collections import Counter, defaultdict


PROJ_NAMES = ("w1", "w2", "w3")
FULL_MENU_KEY = "w1,w2,w3"


def parse_expert_key(expert_key):
    parts = expert_key.split(".")
    layer_id = int(parts[2])
    expert_id = int(parts[5])
    return layer_id, expert_id


def build_global_expert_config(menu_values, avg_bits, low_bit=2, high_bit=4, score_key=FULL_MENU_KEY):
    if high_bit <= low_bit:
        raise ValueError("high_bit must be larger than low_bit")

    expert_scores = []
    for expert_key, menus in menu_values.items():
        score = float(menus.get(score_key, 0.0))
        layer_id, expert_id = parse_expert_key(expert_key)
        expert_scores.append((expert_key, layer_id, expert_id, score))

    n_experts = len(expert_scores)
    if n_experts == 0:
        return {}, {}

    high_ratio = (avg_bits - low_bit) / (high_bit - low_bit)
    high_ratio = max(0.0, min(1.0, high_ratio))
    n_high = int(round(high_ratio * n_experts))

    ranked = sorted(expert_scores, key=lambda x: x[3], reverse=True)
    high_set = {expert_key for expert_key, _, _, _ in ranked[:n_high]}

    config = {}
    layer_counts = defaultdict(lambda: {"n_high": 0, "n_low": 0})

    for expert_key, layer_id, expert_id, score in ranked:
        bits = high_bit if expert_key in high_set else low_bit
        if bits == high_bit:
            layer_counts[layer_id]["n_high"] += 1
        else:
            layer_counts[layer_id]["n_low"] += 1
        for proj_name in PROJ_NAMES:
            module_name = f"model.layers.{layer_id}.block_sparse_moe.experts.{expert_id}.{proj_name}"
            config[module_name] = bits

    summary = {
        "mode": "global_expert_level",
        "score_key": score_key,
        "avg_bits_target": avg_bits,
        "avg_bits_realized": low_bit + (n_high / n_experts) * (high_bit - low_bit),
        "low_bit": low_bit,
        "high_bit": high_bit,
        "high_ratio": n_high / n_experts,
        "num_experts": n_experts,
        "num_high_experts": n_high,
        "num_low_experts": n_experts - n_high,
        "layer_counts": {str(k): v for k, v in sorted(layer_counts.items())},
        "top10_experts": [
            {"expert_key": expert_key, "score": score}
            for expert_key, _, _, score in ranked[:10]
        ],
        "bottom10_experts": [
            {"expert_key": expert_key, "score": score}
            for expert_key, _, _, score in ranked[-10:]
        ],
    }
    return config, summary


def main():
    parser = argparse.ArgumentParser(
        description="Build a global expert-level 2/4-bit config from existing menu values."
    )
    parser.add_argument("--menu_values_path", type=str, required=True)
    parser.add_argument("--avg_bits", type=float, required=True)
    parser.add_argument("--output_config_path", type=str, required=True)
    parser.add_argument("--output_summary_path", type=str, default=None)
    parser.add_argument("--low_bit", type=int, default=2)
    parser.add_argument("--high_bit", type=int, default=4)
    parser.add_argument("--score_key", type=str, default=FULL_MENU_KEY,
                        help="Menu key used as the expert-level upgrade score. Default: w1,w2,w3")
    args = parser.parse_args()

    with open(args.menu_values_path, "r", encoding="utf-8") as f:
        menu_values = json.load(f)

    config, summary = build_global_expert_config(
        menu_values=menu_values,
        avg_bits=args.avg_bits,
        low_bit=args.low_bit,
        high_bit=args.high_bit,
        score_key=args.score_key,
    )

    with open(args.output_config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    if args.output_summary_path:
        with open(args.output_summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    bit_counter = Counter(config.values())
    print("=" * 60)
    print("Global expert-level config generated")
    print(f"input: {args.menu_values_path}")
    print(f"output: {args.output_config_path}")
    print(f"score_key: {args.score_key}")
    print(f"avg_bits_target: {args.avg_bits}")
    print(f"avg_bits_realized: {summary['avg_bits_realized']:.6f}")
    print(f"bits_count: {dict(sorted(bit_counter.items()))}")
    print(f"num_high_experts: {summary['num_high_experts']} / {summary['num_experts']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
