#!/usr/bin/env python3
"""
Build a PMQ-style multimodal calibration set from locally available VLMEvalKit
data.

Design goals:
- Match PMQ's small-budget multimodal calibration spirit (default 128 samples).
- Prefer VQA / OCR / document-style samples over generic multimodal chat tasks.
- Avoid benchmarks that are too far from VQA-style calibration by default.

Output format:
{
  "image_path": "/abs/path/to/image.jpg",
  "conversations": [
    {"role": "user", "content": "<image>\\n..."},
    {"role": "assistant", "content": "..."}
  ],
  "source": "DocVQA_VAL"
}
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
import random
import sys
import types
from typing import Dict, List, Optional, Tuple

import pandas as pd


LMUDATA_ROOT = os.path.expanduser("~/LMUData")
IMAGE_ROOT = os.path.join(LMUDATA_ROOT, "images")
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VLMEVAL_ROOT = os.path.join(PROJECT_ROOT, "04_evaluation", "VLMEvalKit")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build PMQ-style multimodal calibration JSONL.")
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "multimodal_calib.jsonl"),
        help="Output JSONL path.",
    )
    parser.add_argument("--total_samples", type=int, default=128, help="Total number of calibration samples.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    parser.add_argument("--lmudata", type=str, default=LMUDATA_ROOT, help="Root directory of local VLMEvalKit TSVs.")
    parser.add_argument(
        "--profile",
        type=str,
        default="pmq_vqa",
        choices=["pmq_vqa", "legacy_mix"],
        help="Sampling profile. pmq_vqa is recommended for PMQ-style DeepSeek-VL2 calibration.",
    )
    return parser.parse_args()


def find_image_path(row: pd.Series, benchmark_name: str, image_root: Optional[str] = None) -> Optional[str]:
    if image_root is None:
        image_root = IMAGE_ROOT
    img_dir = os.path.join(image_root, benchmark_name)

    if "image_path" in row and pd.notna(row["image_path"]):
        ip = str(row["image_path"])
        if os.path.isabs(ip) and os.path.exists(ip):
            return ip
        candidate = os.path.join(img_dir, os.path.basename(ip))
        if os.path.exists(candidate):
            return candidate

    idx = row.get("index", None)
    if idx is not None:
        for ext in (".jpg", ".png", ".jpeg", ".webp"):
            candidate = os.path.join(img_dir, f"{idx}{ext}")
            if os.path.exists(candidate):
                return candidate

    return None


def normalize_answer(answer) -> str:
    if isinstance(answer, list):
        answer = answer[0] if answer else ""
    answer = str(answer)

    if answer.startswith("[") and answer.endswith("]"):
        try:
            parsed = ast.literal_eval(answer)
            if isinstance(parsed, list) and parsed:
                answer = str(parsed[0])
        except (ValueError, SyntaxError):
            pass
    return answer.strip()


def build_question_and_answer(row: pd.Series) -> Tuple[str, str]:
    question = str(row.get("question", "")).strip()
    answer = normalize_answer(row.get("answer", ""))

    options = []
    for opt_key in ("A", "B", "C", "D", "E"):
        if opt_key in row and pd.notna(row[opt_key]):
            options.append(f"{opt_key}. {row[opt_key]}")
    if options:
        question = question + "\n" + "\n".join(options)

    return question, answer


def _load_vlmeval_dataset(name: str):
    if VLMEVAL_ROOT not in sys.path:
        sys.path.insert(0, VLMEVAL_ROOT)

    from vlmeval.dataset import build_dataset

    dataset = build_dataset(name, skip_noimg=False)
    if dataset is None:
        raise ValueError(f"Unsupported VLMEvalKit dataset: {name}")
    return dataset


def _materialize_image(dataset_obj, row: pd.Series) -> Optional[str]:
    try:
        paths = dataset_obj.dump_image(row)
    except Exception:
        return None

    if isinstance(paths, list):
        return paths[0] if paths else None
    return paths


def load_benchmark_samples(
    name: str,
    tsv_path: str,
    n_samples: int,
    seed: int,
    image_root: Optional[str] = None,
) -> List[Dict]:
    dataset_obj = None
    if os.path.exists(tsv_path):
        df = pd.read_csv(tsv_path, sep="\t")
    else:
        print(f"  info: {tsv_path} not found, trying VLMEvalKit auto-download for {name}")
        try:
            dataset_obj = _load_vlmeval_dataset(name)
        except Exception as exc:
            print(f"  warning: failed to initialize {name} via VLMEvalKit ({exc}), skipping")
            return []
        df = dataset_obj.data

    rng = random.Random(seed)

    valid_indices = []
    for i in range(len(df)):
        row = df.iloc[i]
        img_path = _materialize_image(dataset_obj, row) if dataset_obj is not None else find_image_path(row, name, image_root)
        if img_path is not None:
            valid_indices.append(i)

    if not valid_indices:
        print(f"  warning: no valid images found for {name}, skipping")
        return []

    n = min(n_samples, len(valid_indices))
    selected = rng.sample(valid_indices, n)

    samples = []
    for i in selected:
        row = df.iloc[i]
        img_path = _materialize_image(dataset_obj, row) if dataset_obj is not None else find_image_path(row, name, image_root)
        question, answer = build_question_and_answer(row)

        if not question or not answer:
            continue

        samples.append(
            {
                "image_path": img_path,
                "conversations": [
                    {"role": "user", "content": "<image>\n" + question},
                    {"role": "assistant", "content": answer},
                ],
                "source": name,
            }
        )

    print(f"  {name}: sampled {len(samples)}/{len(valid_indices)} valid items")
    return samples


def get_source_profile(profile: str) -> List[Tuple[str, str, float]]:
    if profile == "pmq_vqa":
        # A PMQ-style VQA calibration mix:
        # document / OCR / chart / science-style Q&A, avoiding generic chat tasks.
        return [
            ("DocVQA_VAL", "DocVQA_VAL.tsv", 0.30),
            ("InfoVQA_VAL", "InfoVQA_VAL.tsv", 0.25),
            ("ScienceQA_VAL", "ScienceQA_VAL.tsv", 0.20),
            ("TextVQA_VAL", "TextVQA_VAL.tsv", 0.15),
            ("ChartQA_TEST", "ChartQA_TEST.tsv", 0.10),
        ]

    # Old behavior preserved as an explicit option.
    return [
        ("ScienceQA_VAL", "ScienceQA_VAL.tsv", 0.20),
        ("DocVQA_VAL", "DocVQA_VAL.tsv", 0.25),
        ("InfoVQA_VAL", "InfoVQA_VAL.tsv", 0.20),
        ("MMVet", "MMVet.tsv", 0.20),
        ("POPE", "POPE.tsv", 0.15),
    ]


def main() -> None:
    args = parse_args()

    lmudata_root = args.lmudata
    os.makedirs(lmudata_root, exist_ok=True)
    os.environ["LMUData"] = lmudata_root
    image_root = os.path.join(lmudata_root, "images")
    sources = get_source_profile(args.profile)

    print(f"Building multimodal calibration set: total_samples={args.total_samples}, profile={args.profile}")
    print(f"Local data root: {lmudata_root}")
    print()

    all_samples: List[Dict] = []
    for idx, (name, tsv_file, weight) in enumerate(sources):
        tsv_path = os.path.join(lmudata_root, tsv_file)
        n = max(1, int(args.total_samples * weight))
        samples = load_benchmark_samples(name, tsv_path, n, args.seed + idx, image_root)
        all_samples.extend(samples)

    if len(all_samples) < args.total_samples:
        deficit = args.total_samples - len(all_samples)
        print(f"\nNeed {deficit} more samples, backfilling from DocVQA/InfoVQA...")
        for backfill_name, backfill_file in [("DocVQA_VAL", "DocVQA_VAL.tsv"), ("InfoVQA_VAL", "InfoVQA_VAL.tsv")]:
            if deficit <= 0:
                break
            tsv_path = os.path.join(lmudata_root, backfill_file)
            extra = load_benchmark_samples(backfill_name, tsv_path, deficit, args.seed + 100, image_root)
            take = extra[:deficit]
            all_samples.extend(take)
            deficit -= len(take)

    rng = random.Random(args.seed)
    rng.shuffle(all_samples)
    all_samples = all_samples[: args.total_samples]

    source_counts: Dict[str, int] = {}
    for sample in all_samples:
        src = sample["source"]
        source_counts[src] = source_counts.get(src, 0) + 1

    print(f"\nFinal calibration set: {len(all_samples)} samples")
    for src, cnt in sorted(source_counts.items()):
        print(f"  {src}: {cnt} ({cnt / max(len(all_samples), 1) * 100:.1f}%)")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
