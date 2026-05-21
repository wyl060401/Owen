import json
import os
from pathlib import Path
from typing import Dict, Iterable, Set, Tuple


def checkpoint_path(checkpoint_dir, permutation_index: int) -> Path:
    return Path(checkpoint_dir) / f"perm_{int(permutation_index):06d}.json"


def save_permutation_checkpoint(path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f)
    os.replace(tmp_path, path)


def _load_valid_checkpoint(path: Path, expected_indices: Set[int]) -> dict:
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return {}

    try:
        perm_idx = int(payload["permutation_index"])
        num_samples = int(payload["num_samples"])
        flat_values = payload["flat_values"]
    except (KeyError, TypeError, ValueError):
        return {}

    if perm_idx not in expected_indices:
        return {}
    if num_samples <= 0 or not isinstance(flat_values, dict):
        return {}
    return payload


def load_completed_checkpoints(
    checkpoint_dir,
    expected_indices: Iterable[int],
) -> Tuple[Set[int], Dict[str, float], int]:
    checkpoint_dir = Path(checkpoint_dir)
    expected = {int(i) for i in expected_indices}
    completed = set()
    flat_sums: Dict[str, float] = {}
    total_samples = 0

    if not checkpoint_dir.exists():
        return completed, flat_sums, total_samples

    for path in sorted(checkpoint_dir.glob("perm_*.json")):
        payload = _load_valid_checkpoint(path, expected)
        if not payload:
            continue
        perm_idx = int(payload["permutation_index"])
        if perm_idx in completed:
            continue
        completed.add(perm_idx)
        total_samples += int(payload["num_samples"])
        for key, value in payload["flat_values"].items():
            flat_sums[key] = flat_sums.get(key, 0.0) + float(value)

    return completed, flat_sums, total_samples
