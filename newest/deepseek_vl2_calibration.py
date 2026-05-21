import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union


PathLike = Union[str, os.PathLike[str]]


def build_text_only_autoround_samples(dataset: Iterable[str]) -> List[Dict[str, Any]]:
    """
    Convert plain text calibration samples into the raw conversation format
    expected by AutoRoundCompatibleDataset.

    Each sample becomes a conversation with an empty assistant turn and no
    image path so downstream processing stays on the text-only path.
    """
    raw_samples: List[Dict[str, Any]] = []
    for sample in dataset:
        text = str(sample).strip()
        if not text:
            continue
        raw_samples.append(
            {
                "conversations": [
                    {"role": "user", "content": text},
                    {"role": "assistant", "content": ""},
                ],
                "image_path": None,
            }
        )
    return raw_samples


def normalize_multimodal_image_path(
    image_path: Optional[PathLike],
    source: Optional[str] = None,
    lmu_image_root: Optional[PathLike] = None,
) -> Optional[str]:
    if image_path is None:
        return None

    image_path_str = str(image_path).strip()
    if not image_path_str:
        return None
    if os.path.exists(image_path_str):
        return image_path_str

    image_root = Path(lmu_image_root) if lmu_image_root is not None else Path.home() / "LMUData" / "images"
    norm = image_path_str.replace("\\", "/")
    marker = "/LMUData/images/"
    if marker in norm:
        rel = norm.split(marker, 1)[1]
        candidate = image_root.joinpath(*rel.split("/"))
        if candidate.exists():
            return str(candidate)

    if source:
        candidate = image_root / str(source) / Path(norm).name
        if candidate.exists():
            return str(candidate)

    return image_path_str


def load_multimodal_autoround_samples(
    jsonl_path: PathLike,
    nsamples: int,
    lmu_image_root: Optional[PathLike] = None,
) -> List[Dict[str, Any]]:
    """
    Read multimodal calibration JSONL and keep only samples that still resolve
    to a valid image file plus a non-empty conversation list.
    """
    raw_samples: List[Dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line.strip())
            except Exception:
                continue

            conversations = item.get("conversations", [])
            image_path = normalize_multimodal_image_path(
                item.get("image_path"),
                source=item.get("source"),
                lmu_image_root=lmu_image_root,
            )
            if not conversations or image_path is None or not os.path.exists(image_path):
                continue

            raw_samples.append(
                {
                    "conversations": conversations,
                    "image_path": image_path,
                }
            )
            if len(raw_samples) >= nsamples:
                break
    return raw_samples
