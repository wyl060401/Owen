import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class DeepSeekVL2ModelSpec:
    model_id: str
    hf_path: str
    local_dirname: str
    num_layers: int
    num_experts: int
    device_split: Optional[list]

    @property
    def config_prefix(self) -> str:
        return self.model_id

    @property
    def file_stem(self) -> str:
        return self.local_dirname


MODEL_SPECS: Dict[str, DeepSeekVL2ModelSpec] = {
    "deepseek_vl2_tiny": DeepSeekVL2ModelSpec(
        model_id="deepseek_vl2_tiny",
        hf_path="deepseek-ai/deepseek-vl2-tiny",
        local_dirname="deepseek-vl2-tiny",
        num_layers=12,
        num_experts=64,
        device_split=None,
    ),
    "deepseek_vl2_small": DeepSeekVL2ModelSpec(
        model_id="deepseek_vl2_small",
        hf_path="deepseek-ai/deepseek-vl2-small",
        local_dirname="deepseek-vl2-small",
        num_layers=27,
        num_experts=64,
        device_split=[13, 14],
    ),
    "deepseek_vl2_large": DeepSeekVL2ModelSpec(
        model_id="deepseek_vl2_large",
        hf_path="deepseek-ai/deepseek-vl2",
        local_dirname="deepseek-vl2",
        num_layers=27,
        num_experts=64,
        device_split=[14, 13],
    ),
}


def _normalize_model_ref(model_ref: Optional[str]) -> str:
    return str(model_ref or "").strip().lower().replace("\\", "/")


def get_model_spec(model_ref: Optional[str]) -> DeepSeekVL2ModelSpec:
    normalized = _normalize_model_ref(model_ref)
    for spec in MODEL_SPECS.values():
        if normalized in {spec.model_id, spec.hf_path.lower(), spec.local_dirname.lower()}:
            return spec
        if spec.local_dirname in normalized or spec.model_id in normalized:
            return spec
    raise KeyError(f"Unsupported DeepSeek-VL2 model reference: {model_ref}")


def resolve_model_path(model_ref: Optional[str], project_root: str) -> str:
    generic_override = os.environ.get("DEEPSEEK_VL2_MODEL_PATH")
    if generic_override:
        return generic_override

    spec = get_model_spec(model_ref or "deepseek_vl2_tiny")
    legacy_tiny_override = os.environ.get("DEEPSEEK_VL2_TINY_MODEL_PATH")
    if spec.model_id == "deepseek_vl2_tiny" and legacy_tiny_override:
        return legacy_tiny_override

    override_path = os.environ.get(f"{spec.model_id.upper()}_MODEL_PATH")
    if override_path:
        return override_path

    raw_ref = str(model_ref or "").strip()
    if raw_ref and (os.path.isdir(raw_ref) or os.path.isfile(raw_ref)):
        return raw_ref

    local_path = os.path.join(project_root, "models", spec.local_dirname)
    if os.path.isdir(local_path):
        return local_path
    return spec.hf_path


def infer_model_config(model_path: str) -> Dict[str, int]:
    if not model_path or not os.path.isdir(model_path):
        return {}

    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        return {}

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    lang_cfg = config.get("language_config", {})
    return {
        "hidden_size": int(lang_cfg["hidden_size"]) if "hidden_size" in lang_cfg else None,
        "intermediate_size": int(lang_cfg["intermediate_size"]) if "intermediate_size" in lang_cfg else None,
        "moe_intermediate_size": int(lang_cfg["moe_intermediate_size"]) if "moe_intermediate_size" in lang_cfg else None,
        "num_layers": int(lang_cfg["num_hidden_layers"]) if "num_hidden_layers" in lang_cfg else None,
        "num_experts": int(lang_cfg["n_routed_experts"]) if "n_routed_experts" in lang_cfg else None,
        "n_shared_experts": int(lang_cfg["n_shared_experts"]) if "n_shared_experts" in lang_cfg else None,
        "q_lora_rank": int(lang_cfg["q_lora_rank"]) if lang_cfg.get("q_lora_rank") is not None else None,
    }


def get_attention_proj_names(
    model_cfg: Optional[Dict[str, int]] = None,
    model_ref: Optional[str] = None,
) -> List[str]:
    """
    DeepSeek-VL2-Tiny uses the standard q/k/v/o attention projections.
    DeepSeek-VL2-Small uses the checkpoint's hybrid attention layout:
    q_proj + kv_a_proj_with_mqa + kv_b_proj + o_proj.
    """
    if model_ref is not None:
        spec = get_model_spec(model_ref)
        if spec.model_id == "deepseek_vl2_small":
            return ["q_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"]
        if spec.model_id == "deepseek_vl2_large":
            return ["q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"]

    if model_cfg is None or model_cfg.get("q_lora_rank") is None:
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
    return ["q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"]


def normalize_layer_config_for_model(
    layer_config: Dict[str, Dict[str, int]],
    model_ref: Optional[str],
) -> Tuple[Dict[str, Dict[str, int]], Dict[str, str]]:
    """
    Normalize legacy layer-config attention keys to the checkpoint layout used
    by the target model.

    DeepSeek-VL2-Small uses:
      self_attn.q_proj / kv_a_proj_with_mqa / kv_b_proj / o_proj

    Some older configs still refer to legacy Tiny-style:
      self_attn.k_proj / self_attn.v_proj

    AutoRound rejects those keys on Small, so we rewrite them to the current
    checkpoint naming while preserving any already-correct explicit entries.
    """
    spec = get_model_spec(model_ref or "deepseek_vl2_tiny")
    if spec.model_id != "deepseek_vl2_small":
        return dict(layer_config), {}

    rewrites = {
        ".self_attn.k_proj": ".self_attn.kv_a_proj_with_mqa",
        ".self_attn.v_proj": ".self_attn.kv_b_proj",
    }
    normalized: Dict[str, Dict[str, int]] = {}
    applied_rewrites: Dict[str, str] = {}

    for key, value in layer_config.items():
        new_key = key
        for old_suffix, new_suffix in rewrites.items():
            if old_suffix in new_key:
                new_key = new_key.replace(old_suffix, new_suffix)
                break

        if new_key != key:
            applied_rewrites[key] = new_key

        # Prefer an explicit modern key over an auto-rewritten legacy alias.
        if new_key in normalized and key != new_key:
            continue

        normalized[new_key] = value

    return normalized, applied_rewrites


def get_output_prefix(model_ref: Optional[str], suffix: str) -> str:
    spec = get_model_spec(model_ref or "deepseek_vl2_tiny")
    return f"{spec.config_prefix}_{suffix}"


def get_menu_values_filename(model_ref: Optional[str]) -> str:
    spec = get_model_spec(model_ref or "deepseek_vl2_tiny")
    return f"{spec.file_stem}_menu_values.json"


def get_owen_values_filename(model_ref: Optional[str]) -> str:
    spec = get_model_spec(model_ref or "deepseek_vl2_tiny")
    return f"{spec.file_stem}_owen_values.json"


def build_device_map_for_model(
    model_ref: Optional[str],
    visible_gpu_count: int,
    local_rank: int = 0,
):
    """
    Build a safe device placement for the requested model given the number of
    currently visible CUDA devices.

    Small defaults to a 2-way split, but falls back to single-GPU placement
    when only one device is visible.
    """
    spec = get_model_spec(model_ref or "deepseek_vl2_tiny")
    if spec.device_split is None:
        return f"cuda:{local_rank}"

    if visible_gpu_count <= 1:
        return "cuda:0"

    if visible_gpu_count == 2:
        num_layers_per_gpu = list(spec.device_split)
    else:
        active_gpu_count = min(visible_gpu_count, spec.num_layers)
        base_layers = spec.num_layers // active_gpu_count
        remainder = spec.num_layers % active_gpu_count
        num_layers_per_gpu = [
            base_layers + (1 if gpu_idx < remainder else 0)
            for gpu_idx in range(active_gpu_count)
        ]

    device_map = {}
    final_gpu = len(num_layers_per_gpu) - 1
    num_layers = sum(num_layers_per_gpu)
    layer_cnt = 0
    for gpu_idx, num_layer in enumerate(num_layers_per_gpu):
        for _ in range(num_layer):
            device_map[f"language.model.layers.{layer_cnt}"] = gpu_idx
            layer_cnt += 1
    device_map["vision"] = 0
    device_map["projector"] = 0
    device_map["image_newline"] = 0
    device_map["view_seperator"] = 0
    device_map["language.model.embed_tokens"] = 0
    device_map["language.model.norm"] = final_gpu
    device_map["language.lm_head"] = final_gpu
    device_map[f"language.model.layers.{num_layers - 1}"] = final_gpu
    return device_map
