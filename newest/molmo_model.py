import json
import os
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class MolmoModelSpec:
    model_id: str
    hf_path: str
    local_dirname: str
    num_layers: int
    num_experts: int

    @property
    def config_prefix(self) -> str:
        return self.model_id

    @property
    def file_stem(self) -> str:
        return self.model_id


MODEL_SPECS: Dict[str, MolmoModelSpec] = {
    "molmoE-1B-0924": MolmoModelSpec(
        model_id="molmoE-1B-0924",
        hf_path="allenai/MolmoE-1B-0924",
        local_dirname="MolmoE-1B-0924",
        num_layers=16,
        num_experts=64,
    )
}


def _normalize_model_ref(model_ref: Optional[str]) -> str:
    return str(model_ref or "").strip().lower().replace("\\", "/")


def get_model_spec(model_ref: Optional[str]) -> MolmoModelSpec:
    normalized = _normalize_model_ref(model_ref)
    for spec in MODEL_SPECS.values():
        aliases = {
            spec.model_id.lower(),
            spec.hf_path.lower(),
            spec.local_dirname.lower(),
        }
        if normalized in aliases:
            return spec
        if spec.model_id.lower() in normalized or spec.local_dirname.lower() in normalized:
            return spec
    raise KeyError(f"Unsupported MolMoE model reference: {model_ref}")


def resolve_model_path(model_ref: Optional[str], project_root: str) -> str:
    generic_override = os.environ.get("MOLMO_MODEL_PATH")
    if generic_override:
        return generic_override

    spec = get_model_spec(model_ref or "molmoE-1B-0924")
    override_path = os.environ.get(f"{spec.model_id.upper().replace('-', '_')}_MODEL_PATH")
    if override_path:
        return override_path

    raw_ref = str(model_ref or "").strip()
    if raw_ref and (os.path.isdir(raw_ref) or os.path.isfile(raw_ref)):
        return raw_ref

    local_path = os.path.join(project_root, "models", spec.local_dirname)
    if os.path.isdir(local_path):
        return local_path
    return spec.hf_path


def _optional_int(config: dict, key: str) -> Optional[int]:
    value = config.get(key)
    if value is None:
        return None
    return int(value)


def infer_model_config(model_path: str) -> Dict[str, Optional[int]]:
    if not model_path or not os.path.isdir(model_path):
        return {}

    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        return {}

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    return {
        "hidden_size": _optional_int(config, "d_model"),
        "num_layers": _optional_int(config, "n_layers"),
        "num_experts": _optional_int(config, "moe_num_experts"),
        "mlp_hidden_size": _optional_int(config, "mlp_hidden_size"),
        "moe_top_k": _optional_int(config, "moe_top_k"),
    }


def get_attention_proj_names() -> list[str]:
    return ["att_proj", "attn_out"]


def get_output_prefix(model_ref: Optional[str], suffix: str) -> str:
    spec = get_model_spec(model_ref or "molmoE-1B-0924")
    return f"{spec.config_prefix}_{suffix}"


def get_menu_values_filename(model_ref: Optional[str]) -> str:
    spec = get_model_spec(model_ref or "molmoE-1B-0924")
    return f"{spec.file_stem}_menu_values.json"


def get_owen_values_filename(model_ref: Optional[str]) -> str:
    spec = get_model_spec(model_ref or "molmoE-1B-0924")
    return f"{spec.file_stem}_owen_values.json"
