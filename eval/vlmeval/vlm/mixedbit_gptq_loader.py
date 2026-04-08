import json
import math
import os
from typing import Dict, Tuple

import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers import AutoConfig

from .DeepSeek_VL2.modeling_deepseek_vl_v2 import DeepseekVLV2ForCausalLM


class MixedBitQuantLinear(nn.Module):
    def __init__(self, bits: int, groupsize: int, infeatures: int, outfeatures: int, bias: bool):
        super().__init__()
        if bits not in [2, 3, 4, 8]:
            raise NotImplementedError("Only 2, 3, 4, 8 bits are supported.")
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.bits = bits
        self.maxq = 2 ** bits - 1
        self.groupsize = groupsize if groupsize != -1 else infeatures

        self.register_buffer("qweight", torch.zeros((infeatures // 32 * bits, outfeatures), dtype=torch.int32))
        self.register_buffer(
            "qzeros",
            torch.zeros((math.ceil(infeatures / self.groupsize), math.ceil(outfeatures / 32) * bits), dtype=torch.int32),
        )
        self.register_buffer("scales", torch.zeros((math.ceil(infeatures / self.groupsize), outfeatures), dtype=torch.float16))
        self.register_buffer("g_idx", torch.tensor([i // self.groupsize for i in range(infeatures)], dtype=torch.int32))
        self.register_buffer("_cached_weight", torch.empty(0, dtype=torch.float16), persistent=False)
        if bias:
            self.register_buffer("bias", torch.zeros((outfeatures,), dtype=torch.float16))
        else:
            self.bias = None

    def _expand_packed_generic(self, packed: torch.Tensor, width: int) -> torch.Tensor:
        words_per_block = self.bits
        values_per_block = 32
        n_blocks = packed.shape[0] // words_per_block
        packed_u32 = packed.to(torch.int64) & 0xFFFFFFFF
        packed_u32 = packed_u32.view(n_blocks, words_per_block, packed.shape[1])

        values = []
        for idx in range(values_per_block):
            bit_pos = idx * self.bits
            word_idx = bit_pos // 32
            offset = bit_pos % 32

            value = packed_u32[:, word_idx, :] >> offset
            if offset + self.bits > 32:
                spill = offset + self.bits - 32
                value = value | (packed_u32[:, word_idx + 1, :] << (32 - offset))
                value = value & ((1 << self.bits) - 1)
            else:
                value = value & self.maxq
            values.append(value)

        unpacked = torch.stack(values, dim=1).reshape(n_blocks * values_per_block, packed.shape[1])
        return unpacked[:width].to(torch.int32)

    def _expand_packed(self, packed: torch.Tensor, width: int) -> torch.Tensor:
        if self.bits == 3:
            return self._expand_packed_generic(packed, width)
        pack = 32 // self.bits
        shifts = torch.arange(pack, device=packed.device, dtype=torch.int32) * self.bits
        unpacked = (packed.unsqueeze(1) >> shifts.view(1, -1, 1)) & self.maxq
        unpacked = unpacked.reshape(packed.shape[0] * pack, packed.shape[1])
        return unpacked[:width]

    def _expand_packed_zeros(self, packed: torch.Tensor, width: int) -> torch.Tensor:
        if self.bits == 3:
            unpacked = self._expand_packed_generic(
                packed.transpose(0, 1).contiguous(),
                width,
            ).transpose(0, 1).contiguous()
            return unpacked[:, :width] + 1
        pack = 32 // self.bits
        shifts = torch.arange(pack, device=packed.device, dtype=torch.int32) * self.bits
        unpacked = (packed.unsqueeze(2) >> shifts.view(1, 1, -1)) & self.maxq
        unpacked = unpacked.reshape(packed.shape[0], packed.shape[1] * pack)
        return unpacked[:, :width] + 1

    def _dequantize_weight(self, dtype: torch.dtype) -> torch.Tensor:
        intweight = self._expand_packed(self.qweight, self.infeatures).to(dtype)
        zeros = self._expand_packed_zeros(self.qzeros, self.outfeatures).to(dtype)
        g_idx = self.g_idx.to(torch.long)
        scales = self.scales.to(dtype)
        weight = (intweight - zeros[g_idx]) * scales[g_idx]
        return weight

    @torch.no_grad()
    def _get_cached_weight(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if (
            self._cached_weight.numel() == 0
            or self._cached_weight.device != device
            or self._cached_weight.dtype != dtype
        ):
            self._cached_weight = self._dequantize_weight(dtype).to(device=device, dtype=dtype)
        return self._cached_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_shape = x.shape[:-1] + (self.outfeatures,)
        x_flat = x.reshape(-1, x.shape[-1])
        compute_dtype = x_flat.dtype if x_flat.dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.float16
        weight = self._get_cached_weight(compute_dtype, x_flat.device)
        out = torch.matmul(x_flat.to(compute_dtype), weight)
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)
        return out.reshape(out_shape)


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_parent_and_attr(model: nn.Module, module_name: str) -> Tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def _resolve_bits_and_group_size(cfg, default_bits: int, default_group_size: int) -> Tuple[int, int]:
    if isinstance(cfg, dict):
        bits = int(cfg.get("bits", default_bits))
        group_size = int(cfg.get("group_size", default_group_size))
    elif isinstance(cfg, int):
        bits = int(cfg)
        group_size = default_group_size
    else:
        bits = default_bits
        group_size = default_group_size
    return bits, group_size


def _replace_quant_modules(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
    per_layer_cfg: dict,
    default_bits: int,
    default_group_size: int,
):
    named_modules = dict(model.named_modules())
    quant_module_names = sorted({key.rsplit(".", 1)[0] for key in state_dict if key.endswith(".qweight")})

    for module_name in quant_module_names:
        original = named_modules.get(module_name)
        if original is None or not hasattr(original, "in_features") or not hasattr(original, "out_features"):
            continue
        cfg = per_layer_cfg.get(module_name, {})
        bits, group_size = _resolve_bits_and_group_size(cfg, default_bits, default_group_size)
        quant_linear = MixedBitQuantLinear(
            bits=bits,
            groupsize=group_size,
            infeatures=original.in_features,
            outfeatures=original.out_features,
            bias=getattr(original, "bias", None) is not None,
        )
        parent, attr = _get_parent_and_attr(model, module_name)
        setattr(parent, attr, quant_linear)


def _dequantize_quant_modules_to_linear(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
    per_layer_cfg: dict,
    default_bits: int,
    default_group_size: int,
    target_dtype: torch.dtype,
):
    named_modules = dict(model.named_modules())
    quant_module_names = sorted({key.rsplit(".", 1)[0] for key in state_dict if key.endswith(".qweight")})

    for module_name in quant_module_names:
        original = named_modules.get(module_name)
        if original is None or not hasattr(original, "in_features") or not hasattr(original, "out_features"):
            continue

        cfg = per_layer_cfg.get(module_name, {})
        bits, group_size = _resolve_bits_and_group_size(cfg, default_bits, default_group_size)
        tmp_quant = MixedBitQuantLinear(
            bits=bits,
            groupsize=group_size,
            infeatures=original.in_features,
            outfeatures=original.out_features,
            bias=getattr(original, "bias", None) is not None,
        )

        for suffix in ("qweight", "qzeros", "scales", "g_idx", "bias"):
            key = f"{module_name}.{suffix}"
            if key in state_dict and hasattr(tmp_quant, suffix):
                getattr(tmp_quant, suffix).copy_(state_dict[key])

        linear = nn.Linear(
            original.in_features,
            original.out_features,
            bias=getattr(original, "bias", None) is not None,
            device="cpu",
            dtype=target_dtype,
        )
        with torch.no_grad():
            linear.weight.copy_(tmp_quant._dequantize_weight(target_dtype).t().contiguous())
            if linear.bias is not None and tmp_quant.bias is not None:
                linear.bias.copy_(tmp_quant.bias.to(dtype=target_dtype))

        parent, attr = _get_parent_and_attr(model, module_name)
        setattr(parent, attr, linear)

        for suffix in ("qweight", "qzeros", "scales", "g_idx", "bias"):
            state_dict.pop(f"{module_name}.{suffix}", None)


def load_mixedbit_quantized_deepseek_vl2(
    model_path: str,
    device: str = "cuda:0",
    torch_dtype: torch.dtype = torch.bfloat16,
    dequantize_to_linear: bool = True,
):
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if hasattr(config, "quantization_config"):
        try:
            delattr(config, "quantization_config")
        except Exception:
            config.__dict__.pop("quantization_config", None)

    quant_cfg_path = os.path.join(model_path, "quant_config_original.json")
    per_layer_cfg = _load_json(quant_cfg_path) if os.path.exists(quant_cfg_path) else {}

    global_quant_cfg = getattr(config, "quantization_config", None) or {}
    default_bits = int(global_quant_cfg.get("bits", 4)) if isinstance(global_quant_cfg, dict) else 4
    default_group_size = int(global_quant_cfg.get("group_size", 128)) if isinstance(global_quant_cfg, dict) else 128

    model = DeepseekVLV2ForCausalLM(config)
    state_dict = load_file(os.path.join(model_path, "model.safetensors"), device="cpu")

    if dequantize_to_linear:
        _dequantize_quant_modules_to_linear(
            model,
            state_dict,
            per_layer_cfg,
            default_bits,
            default_group_size,
            torch_dtype,
        )
    else:
        _replace_quant_modules(model, state_dict, per_layer_cfg, default_bits, default_group_size)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    missing = [x for x in missing if not x.endswith(("qweight", "qzeros", "scales", "g_idx", "bias"))]
    if missing:
        print(f"[MixedBitLoader] missing keys: {len(missing)}")
    if unexpected:
        print(f"[MixedBitLoader] unexpected keys: {len(unexpected)}")

    model = model.to(torch_dtype).to(device).eval()
    for param in model.parameters():
        param.requires_grad = False
    return model
