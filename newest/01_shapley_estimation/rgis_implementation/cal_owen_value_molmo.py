import argparse
import gc
import json
import math
import os
import sys
import time
from typing import Dict, List, Tuple

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer

from menu_checkpoint import checkpoint_path, load_completed_checkpoints, save_permutation_checkpoint
from rgis_shapley_molmo import RGISShapleyEstimatorMolmo

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_VLM_DIR = os.path.join(_PROJECT_ROOT, "04_evaluation/VLMEvalKit")
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _VLM_DIR not in sys.path:
    sys.path.insert(0, _VLM_DIR)

from cal_owen_value import build_checkpoint_run_id, eval_ppl, get_c4
from deepseek_vl2_calibration import load_multimodal_autoround_samples
from molmo_model import (
    get_menu_values_filename,
    get_model_spec,
    get_owen_values_filename,
    infer_model_config,
    resolve_model_path,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase I: MolMoE menu value estimation")
    parser.add_argument("--model", type=str, default="molmoE-1B-0924")
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=128)
    parser.add_argument("--monte_carlo_samples", type=int, default=20)
    parser.add_argument("--tau_temp", type=float, default=0.7)
    parser.add_argument("--lambda_mix", type=float, default=0.1)
    parser.add_argument("--use_self_normalized", action="store_true")
    parser.add_argument("--early_stop_threshold", type=float, default=0.5)
    parser.add_argument("--max_expert_fraction_per_permutation", type=float, default=1.0)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument(
        "--image_batch_size",
        type=int,
        default=4,
        help="Batch size for raw multimodal image/embedding preprocessing before cached eval batches are available.",
    )
    parser.add_argument(
        "--cache_eval_batches_on_gpu",
        action="store_true",
        help="Pre-collate MolMoE calibration batches on GPU to avoid repeated CPU/GPU transfers in value_fn.",
    )
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--disable_checkpoints", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--gpus_per_worker", type=int, default=1)
    parser.add_argument("--cache_dir", type=str, default=os.path.join(os.path.expanduser("~"), ".cache/huggingface"))
    parser.add_argument("--output_dir", type=str, default=os.path.join(_PROJECT_ROOT, "shapley_results"))
    parser.add_argument("--start_layer", type=int, default=0)
    parser.add_argument("--calib_data", type=str, default="multimodal", choices=["multimodal", "c4"])
    parser.add_argument("--calib_data_path", type=str, default=None)
    parser.add_argument("--debug_loss_samples", type=int, default=0)
    parser.add_argument("--min_loss_tokens", type=int, default=0)
    parser.add_argument(
        "--fp32_loss_eval",
        action="store_true",
        help="Recompute NLL from fp32 logits for diagnostics. Slower than using model loss.",
    )
    parser.add_argument(
        "--use_cached_full_forward",
        action="store_true",
        help="Cache MolMoE prefix/current-layer attention while still running suffix/loss.",
    )
    parser.add_argument(
        "--verify_cached_full_forward",
        action="store_true",
        help="Before cached estimation, compare cached full-forward value against the normal value_fn.",
    )
    parser.add_argument("--cached_full_forward_tolerance", type=float, default=0.05)
    return parser


def build_molmo_estimator(args, num_experts: int, device: torch.device) -> RGISShapleyEstimatorMolmo:
    estimator = RGISShapleyEstimatorMolmo(
        num_experts=num_experts,
        tau=args.tau_temp,
        lambda_mix=args.lambda_mix,
        use_self_normalized=args.use_self_normalized,
        early_stop_threshold=args.early_stop_threshold,
        bits_high=4,
        bits_low=2,
        device=device,
    )
    estimator.max_expert_fraction_per_permutation = args.max_expert_fraction_per_permutation
    return estimator


def _sanitize_multimodal_text(content: str) -> str:
    return str(content or "").replace("<image>", "").strip()


def split_molmo_multimodal_sample(sample: Dict[str, object]) -> Tuple[str, str]:
    user_parts: List[str] = []
    assistant_parts: List[str] = []
    for turn in sample.get("conversations", []):
        role = str(turn.get("role", "")).lower()
        content = _sanitize_multimodal_text(turn.get("content", ""))
        if not content:
            continue
        if "assistant" in role:
            assistant_parts.append(content)
        else:
            user_parts.append(content)

    prompt = "\n".join(user_parts).strip()
    answer = "\n".join(assistant_parts).strip()
    if not prompt:
        raise ValueError("multimodal sample is missing a user prompt")
    return prompt, answer


def _prepare_molmo_batch(processor, image: Image.Image, text: str) -> Dict[str, torch.Tensor]:
    processed = processor.process(
        images=[image],
        text=text,
        images_kwargs={"max_crops": 12},
    )
    return {key: value.unsqueeze(0).cpu() for key, value in processed.items()}


def _truncate_molmo_batch(
    batch: Dict[str, torch.Tensor],
    seqlen: int,
    prompt_len: int,
) -> Tuple[Dict[str, torch.Tensor], int]:
    seq_len = int(batch["input_ids"].shape[1])
    if seq_len <= seqlen:
        return batch, prompt_len

    offset = seq_len - seqlen
    truncated = dict(batch)
    for key in ("input_ids", "attention_mask", "labels"):
        if key in truncated:
            truncated[key] = truncated[key][:, -seqlen:].clone()

    if "image_input_idx" in truncated:
        image_input_idx = truncated["image_input_idx"].clone()
        valid = image_input_idx >= 0
        image_input_idx[valid] -= offset
        image_input_idx[image_input_idx < 0] = -1
        image_input_idx[image_input_idx >= seqlen] = -1
        truncated["image_input_idx"] = image_input_idx

    return truncated, max(prompt_len - offset, 0)


def _mask_trailing_assistant_marker(labels: torch.Tensor, tokenizer, max_suffix_tokens: int = 8) -> torch.Tensor:
    """Mask Molmo processor's trailing generation marker from answer labels.

    `processor.process(text=prompt + answer)` can append a final "Assistant:"
    marker. That marker is part of the prompt template, not the target answer,
    and should not contribute to calibration loss.
    """
    if labels.shape[0] != 1:
        raise ValueError("_mask_trailing_assistant_marker expects a single-sample label tensor")

    masked = labels.clone()
    if tokenizer is None or not hasattr(tokenizer, "decode"):
        return masked
    valid_positions = torch.nonzero(masked[0] != -100, as_tuple=False).flatten()
    if valid_positions.numel() == 0:
        return masked

    max_suffix = min(max_suffix_tokens, int(valid_positions.numel()))
    for suffix_len in range(1, max_suffix + 1):
        suffix_positions = valid_positions[-suffix_len:]
        suffix_ids = masked[0, suffix_positions].tolist()
        decoded = tokenizer.decode(suffix_ids, skip_special_tokens=False).strip()
        if decoded in {"Assistant:", "Assistant"}:
            masked[0, suffix_positions] = -100
            break
    return masked


def _encode_without_special_tokens(tokenizer, text: str) -> List[int]:
    if tokenizer is None:
        return []
    if hasattr(tokenizer, "encode"):
        return tokenizer.encode(text, add_special_tokens=False)
    encoded = tokenizer(text, add_special_tokens=False)
    input_ids = encoded["input_ids"] if isinstance(encoded, dict) else getattr(encoded, "input_ids", [])
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    return list(input_ids)


def _find_last_subsequence(values: List[int], pattern: List[int]) -> int:
    if not pattern or len(pattern) > len(values):
        return -1
    for start in range(len(values) - len(pattern), -1, -1):
        if values[start:start + len(pattern)] == pattern:
            return start
    return -1


def _mask_to_answer_span(
    input_ids: torch.Tensor,
    answer_text: str,
    tokenizer,
) -> torch.Tensor:
    """Keep loss only on the answer token span when it can be identified."""
    labels = input_ids.clone()
    labels[:] = -100
    if not answer_text:
        return labels

    flat_ids = input_ids[0].tolist()
    answer_variants = [
        answer_text,
        "\n" + answer_text,
        " " + answer_text,
        ":" + answer_text,
        ": " + answer_text,
    ]

    candidates = []
    seen = set()
    for variant in answer_variants:
        token_ids = _encode_without_special_tokens(tokenizer, variant)
        if not token_ids:
            continue
        key = tuple(token_ids)
        if key not in seen:
            candidates.append(token_ids)
            seen.add(key)

    # Prefer longer matches so a short answer like "A" does not match an
    # unrelated earlier token if the tokenizer gives a longer contextual form.
    candidates.sort(key=len, reverse=True)
    for token_ids in candidates:
        start = _find_last_subsequence(flat_ids, token_ids)
        if start >= 0:
            end = start + len(token_ids)
            labels[:, start:end] = input_ids[:, start:end]
            return labels

    return labels


def _pad_tensor_to_shape(tensor: torch.Tensor, target_shape: Tuple[int, ...], pad_value: float) -> torch.Tensor:
    if tuple(tensor.shape) == tuple(target_shape):
        return tensor

    padded = tensor.new_full(target_shape, pad_value)
    slices = tuple(slice(0, dim) for dim in tensor.shape)
    padded[slices] = tensor
    return padded


def collate_molmo_batches(samples: List[Dict[str, torch.Tensor]], device) -> Dict[str, torch.Tensor]:
    if not samples:
        raise ValueError("samples must be non-empty")

    pad_values = {
        "input_ids": -1,
        "inputs_embeds": 0,
        "attention_mask": 0,
        "images": 0,
        "image_masks": 0,
        "image_input_idx": -1,
        "labels": -100,
    }

    batch = {}
    keys = samples[0].keys()
    for key in keys:
        target_shape = tuple(max(sample[key].shape[dim] for sample in samples) for dim in range(samples[0][key].dim()))
        padded = [
            _pad_tensor_to_shape(sample[key], target_shape, pad_values.get(key, 0))
            for sample in samples
        ]
        batch[key] = torch.cat(padded, dim=0).to(device)
    return batch


def cast_molmo_batch_floats(batch: Dict[str, torch.Tensor], dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    casted = {}
    for key, value in batch.items():
        if torch.is_floating_point(value):
            casted[key] = value.to(dtype=dtype)
        else:
            casted[key] = value
    return casted


def _build_molmo_inputs_embeds(model, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Precompute embeddings after image-feature insertion, before emb_drop/norm.

    MolmoModel.forward applies embedding dropout and optional normalization after
    image features are inserted. We cache the pre-dropout tensor so the later
    inputs_embeds path remains numerically equivalent to the normal image path.
    """
    if getattr(model.config, "use_cls_feature", False):
        raise NotImplementedError("inputs_embeds cache does not support MolMoE use_cls_feature=True")

    input_ids = batch["input_ids"]
    attention_mask = batch.get("attention_mask")
    input_ids_for_embed = input_ids * (input_ids != -1).to(input_ids.dtype)
    x = model.model.transformer.wte(input_ids_for_embed)

    images = batch.get("images")
    if images is not None:
        image_features, _ = model.model.vision_backbone(images, batch["image_masks"])
        batch_size, num_image, num_patch = image_features.shape[:3]
        image_input_idx = batch["image_input_idx"]
        assert image_input_idx.shape == (batch_size, num_image, num_patch)

        image_features = image_features.view(batch_size, num_image * num_patch, -1).to(x.device)
        image_input_idx = image_input_idx.view(batch_size, num_image * num_patch)
        valid = image_input_idx >= 0
        batch_idx = torch.arange(batch_size, device=x.device)
        batch_idx = torch.tile(batch_idx[:, None], [1, image_features.shape[1]])
        x[batch_idx[valid], image_input_idx[valid]] += image_features[valid]

    return {
        "input_ids": input_ids.detach().cpu(),
        "inputs_embeds": x.detach().cpu(),
        "attention_mask": attention_mask.detach().cpu() if attention_mask is not None else None,
        "labels": batch["labels"].detach().cpu(),
    }


def precompute_molmo_inputs_embeds(model, inputs, device, batch_size: int = 4):
    if not inputs or not isinstance(inputs[0], dict) or "inputs_embeds" in inputs[0]:
        return inputs

    model.eval()
    model_dtype = next(model.parameters()).dtype
    cached = []
    with torch.no_grad():
        for batch_start in range(0, len(inputs), batch_size):
            batch_end = min(batch_start + batch_size, len(inputs))
            batch = cast_molmo_batch_floats(
                collate_molmo_batches(inputs[batch_start:batch_end], device),
                model_dtype,
            )
            cached_batch = _build_molmo_inputs_embeds(model, batch)
            for row in range(batch_end - batch_start):
                sample = {
                    key: value[row:row + 1].contiguous()
                    for key, value in cached_batch.items()
                    if value is not None
                }
                cached.append(sample)
            del batch, cached_batch
            torch.cuda.empty_cache()
    return cached


def prepare_molmo_eval_batches(inputs, device, model_dtype: torch.dtype, batch_size: int = 4):
    """Pre-collate calibration samples into reusable GPU batches.

    This keeps the numerical path identical to eval_ppl_molmo's regular
    batching, but avoids rebuilding/padding/transferring the same tensors for
    every expert/menu value evaluation.
    """
    if not inputs or not isinstance(inputs[0], dict):
        return inputs

    eval_batches = []
    for batch_start in range(0, len(inputs), batch_size):
        batch_end = min(batch_start + batch_size, len(inputs))
        batch = cast_molmo_batch_floats(
            collate_molmo_batches(inputs[batch_start:batch_end], device),
            model_dtype,
        )
        batch["__molmo_eval_batch__"] = True
        eval_batches.append(batch)
    return eval_batches


def _is_molmo_eval_batches(inputs) -> bool:
    return (
        bool(inputs)
        and isinstance(inputs[0], dict)
        and inputs[0].get("__molmo_eval_batch__", False) is True
    )


def get_multimodal_calib_molmo(
    nsamples: int,
    seed: int,
    processor,
    seqlen: int,
    data_path: str = None,
    min_loss_tokens: int = 0,
):
    if data_path is None:
        data_path = os.path.join(_PROJECT_ROOT, "data/multimodal_calib.jsonl")

    raw_samples = load_multimodal_autoround_samples(data_path, nsamples=max(nsamples * 20, nsamples + 64))
    if not raw_samples:
        raise RuntimeError(f"No usable multimodal calibration samples found in {data_path}")

    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(raw_samples), generator=generator).tolist()

    batches = []
    skipped = 0
    skipped_short_loss = 0
    for index in order:
        sample = raw_samples[index]
        try:
            prompt_text, answer_text = split_molmo_multimodal_sample(sample)
            image = Image.open(sample["image_path"]).convert("RGB")
            full_text = prompt_text if not answer_text else f"{prompt_text}\n{answer_text}"
            full_batch = _prepare_molmo_batch(processor, image, full_text)
            prompt_batch = _prepare_molmo_batch(processor, image, prompt_text)
        except Exception:
            skipped += 1
            continue

        prompt_len = int(prompt_batch["input_ids"].shape[1])
        labels = _mask_to_answer_span(full_batch["input_ids"], answer_text, getattr(processor, "tokenizer", None))
        if answer_text:
            # Fallback for lightweight test processors or tokenizer mismatches.
            if int((labels != -100).sum().item()) == 0:
                labels = full_batch["input_ids"].clone()
                labels[labels < 0] = -100
                labels[:, :prompt_len] = -100
        labels = _mask_trailing_assistant_marker(labels, getattr(processor, "tokenizer", None))

        full_batch["labels"] = labels
        full_batch, prompt_len = _truncate_molmo_batch(full_batch, seqlen, prompt_len)
        loss_tokens = int((full_batch["labels"][:, 1:] != -100).sum().item())
        if loss_tokens < min_loss_tokens:
            skipped_short_loss += 1
            continue
        batches.append(full_batch)
        if len(batches) >= nsamples:
            break

    if not batches:
        raise RuntimeError(
            "MolMoE multimodal calibration produced no valid samples from "
            f"{data_path}; skipped_errors={skipped}, skipped_short_loss={skipped_short_loss}. "
            "Increase --seqlen or lower --min_loss_tokens."
        )
    if len(batches) < nsamples:
        print(
            f"Warning: requested {nsamples} MolMoE calibration samples, got {len(batches)}; "
            f"skipped_errors={skipped}, skipped_short_loss={skipped_short_loss}. "
            "Consider increasing --seqlen."
        )
    return batches


def count_activation_frequency_molmo(model, inputs, num_layers, num_experts_per_layer, start_layer, device, batch_size=4):
    total_experts = (num_layers - start_layer) * num_experts_per_layer
    activation_counts = torch.zeros(total_experts, device="cpu")
    total_tokens = 0
    num_samples = len(inputs)

    moe_blocks = []
    for layer_id, block in enumerate(model.model.transformer.blocks):
        mlp = getattr(block, "mlp", None)
        if hasattr(mlp, "get_layer_stats") and hasattr(mlp, "get_expert_activation"):
            mlp.layer_id = layer_id
            moe_blocks.append(mlp)

    is_multimodal = isinstance(inputs[0], dict)
    model_dtype = next(model.parameters()).dtype

    for batch_start in range(0, num_samples, batch_size):
        batch_end = min(batch_start + batch_size, num_samples)
        if is_multimodal:
            batch = cast_molmo_batch_floats(
                collate_molmo_batches(inputs[batch_start:batch_end], device),
                model_dtype,
            )
            batch_inputs = batch["input_ids"]
            total_tokens += int((batch_inputs != -1).sum().item())
        else:
            batch_inputs = torch.cat([inputs[j][0] for j in range(batch_start, batch_end)], dim=0).to(device)
            total_tokens += batch_inputs.numel()

        for block in moe_blocks:
            block.expert_activation_counts = {f"expert_{i}": 0 for i in range(block.num_experts)}

        with torch.no_grad():
            if is_multimodal:
                model(
                    input_ids=batch["input_ids"],
                    images=batch["images"],
                    image_masks=batch["image_masks"],
                    image_input_idx=batch["image_input_idx"],
                )
            else:
                model(input_ids=batch_inputs)

        for block in moe_blocks:
            layer_id = getattr(block, "layer_id", None)
            if layer_id is None or layer_id < start_layer:
                continue
            layer_offset = (layer_id - start_layer) * num_experts_per_layer
            counts = block.get_expert_activation()
            for expert_id in range(min(block.num_experts, num_experts_per_layer)):
                activation_counts[layer_offset + expert_id] += counts[f"expert_{expert_id}"]

    if total_tokens > 0:
        activation_counts /= total_tokens
    activation_counts = activation_counts.clamp(min=1e-8)
    activation_counts /= activation_counts.sum()
    return activation_counts


def eval_nll_molmo(model, inputs, device, batch_size: int = 4, fp32_loss_eval: bool = False) -> float:
    if not inputs or not isinstance(inputs[0], dict):
        ppl = eval_ppl(model, inputs, device, batch_size=batch_size)
        return math.log(max(ppl, 1e-12))

    model.eval()
    total_loss = 0.0
    total_tokens = 0
    model_dtype = next(model.parameters()).dtype

    with torch.no_grad():
        if _is_molmo_eval_batches(inputs):
            batch_iter = inputs
        else:
            batch_iter = []
            for batch_start in range(0, len(inputs), batch_size):
                batch_end = min(batch_start + batch_size, len(inputs))
                batch_iter.append(
                    cast_molmo_batch_floats(
                        collate_molmo_batches(inputs[batch_start:batch_end], device),
                        model_dtype,
                    )
                )

        for batch in batch_iter:
            if "inputs_embeds" in batch:
                outputs = model(
                    input_ids=batch["input_ids"],
                    inputs_embeds=batch["inputs_embeds"],
                    attention_mask=batch.get("attention_mask"),
                    labels=batch["labels"],
                )
            else:
                outputs = model(
                    input_ids=batch["input_ids"],
                    images=batch["images"],
                    image_masks=batch["image_masks"],
                    image_input_idx=batch["image_input_idx"],
                    labels=batch["labels"],
                )
            n_tokens = int((batch["labels"][:, 1:] != -100).sum().item())
            if n_tokens > 0:
                if fp32_loss_eval:
                    shifted_logits = outputs.logits[:, :-1, :].contiguous().float()
                    shifted_labels = batch["labels"][:, 1:].contiguous()
                    loss_sum = F.cross_entropy(
                        shifted_logits.view(-1, shifted_logits.size(-1)),
                        shifted_labels.view(-1),
                        ignore_index=-100,
                        reduction="sum",
                    )
                    total_loss += float(loss_sum.item())
                else:
                    total_loss += float(outputs.loss.item()) * n_tokens
                total_tokens += n_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    return avg_loss


def eval_ppl_molmo(model, inputs, device, batch_size: int = 4) -> float:
    return math.exp(min(eval_nll_molmo(model, inputs, device, batch_size=batch_size), 100))


def debug_molmo_loss_samples(model, inputs, tokenizer, device, max_samples: int = 3) -> None:
    if not inputs or not isinstance(inputs[0], dict) or max_samples <= 0:
        return

    model.eval()
    model_dtype = next(model.parameters()).dtype
    sample_count = min(max_samples, len(inputs))
    print("\n" + "=" * 60)
    print(f"MolMoE loss debug: first {sample_count} sample(s)")
    print("=" * 60)

    with torch.no_grad():
        for index in range(sample_count):
            batch = cast_molmo_batch_floats(collate_molmo_batches([inputs[index]], device), model_dtype)
            if "inputs_embeds" in batch:
                outputs = model(
                    input_ids=batch["input_ids"],
                    inputs_embeds=batch["inputs_embeds"],
                    attention_mask=batch.get("attention_mask"),
                    labels=batch["labels"],
                )
            else:
                outputs = model(
                    input_ids=batch["input_ids"],
                    images=batch["images"],
                    image_masks=batch["image_masks"],
                    image_input_idx=batch["image_input_idx"],
                    labels=batch["labels"],
                )

            labels = batch["labels"][0].detach().cpu()
            shifted_labels = labels[1:]
            valid_positions = torch.nonzero(shifted_labels != -100, as_tuple=False).flatten()
            valid_label_ids = shifted_labels[valid_positions].tolist()
            preview_ids = valid_label_ids[:128]
            preview = tokenizer.decode(preview_ids, skip_special_tokens=False) if preview_ids else ""
            n_tokens = len(valid_label_ids)
            loss = float(outputs.loss.item())
            ppl = math.exp(min(loss, 100))
            seq_len = int(labels.numel())
            if valid_positions.numel() > 0:
                first_pos = int(valid_positions[0].item()) + 1
                last_pos = int(valid_positions[-1].item()) + 1
            else:
                first_pos = -1
                last_pos = -1

            print(f"[sample {index}] seq_len={seq_len} loss_tokens={n_tokens} loss={loss:.6f} ppl={ppl:.2f}")
            print(f"[sample {index}] label_pos_range=[{first_pos}, {last_pos}]")
            print(f"[sample {index}] decoded_loss_labels_preview={preview!r}")
    print("=" * 60 + "\n")


def build_worker_device_groups(num_requested_gpus: int, gpus_per_worker: int, num_visible_gpus: int) -> List[List[int]]:
    if num_requested_gpus < 1:
        raise ValueError("num_requested_gpus must be >= 1")
    if gpus_per_worker < 1:
        raise ValueError("gpus_per_worker must be >= 1")
    if num_visible_gpus < 1:
        raise ValueError("num_visible_gpus must be >= 1")

    usable = min(num_requested_gpus, num_visible_gpus)
    if usable < gpus_per_worker:
        raise ValueError(
            f"Requested {gpus_per_worker} GPUs per worker, but only {usable} usable GPU(s) are available"
        )

    groups = []
    for start in range(0, usable, gpus_per_worker):
        group = list(range(start, min(start + gpus_per_worker, usable)))
        if len(group) == gpus_per_worker:
            groups.append(group)
    if not groups:
        raise ValueError("No complete worker device groups could be formed")
    return groups


def _infer_worker_primary_device(device_group: List[int]) -> torch.device:
    return torch.device(f"cuda:{device_group[0]}")


def _build_molmo_balanced_max_memory(device_group: List[int]) -> Dict[int, str]:
    max_memory = {}
    for gpu_id in device_group:
        free_bytes, _ = torch.cuda.mem_get_info(gpu_id)
        reserve_bytes = min(4 * 1024**3, max(0, free_bytes // 8))
        usable_bytes = max(free_bytes - reserve_bytes, 8 * 1024**3)
        usable_gib = max(8, int(usable_bytes // 1024**3))
        max_memory[gpu_id] = f"{usable_gib}GiB"
    return max_memory


def load_molmo_model(model_path, device_group, cache_dir):
    from vlmeval.vlm.MolmoE.quant_modeling_molmoe import MolmoForCausalLM

    if isinstance(device_group, torch.device):
        device_group = [device_group.index if device_group.index is not None else 0]
    elif isinstance(device_group, int):
        device_group = [device_group]
    else:
        device_group = list(device_group)

    if not device_group:
        raise ValueError("device_group must be non-empty")

    if len(device_group) == 1:
        device_map = {"": f"cuda:{device_group[0]}"}
        max_memory = None
    else:
        device_map = "balanced"
        max_memory = _build_molmo_balanced_max_memory(device_group)

    model = MolmoForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        max_memory=max_memory,
        cache_dir=cache_dir,
        local_files_only=True,
    )
    model.eval()
    for layer_id, block in enumerate(model.model.transformer.blocks):
        mlp = getattr(block, "mlp", None)
        if hasattr(mlp, "get_layer_stats"):
            mlp.layer_id = layer_id
    return model


def worker(
    worker_id,
    device_group,
    args,
    model_path,
    inputs,
    permutations,
    activations,
    p_tilde,
    menu_values_shared,
    num_samples_shared,
    lock,
    num_experts_per_layer,
    start_layer,
    num_layers,
    output_dir,
    checkpoint_dir=None,
    permutation_start_index=0,
    completed_permutation_indices=None,
):
    os.makedirs(output_dir, exist_ok=True)
    group_label = "_".join(str(gpu_id) for gpu_id in device_group)
    log_file = open(os.path.join(output_dir, f"worker_gpu{group_label}.log"), "w", buffering=1)

    def log(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    device = _infer_worker_primary_device(device_group)
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    log(f"[Worker {worker_id}] starting worker on GPUs {device_group} with {permutations.shape[0]} permutations")

    if worker_id > 0:
        time.sleep(worker_id * 30)

    model = load_molmo_model(model_path, device_group, args.cache_dir)
    eval_batch_size = max(1, args.eval_batch_size)
    if inputs and isinstance(inputs[0], dict) and "inputs_embeds" not in inputs[0]:
        log(f"[Worker {worker_id}] precomputing MolMoE input embeddings once...")
        image_batch_size = max(1, args.image_batch_size)
        inputs = precompute_molmo_inputs_embeds(model, inputs, device, batch_size=image_batch_size)
        torch.cuda.empty_cache()
    if (args.cache_eval_batches_on_gpu or args.use_cached_full_forward) and inputs and isinstance(inputs[0], dict):
        log(
            f"[Worker {worker_id}] caching {len(inputs)} MolMoE calibration samples "
            f"as GPU eval batches (batch_size={eval_batch_size})..."
        )
        model_dtype = next(model.parameters()).dtype
        inputs = prepare_molmo_eval_batches(inputs, device, model_dtype, batch_size=eval_batch_size)
        torch.cuda.empty_cache()
    permutations = permutations.to(device)
    activations = activations.to(device)
    p_tilde = p_tilde.to(device)

    estimator = build_molmo_estimator(
        args,
        num_experts=(num_layers - start_layer) * num_experts_per_layer,
        device=device,
    )

    cached_full_forward = bool(args.use_cached_full_forward)

    def value_fn(model, inputs):
        nll = eval_nll_molmo(
            model,
            inputs,
            device,
            batch_size=eval_batch_size,
            fp32_loss_eval=args.fp32_loss_eval or cached_full_forward,
        )
        return math.exp(-min(nll, 100))

    completed_permutation_indices = set(completed_permutation_indices or [])
    completed_local_indices = {
        idx - permutation_start_index
        for idx in completed_permutation_indices
        if permutation_start_index <= idx < permutation_start_index + permutations.shape[0]
    }

    if checkpoint_dir is not None and completed_local_indices:
        log(
            f"[Worker {worker_id}] warning: checkpoint resume is not implemented for MolMoE slow mode; ignoring saved state"
        )

    if args.use_cached_full_forward:
        if args.verify_cached_full_forward:
            verify_layer = start_layer
            full_value = value_fn(model, inputs)
            cached_value = estimator.eval_cached_full_forward_value(model, inputs, verify_layer)
            diff = abs(full_value - cached_value)
            log(
                f"[Worker {worker_id}] cached full-forward verify: "
                f"full={full_value:.8f}, cached={cached_value:.8f}, diff={diff:.8f}, "
                f"layer={verify_layer}"
            )
            if diff > args.cached_full_forward_tolerance:
                raise RuntimeError(
                    f"Cached full-forward verification failed: diff {diff:.6f} "
                    f"> tolerance {args.cached_full_forward_tolerance:.6f}"
                )
        log(f"[Worker {worker_id}] estimating menu values (cached full-forward MolMoE path)...")
        menu_local = estimator.estimate_menu_value_cached_full_forward(
            model=model,
            inputs=inputs,
            permutations=permutations,
            p_tilde=p_tilde,
            num_experts_per_layer=num_experts_per_layer,
            start_layer=start_layer,
            value_fn=value_fn,
        )
    else:
        log(f"[Worker {worker_id}] estimating menu values (slow MolMoE path)...")
        menu_local = estimator.estimate_menu_value(
            model=model,
            inputs=inputs,
            permutations=permutations,
            p_tilde=p_tilde,
            num_experts_per_layer=num_experts_per_layer,
            start_layer=start_layer,
            value_fn=value_fn,
        )

    M_local = permutations.shape[0]
    with lock:
        for expert_key, menu_dict in menu_local.items():
            for menu_key, val in menu_dict.items():
                flat_key = f"{expert_key}|{menu_key}"
                if flat_key in menu_values_shared:
                    menu_values_shared[flat_key] += val * M_local
                else:
                    menu_values_shared[flat_key] = val * M_local
        num_samples_shared[worker_id] = M_local

    log(f"[Worker {worker_id}] done")
    log_file.close()


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    model_spec = get_model_spec(args.model)
    resolved_model_path = resolve_model_path(args.model, _PROJECT_ROOT)
    model_config = infer_model_config(resolved_model_path)
    args.model = resolved_model_path
    args.model_id = model_spec.model_id
    if not args.disable_checkpoints:
        if args.checkpoint_dir is None:
            args.checkpoint_dir = os.path.join(
                args.output_dir, "checkpoints", build_checkpoint_run_id(args, model_spec.model_id)
            )
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    num_layers = model_config.get("num_layers") or model_spec.num_layers
    num_experts = model_config.get("num_experts") or model_spec.num_experts
    start_layer = args.start_layer
    total_experts = (num_layers - start_layer) * num_experts

    print("\n" + "=" * 60)
    print("Phase I: MolMoE Menu Value Estimation")
    print("=" * 60)
    print(f"Model: {model_spec.model_id}")
    print(f"Model path: {resolved_model_path}")
    print(f"GPUs: {args.num_gpus}")
    print(f"GPUs per worker: {args.gpus_per_worker}")
    print(f"MC samples: {args.monte_carlo_samples}")
    print(f"Total experts: {total_experts}")
    print("=" * 60 + "\n")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, cache_dir=args.cache_dir, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.calib_data == "multimodal":
        print("Loading multimodal calibration data...")
        processor = AutoProcessor.from_pretrained(
            args.model,
            trust_remote_code=True,
            cache_dir=args.cache_dir,
            local_files_only=True,
        )
        inputs = get_multimodal_calib_molmo(
            args.nsamples,
            args.seed,
            processor,
            args.seqlen,
            data_path=args.calib_data_path,
            min_loss_tokens=args.min_loss_tokens,
        )
    else:
        print("Loading C4 calibration data...")
        inputs = get_c4(args.nsamples, args.seed, args.seqlen, tokenizer, data_path=args.calib_data_path)
    print(f"Loaded {len(inputs)} samples")

    worker_device_groups = build_worker_device_groups(
        args.num_gpus,
        args.gpus_per_worker,
        torch.cuda.device_count(),
    )

    def pick_stats_device(device_groups):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for MolMoE Phase I")
        best_group = device_groups[0]
        best_free = -1
        for group in device_groups:
            free_bytes, _ = torch.cuda.mem_get_info(group[0])
            if free_bytes > best_free:
                best_free = free_bytes
                best_group = group
        return best_group, best_free

    gc.collect()
    torch.cuda.empty_cache()
    stats_group, stats_free = pick_stats_device(worker_device_groups)
    print(
        f"Loading model for activation stats on GPUs {stats_group} "
        f"(primary free {stats_free / 1024**3:.2f} GiB)"
    )
    device = _infer_worker_primary_device(stats_group)
    torch.cuda.set_device(device)
    model = load_molmo_model(args.model, stats_group, args.cache_dir)

    print("Counting activation frequencies...")
    image_batch_size = max(1, args.image_batch_size)
    activations = count_activation_frequency_molmo(
        model, inputs, num_layers, num_experts, start_layer, device, batch_size=image_batch_size
    )
    if args.debug_loss_samples > 0:
        debug_molmo_loss_samples(model, inputs, tokenizer, device, max_samples=args.debug_loss_samples)
        fp_ppl = eval_ppl_molmo(model, inputs, device, batch_size=image_batch_size)
        print(f"FP calibration PPL before Phase I quant toggles: {fp_ppl:.6f}")
    estimator = build_molmo_estimator(args, num_experts=total_experts, device=device)
    p_tilde = estimator.compute_p_tilde(activations.to(device))

    del model
    gc.collect()
    torch.cuda.empty_cache()

    print("Sampling expert permutations...")
    permutations = estimator.sample_plackett_luce(p_tilde, num_samples=args.monte_carlo_samples)

    expected_perm_indices = set(range(args.monte_carlo_samples))
    completed_checkpoint_indices = set()
    if not args.disable_checkpoints:
        completed_checkpoint_indices, _, completed_samples = load_completed_checkpoints(
            args.checkpoint_dir, expected_perm_indices
        )
        print(
            f"Checkpoint resume: {len(completed_checkpoint_indices)}/{args.monte_carlo_samples} permutations "
            f"already complete ({completed_samples} samples)"
        )

    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    menu_values_shared = manager.dict()
    num_samples_shared = manager.dict()
    lock = manager.Lock()

    processes = []
    worker_count = len(worker_device_groups)
    rank_to_process = {}
    for worker_id, device_group in enumerate(worker_device_groups):
        start_idx = int(worker_id * permutations.shape[0] / worker_count)
        end_idx = int((worker_id + 1) * permutations.shape[0] / worker_count)
        rank_perms = permutations[start_idx:end_idx]
        if rank_perms.shape[0] == 0:
            continue
        p = mp.Process(
            target=worker,
            args=(
                worker_id,
                device_group,
                args,
                args.model,
                inputs,
                rank_perms,
                activations,
                p_tilde.cpu(),
                menu_values_shared,
                num_samples_shared,
                lock,
                num_experts,
                start_layer,
                num_layers,
                args.output_dir,
                None if args.disable_checkpoints else args.checkpoint_dir,
                start_idx,
                completed_checkpoint_indices,
            ),
        )
        p.start()
        processes.append(p)
        rank_to_process[worker_id] = p

    for p in processes:
        p.join()

    failed_workers = [worker_id for worker_id, p in rank_to_process.items() if p.exitcode != 0]
    if failed_workers:
        raise RuntimeError(f"Worker failed on worker ids: {failed_workers}")

    flat_values = dict(menu_values_shared)
    total_samples = sum(num_samples_shared.values())
    if total_samples == 0:
        raise RuntimeError("No worker produced samples.")

    menu_values_std = {}
    for flat_key, val in flat_values.items():
        expert_part, menu_key = flat_key.split("|", 1)
        parts = expert_part.split(".")
        layer_id = int(parts[0][1:])
        expert_id = int(parts[1][1:])
        std_key = f"model.transformer.blocks.{layer_id}.mlp.experts.{expert_id}"
        menu_values_std.setdefault(std_key, {})
        menu_values_std[std_key][menu_key] = val / total_samples

    os.makedirs(args.output_dir, exist_ok=True)
    menu_values_path = os.path.join(args.output_dir, get_menu_values_filename(model_spec.model_id))
    with open(menu_values_path, "w", encoding="utf-8") as f:
        json.dump(menu_values_std, f, indent=4)

    owen_compat = {}
    for expert_key, menus in menu_values_std.items():
        for proj in ["gate_proj", "up_proj", "down_proj"]:
            if proj in menus:
                owen_compat[f"{expert_key}.{proj}"] = menus[proj]
    owen_values_path = os.path.join(args.output_dir, get_owen_values_filename(model_spec.model_id))
    with open(owen_values_path, "w", encoding="utf-8") as f:
        json.dump(owen_compat, f, indent=4)

    print(f"Saved menu values: {menu_values_path}")
    print(f"Saved Owen-compatible projection values: {owen_values_path}")


if __name__ == "__main__":
    main()
