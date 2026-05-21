"""
Phase I：基于先验联盟的双层 Owen Value 估计

外层：Plackett-Luce 引导的专家级 MC 重要性采样
内层：固定宏观背景下，穷举 3! = 6 条投影层降级路径

输出：所有 3N 个投影层的 Owen Value
"""

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from transformers import AutoTokenizer
import numpy as np
import json
import os
import argparse
import math
import sys
import time
import gc
from typing import List, Dict

from menu_checkpoint import (
    checkpoint_path,
    load_completed_checkpoints,
    save_permutation_checkpoint,
)
from rgis_shapley import RGISShapleyEstimator

# 项目根目录（从当前脚本位置推导）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_VLM_DIR = os.path.join(_PROJECT_ROOT, '04_evaluation/VLMEvalKit/vlmeval/vlm')
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from deepseek_vl2_model import (
    get_attention_proj_names,
    get_menu_values_filename,
    get_model_spec,
    get_owen_values_filename,
    infer_model_config,
    resolve_model_path,
)


def build_checkpoint_run_id(args, model_id: str) -> str:
    early = str(args.early_stop_threshold).replace('.', 'p')
    frac = str(args.max_expert_fraction_per_permutation).replace('.', 'p')
    return (
        f"{model_id}_menu"
        f"_s{args.seqlen}_n{args.nsamples}_mc{args.monte_carlo_samples}"
        f"_seed{args.seed}_start{args.start_layer}"
        f"_estop{early}_frac{frac}_b{args.eval_batch_size}"
    )


def build_rank_permutation_assignments(permutations, num_gpus: int, completed_checkpoint_indices):
    completed_checkpoint_indices = {int(i) for i in completed_checkpoint_indices}
    all_indices = list(range(permutations.shape[0]))
    missing_indices = [idx for idx in all_indices if idx not in completed_checkpoint_indices]
    indices_to_assign = missing_indices if missing_indices else all_indices

    assignments = []
    total = len(indices_to_assign)
    for rank in range(num_gpus):
        start = int(rank * total / num_gpus)
        end = int((rank + 1) * total / num_gpus)
        rank_global_indices = indices_to_assign[start:end]
        if not rank_global_indices:
            assignments.append(
                {
                    "rank": rank,
                    "global_indices": [],
                    "permutations": permutations[0:0],
                    "resumed_count": 0,
                    "missing_count": 0,
                }
            )
            continue

        rank_tensor = torch.tensor(rank_global_indices, dtype=torch.long)
        rank_perms = permutations.index_select(0, rank_tensor)
        assignments.append(
            {
                "rank": rank,
                "global_indices": rank_global_indices,
                "permutations": rank_perms,
                "resumed_count": sum(1 for idx in rank_global_indices if idx in completed_checkpoint_indices),
                "missing_count": len(rank_global_indices),
            }
        )
    return assignments

# ==================== C4数据加载 ====================

def get_c4(nsamples, seed, seqlen, tokenizer,
           data_path=None):
    if data_path is None:
        data_path = os.path.join(_PROJECT_ROOT, 'data/c4_train.json')
    import random
    traindata = []
    with open(data_path, 'r') as f:
        for i, line in enumerate(f):
            try:
                traindata.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    print(f"加载了 {len(traindata)} 条C4数据")

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            text = traindata[i]['text'] if isinstance(traindata[i], dict) else traindata[i]
            trainenc = tokenizer(text, return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader


def get_multimodal_calib(nsamples, seed, model, device,
                         data_path=None, model_path=None, seqlen=128):
    """
    加载多模态校准数据，预计算 inputs_embeds。

    返回格式与 get_c4 兼容: List[(inputs_embeds, labels)]
      - inputs_embeds: [1, seqlen, hidden_dim], bf16
      - labels: [1, seqlen], long (-100 for non-target tokens)

    注意: 必须在模型加载后、释放前调用。
    embeds 会截断到 seqlen 以与文本校准对齐，同时控制显存和计算量。
    """
    if data_path is None:
        data_path = os.path.join(_PROJECT_ROOT, 'data/multimodal_calib.jsonl')
    if model_path is None:
        model_path = resolve_model_path('deepseek_vl2_tiny', _PROJECT_ROOT)

    import random
    from PIL import Image

    sys.path.insert(0, _VLM_DIR)
    from DeepSeek_VL2.processing_deepseek_vl_v2 import DeepseekVLV2Processor

    # 加载 processor
    processor = DeepseekVLV2Processor.from_pretrained(model_path)

    # 加载数据
    samples = []
    with open(data_path, 'r') as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    print(f"加载了 {len(samples)} 条多模态校准数据")

    random.seed(seed)
    random.shuffle(samples)
    samples = samples[:nsamples]

    # 预计算 inputs_embeds
    trainloader = []
    model.eval()
    skipped = 0
    for idx, sample in enumerate(samples):
        try:
            img_path = sample["image_path"]
            image = Image.open(img_path).convert("RGB")

            # 构建对话格式
            conversations = sample["conversations"]
            user_msg = conversations[0]["content"]
            assistant_msg = conversations[1]["content"] if len(conversations) > 1 else ""

            # 使用 processor 处理
            conversation = [
                {"role": "<|User|>", "content": user_msg},
                {"role": "<|Assistant|>", "content": assistant_msg},
            ]
            prepare = processor(
                conversations=conversation,
                images=[image],
                force_batchify=True,
                system_prompt="",
            )
            prepare = prepare.to(device)

            with torch.no_grad():
                inputs_embeds = model.prepare_inputs_embeds(**prepare)

            # 使用 processor 提供的精确 labels（只对 assistant 回复计算 loss）
            labels = prepare.labels  # [1, seq_len], 已正确 mask

            # 截断到 seqlen，保留尾部（包含 assistant 回复的 loss token）
            orig_len = inputs_embeds.shape[1]
            if orig_len > seqlen:
                inputs_embeds = inputs_embeds[:, -seqlen:, :]
                labels = labels[:, -seqlen:]

            trainloader.append((
                inputs_embeds.cpu(),  # [1, seqlen, hidden_dim]
                labels.cpu(),  # [1, seqlen]
            ))

            if (idx + 1) % 20 == 0:
                print(f"  预计算 embeds: {idx+1}/{len(samples)}, "
                      f"orig_len={orig_len}, truncated_to={inputs_embeds.shape[1]}")

        except Exception as e:
            skipped += 1
            if skipped <= 3:
                print(f"  跳过样本 {idx}: {e}")
            continue

    if skipped > 0:
        print(f"  共跳过 {skipped} 个样本")
    print(f"多模态校准数据: {len(trainloader)} 个有效样本")
    if len(trainloader) == 0:
        raise RuntimeError("多模态校准数据为空，请检查图像路径和 processor 配置")
    return trainloader


# ==================== PPL 评估 ====================

def eval_ppl_prebuilt(model, prebuilt_batches, device) -> float:
    """预构建 batch 版本的 PPL 评估，避免重复 cat 开销。"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch_embeds, batch_labels in prebuilt_batches:
            outputs = model.language(inputs_embeds=batch_embeds, labels=batch_labels)
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
            n_tokens = (batch_labels != -100).sum().item()
            if n_tokens > 0:
                total_loss += loss.item() * n_tokens
                total_tokens += n_tokens
    avg_loss = total_loss / max(total_tokens, 1)
    return math.exp(min(avg_loss, 100))


def eval_ppl(model, inputs, device, batch_size: int = 4) -> float:
    """
    计算 PPL。支持两种输入格式:
      - 文本模式: inputs[i] = (input_ids [1, seq_len], targets [1, seq_len])
      - 多模态模式: inputs[i] = (inputs_embeds [1, seq_len, D], labels [1, seq_len])
    通过第一个样本的维度自动判断。
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    num_samples = len(inputs)

    # 检测模式: 3D = embeds, 2D = input_ids
    is_embeds = (inputs[0][0].dim() == 3)

    with torch.no_grad():
        if is_embeds:
            # 多模态模式: 按 batch_size 分组，与 routing cache 对齐
            for i in range(0, num_samples, batch_size):
                batch_embeds = []
                batch_labels = []
                for j in range(i, min(i + batch_size, num_samples)):
                    batch_embeds.append(inputs[j][0])
                    batch_labels.append(inputs[j][1])
                batch_embeds = torch.cat(batch_embeds, dim=0).to(device)
                batch_labels = torch.cat(batch_labels, dim=0).to(device)
                outputs = model.language(inputs_embeds=batch_embeds, labels=batch_labels)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                n_tokens = (batch_labels != -100).sum().item()
                if n_tokens > 0:
                    total_loss += loss.item() * n_tokens
                    total_tokens += n_tokens
        else:
            # 文本模式: 批量处理（原逻辑）
            for i in range(0, num_samples, batch_size):
                batch_inputs = []
                for j in range(i, min(i + batch_size, num_samples)):
                    batch_inputs.append(inputs[j][0])
                batch_inputs = torch.cat(batch_inputs, dim=0).to(device)
                outputs = model(input_ids=batch_inputs, labels=batch_inputs)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                batch_tokens = batch_inputs.numel()
                total_loss += loss.item() * batch_tokens
                total_tokens += batch_tokens
    avg_loss = total_loss / max(total_tokens, 1)
    return math.exp(min(avg_loss, 100))

# ==================== 激活频率统计 ====================

def count_activation_frequency(model, inputs, num_layers, num_experts_per_layer,
                                start_layer, device, batch_size=4):
    import torch.nn.functional as F
    total_experts = (num_layers - start_layer) * num_experts_per_layer
    activation_counts = torch.zeros(total_experts, device='cpu')
    total_tokens = 0
    num_samples = len(inputs)

    # 启用激活统计
    for name, module in model.named_modules():
        if 'DeepseekV2MoE' in type(module).__name__:
            if hasattr(module, 'layer_id') and module.layer_id >= start_layer:
                module.count_activations = True

    def _collect_activation_counts():
        for name, module in model.named_modules():
            if 'DeepseekV2MoE' in type(module).__name__:
                if not hasattr(module, 'layer_id'):
                    continue
                layer_id = module.layer_id
                if layer_id < start_layer:
                    continue
                if hasattr(module, 'expert_activation_counts'):
                    counts = module.expert_activation_counts
                    if counts is not None:
                        layer_offset = (layer_id - start_layer) * num_experts_per_layer
                        for eid in range(min(len(counts), num_experts_per_layer)):
                            activation_counts[layer_offset + eid] += counts[eid].cpu().item()
                    module.expert_activation_counts = None

    # 检测模式
    is_embeds = (inputs[0][0].dim() == 3)

    for batch_start in range(0, num_samples, batch_size):
        batch_end = min(batch_start + batch_size, num_samples)

        if is_embeds:
            batch_embeds = torch.cat(
                [inputs[j][0] for j in range(batch_start, batch_end)], dim=0
            ).to(device)
            total_tokens += batch_embeds.shape[0] * batch_embeds.shape[1]
            with torch.no_grad():
                model.language(inputs_embeds=batch_embeds)
            _collect_activation_counts()
        else:
            batch_inputs = []
            for j in range(batch_start, batch_end):
                batch_inputs.append(inputs[j][0])
            batch_inputs = torch.cat(batch_inputs, dim=0).to(device)
            batch_tokens = batch_inputs.numel()
            total_tokens += batch_tokens
            with torch.no_grad():
                model(input_ids=batch_inputs)
            _collect_activation_counts()

    # 关闭激活统计
    for name, module in model.named_modules():
        if 'DeepseekV2MoE' in type(module).__name__:
            if hasattr(module, 'layer_id') and module.layer_id >= start_layer:
                module.count_activations = False

    if total_tokens > 0:
        activation_counts /= total_tokens
    activation_counts = activation_counts.clamp(min=1e-8)
    activation_counts /= activation_counts.sum()
    return activation_counts

# ==================== 路由冻结 ====================

def cache_routing_decisions(model, inputs, num_layers, start_layer, device, batch_size=4):
    routing_cache = {layer_id: [] for layer_id in range(start_layer, num_layers)}
    num_samples = len(inputs)

    for name, module in model.named_modules():
        if 'DeepseekV2MoE' in type(module).__name__:
            if hasattr(module, 'layer_id') and module.layer_id >= start_layer:
                module.record_routing = True

    def _collect_routing():
        for name, module in model.named_modules():
            if 'DeepseekV2MoE' in type(module).__name__:
                if hasattr(module, 'layer_id') and module.layer_id >= start_layer:
                    if hasattr(module, 'cached_routing') and module.cached_routing is not None:
                        topk_idx, topk_weight = module.cached_routing
                        routing_cache[module.layer_id].append((topk_idx.cpu(), topk_weight.cpu()))
                        module.cached_routing = None

    # 检测模式
    is_embeds = (inputs[0][0].dim() == 3)

    if is_embeds:
        # 多模态: 按 batch_size 分组，cat 后一次 forward，routing 按组存储
        for batch_start in range(0, num_samples, batch_size):
            batch_end = min(batch_start + batch_size, num_samples)
            batch_embeds = torch.cat(
                [inputs[j][0] for j in range(batch_start, batch_end)], dim=0
            ).to(device)  # [B, seqlen, D]
            with torch.no_grad():
                model.language(inputs_embeds=batch_embeds)
            _collect_routing()
    else:
        for batch_start in range(0, num_samples, batch_size):
            batch_end = min(batch_start + batch_size, num_samples)
            batch_inputs = []
            for j in range(batch_start, batch_end):
                batch_inputs.append(inputs[j][0])
            batch_inputs = torch.cat(batch_inputs, dim=0).to(device)
            with torch.no_grad():
                model(input_ids=batch_inputs)
            _collect_routing()

    for name, module in model.named_modules():
        if 'DeepseekV2MoE' in type(module).__name__:
            if hasattr(module, 'record_routing'):
                module.record_routing = False

    total_batches = len(routing_cache[start_layer])
    print(f"路由缓存完成: {num_layers - start_layer} 层 x {total_batches} 个batch")
    return routing_cache


def apply_routing_frozen(model, routing_cache, start_layer, num_layers, device):
    for layer_id in range(start_layer, num_layers):
        for name, module in model.named_modules():
            if 'DeepseekV2MoE' in type(module).__name__:
                if hasattr(module, 'layer_id') and module.layer_id == layer_id:
                    if layer_id in routing_cache and routing_cache[layer_id]:
                        frozen_decisions = [
                            (topk_idx.to(device), topk_weight.to(device))
                            for topk_idx, topk_weight in routing_cache[layer_id]
                        ]
                        module.frozen_routing = frozen_decisions
                        module.frozen_routing_idx = 0
                        module.use_frozen_routing = True
    print(f"routing-frozen 已应用到层 {start_layer}-{num_layers-1}")


def reset_routing_counters(model, start_layer, num_layers):
    for layer_id in range(start_layer, num_layers):
        for name, module in model.named_modules():
            if 'DeepseekV2MoE' in type(module).__name__:
                if hasattr(module, 'layer_id') and module.layer_id == layer_id:
                    if hasattr(module, 'frozen_routing_idx'):
                        module.frozen_routing_idx = 0

# PLACEHOLDER_WORKER

# ==================== Worker 进程 ====================

def worker(rank, args, model_path, inputs, permutations, activations, p_tilde,
           owen_values_shared, num_samples_shared, lock,
           num_experts_per_layer, start_layer, num_layers, output_dir,
           checkpoint_dir=None, permutation_indices=None,
           completed_permutation_indices=None):
    """Worker进程：在单个GPU上估计分配到的排列的 Owen Value"""
    os.makedirs(output_dir, exist_ok=True)
    log_file = open(os.path.join(output_dir, f"worker_gpu{rank}.log"), "w", buffering=1)

    def log(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    log(f"[GPU {rank}] 启动worker，处理 {permutations.shape[0]} 个排列")

    sys.path.insert(0, _VLM_DIR)
    try:
        from DeepSeek_VL2.modeling_deepseek import DeepseekV2MoE
        from DeepSeek_VL2.modeling_deepseek_vl_v2 import DeepseekVLV2ForCausalLM
    except ImportError:
        from DeepSeek_VL2_quant.modeling_deepseek import DeepseekV2MoE
        from DeepSeek_VL2_quant.modeling_deepseek_vl_v2 import DeepseekVLV2ForCausalLM

    # 错开模型加载，避免多进程同时访问 HuggingFace 文件缓存导致文件锁死锁
    if rank > 0:
        import time as _time
        _time.sleep(rank * 60)  # 每个 worker 错开 60 秒
    log(f"[GPU {rank}] 加载模型...")
    model = DeepseekVLV2ForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
        device_map={"": device}, cache_dir=args.cache_dir, local_files_only=True
    )
    model.eval()

    # 设置 layer_id
    for name, module in model.named_modules():
        if 'DeepseekV2MoE' in type(module).__name__:
            parts = name.split('.')
            for i, part in enumerate(parts):
                if part == 'layers' and i + 1 < len(parts):
                    module.layer_id = int(parts[i + 1])
                    break

    permutations = permutations.to(device)
    activations = activations.to(device)
    p_tilde = p_tilde.to(device)

    # 预量化 attention / shared expert / dense FFN 层（模拟实际部署背景）
    if args.prequant_background:
        from rgis_shapley import fake_quantize_weight
        attention_proj_names = tuple(get_attention_proj_names(model_ref=args.model))
        prequant_count = 0
        with torch.no_grad():
            for name, module in model.named_modules():
                if not isinstance(module, nn.Linear):
                    continue
                should_quant = False
                # attention: select the projection layout that matches the target model
                if 'self_attn' in name and any(p in name for p in attention_proj_names):
                    should_quant = True
                # shared experts: mlp.shared_experts.{gate,up,down}_proj
                elif 'shared_experts' in name and any(p in name for p in ['gate_proj', 'up_proj', 'down_proj']):
                    should_quant = True
                # dense FFN (layer 0): layers.0.mlp.{gate,up,down}_proj (不含 experts)
                elif 'experts' not in name and 'layers.0.mlp' in name and any(p in name for p in ['gate_proj', 'up_proj', 'down_proj']):
                    should_quant = True

                if should_quant:
                    module.weight.data = fake_quantize_weight(
                        module.weight.data, bits=args.fixed_bits, group_size=args.group_size
                    )
                    prequant_count += 1
        log(f"[GPU {rank}] 预量化了 {prequant_count} 个背景层到 {args.fixed_bits}-bit")

    # 创建 RGIS 估计器
    total_experts = (num_layers - start_layer) * num_experts_per_layer
    estimator = RGISShapleyEstimator(
        num_experts=total_experts,
        tau=args.tau_temp,
        lambda_mix=args.lambda_mix,
        use_self_normalized=args.use_self_normalized,
        early_stop_threshold=args.early_stop_threshold,
        bits_high=4, bits_low=2,
        max_expert_fraction_per_permutation=args.max_expert_fraction_per_permutation,
        device=device
    )

    # 预构建 batch 到 GPU，避免每次 eval_ppl 重复 cat + CPU→GPU 传输
    is_embeds = (inputs[0][0].dim() == 3)
    eval_batch_size = max(1, args.eval_batch_size)
    if is_embeds:
        prebuilt_batches = []
        for i in range(0, len(inputs), eval_batch_size):
            batch_embeds = torch.cat(
                [inputs[j][0] for j in range(i, min(i + eval_batch_size, len(inputs)))], dim=0
            ).contiguous()
            batch_labels = torch.cat(
                [inputs[j][1] for j in range(i, min(i + eval_batch_size, len(inputs)))], dim=0
            ).contiguous()
            prebuilt_batches.append((batch_embeds, batch_labels))
        # 同时保留原始 per-sample inputs 在 GPU 上（_capture_single_expert 需要）
        inputs = [(t[0].contiguous(), t[1].contiguous()) for t in inputs]
    else:
        prebuilt_batches = None
        inputs = [(t[0].contiguous(), t[1].contiguous()) for t in inputs]

    def value_fn(model, inputs):
        reset_routing_counters(model, start_layer, num_layers)
        if False and prebuilt_batches is not None:
            return 1.0 / eval_ppl_prebuilt(model, prebuilt_batches, device)
        ppl = eval_ppl(model, inputs, device, batch_size=eval_batch_size)
        return 1.0 / ppl

    permutation_indices = [int(idx) for idx in (permutation_indices or [])]
    completed_permutation_indices = set(completed_permutation_indices or [])
    local_to_global_perm_idx = {
        local_idx: global_idx for local_idx, global_idx in enumerate(permutation_indices)
    }
    completed_local_indices = {
        local_idx
        for local_idx, global_idx in local_to_global_perm_idx.items()
        if global_idx in completed_permutation_indices
    }

    def checkpoint_callback(local_perm_idx, flat_values):
        if checkpoint_dir is None:
            return
        global_perm_idx = local_to_global_perm_idx[int(local_perm_idx)]
        payload = {
            "version": 1,
            "model_id": getattr(args, "model_id", None),
            "permutation_index": global_perm_idx,
            "rank": rank,
            "num_samples": 1,
            "flat_values": flat_values,
        }
        save_permutation_checkpoint(
            checkpoint_path(checkpoint_dir, global_perm_idx), payload
        )
        log(f"[GPU {rank}] checkpoint saved: permutation {global_perm_idx}")

    # 估计 Menu Value（使用前缀共享加速版）
    M_local = permutations.shape[0] - len(completed_local_indices)
    if getattr(args, 'no_fast', False):
        if checkpoint_dir is not None:
            log(f"[GPU {rank}] warning: checkpoint resume is only supported in fast mode")
            checkpoint_dir = None
            completed_local_indices = set()
            M_local = permutations.shape[0]
        log(f"[GPU {rank}] 开始估计 Menu Value (original mode)...")
        menu_local = estimator.estimate_menu_value(
            model=model, inputs=inputs, permutations=permutations,
            p_tilde=p_tilde, num_experts_per_layer=num_experts_per_layer,
            start_layer=start_layer, value_fn=value_fn,
        )
    else:
        log(f"[GPU {rank}] 开始估计 Menu Value (fast mode)...")
        menu_local = estimator.estimate_menu_value_fast(
            model=model, inputs=inputs, permutations=permutations,
            p_tilde=p_tilde, num_experts_per_layer=num_experts_per_layer,
            start_layer=start_layer, value_fn=value_fn,
            batch_size=eval_batch_size,
            checkpoint_callback=checkpoint_callback if checkpoint_dir is not None else None,
            skip_permutation_indices=completed_local_indices,
        )

    # 原子累积到共享字典（扁平 key: "expert_key|menu_key"）
    with lock:
        for expert_key, menu_dict in menu_local.items():
            for menu_key, val in menu_dict.items():
                flat_key = f"{expert_key}|{menu_key}"
                if flat_key in owen_values_shared:
                    owen_values_shared[flat_key] += val * M_local
                else:
                    owen_values_shared[flat_key] = val * M_local
        num_samples_shared[rank] = M_local

    n_experts = len(menu_local)
    n_menus = sum(len(v) for v in menu_local.values())
    log(f"[GPU {rank}] 完成! 估计了 {n_experts} 个专家 × {n_menus//max(n_experts,1)} 个menu，处理了 {M_local} 个排列")
    log_file.close()

# PLACEHOLDER_MAIN

# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(description='Phase I: 双层 Owen Value 估计')
    parser.add_argument('--model', type=str, default='deepseek_vl2_tiny')
    parser.add_argument('--nsamples', type=int, default=128, help='C4数据样本数')
    parser.add_argument('--seqlen', type=int, default=128, help='序列长度')
    parser.add_argument('--monte_carlo_samples', type=int, default=20, help='外层MC采样数')
    parser.add_argument('--tau_temp', type=float, default=0.7, help='温度化参数')
    parser.add_argument('--lambda_mix', type=float, default=0.1, help='均匀混合比例')
    parser.add_argument('--use_self_normalized', action='store_true',
                        help='使用自归一化IS（仅用于旧的 estimate_shapley 方法，Owen Value 不使用）')
    parser.add_argument('--early_stop_threshold', type=float, default=0.5, help='早停阈值')
    parser.add_argument('--max_expert_fraction_per_permutation', type=float, default=1.0,
                        help='每条 RGIS 排列最多评估的前缀比例；1.0 表示完整遍历')
    parser.add_argument('--eval_batch_size', type=int, default=4,
                        help='PPL/menu 估值 batch size；显存允许时可调大')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Directory for per-permutation checkpoints')
    parser.add_argument('--disable_checkpoints', action='store_true',
                        help='Disable per-permutation checkpointing and resume')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_gpus', type=int, default=2)
    parser.add_argument('--cache_dir', type=str,
                        default=os.path.join(os.path.expanduser('~'), '.cache/huggingface'))
    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(_PROJECT_ROOT, 'shapley_results'))
    parser.add_argument('--start_layer', type=int, default=1)
    parser.add_argument('--prequant_background', action='store_true',
                        help='在计算 Owen Value 前将 attention/shared expert/dense FFN 预量化到 fixed_bits')
    parser.add_argument('--fixed_bits', type=int, default=4,
                        help='预量化的目标位宽')
    parser.add_argument('--group_size', type=int, default=128,
                        help='预量化的 group size')
    parser.add_argument('--calib_data', type=str, default='c4',
                        choices=['c4', 'multimodal'],
                        help='校准数据类型: c4(纯文本) 或 multimodal(多模态混合)')
    parser.add_argument('--calib_data_path', type=str, default=None,
                        help='校准数据文件路径 (默认自动选择)')
    parser.add_argument('--no_fast', action='store_true',
                        help='禁用前缀共享加速，使用原始全前向方法（用于验证）')
    parser.add_argument('--use_activation_cache', action='store_true',
                        help='兼容旧脚本参数；当前版本始终使用内置的激活统计流程')
    args = parser.parse_args()

    model_spec = get_model_spec(args.model)
    resolved_model_path = resolve_model_path(args.model, _PROJECT_ROOT)
    model_config = infer_model_config(resolved_model_path)
    args.model = resolved_model_path
    args.model_id = model_spec.model_id
    if not args.disable_checkpoints:
        if args.checkpoint_dir is None:
            args.checkpoint_dir = os.path.join(
                args.output_dir, 'checkpoints',
                build_checkpoint_run_id(args, model_spec.model_id),
            )
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    num_layers = model_config.get('num_layers') or model_spec.num_layers
    num_experts = model_config.get('num_experts') or model_spec.num_experts
    start_layer = args.start_layer
    total_experts = (num_layers - start_layer) * num_experts

    print(f"\n{'='*60}")
    print(f"Phase I: Menu Value 估计")
    print(f"{'='*60}")
    print(f"模型: {model_spec.model_id}")
    print(f"模型路径: {resolved_model_path}")
    print(f"GPU数量: {args.num_gpus}")
    print(f"外层MC采样: {args.monte_carlo_samples}")
    print(f"内层穷举: 8 个 menu (2^3 子集)")
    print(f"温度化参数: {args.tau_temp}, 混合比例: {args.lambda_mix}")
    print(f"早停阈值: {args.early_stop_threshold}")
    print(f"排列前缀截断比例: {args.max_expert_fraction_per_permutation}")
    print(f"估值 batch size: {args.eval_batch_size}")
    print(f"Checkpoint: {'disabled' if args.disable_checkpoints else args.checkpoint_dir}")
    print(f"总专家数: {total_experts}, 总投影层数: {total_experts * 3}")
    print(f"校准数据: {args.calib_data}")
    print(f"{'='*60}\n")

    # 加载分词器
    print("加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, cache_dir=args.cache_dir, local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载校准数据（文本模式先加载，多模态模式在模型加载后处理）
    if args.calib_data == 'c4':
        print("加载C4数据...")
        inputs = get_c4(args.nsamples, args.seed, args.seqlen, tokenizer,
                        data_path=args.calib_data_path)
        print(f"加载了 {len(inputs)} 个样本")
    else:
        inputs = None  # 多模态数据需要模型来预计算 embeds

    def pick_stats_device(num_requested_gpus):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用，无法运行 DeepSeek Phase I")
        num_visible = torch.cuda.device_count()
        search_n = min(max(num_requested_gpus, 1), num_visible)
        best_rank = 0
        best_free = -1
        for rank in range(search_n):
            free_bytes, _ = torch.cuda.mem_get_info(rank)
            if free_bytes > best_free:
                best_free = free_bytes
                best_rank = rank
        return best_rank, best_free

    sys.path.insert(0, _VLM_DIR)
    try:
        from DeepSeek_VL2.modeling_deepseek import DeepseekV2MoE
        from DeepSeek_VL2.modeling_deepseek_vl_v2 import DeepseekVLV2ForCausalLM
    except ImportError:
        from DeepSeek_VL2_quant.modeling_deepseek import DeepseekV2MoE
        from DeepSeek_VL2_quant.modeling_deepseek_vl_v2 import DeepseekVLV2ForCausalLM

    gc.collect()
    torch.cuda.empty_cache()
    stats_rank, stats_free = pick_stats_device(args.num_gpus)
    print(f"\n在 GPU {stats_rank} 上加载模型统计激活频率... "
          f"(当前空闲 {stats_free / 1024**3:.2f} GiB)")
    device = torch.device(f"cuda:{stats_rank}")
    torch.cuda.set_device(device)
    model = DeepseekVLV2ForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.bfloat16,
        device_map={"": device}, cache_dir=args.cache_dir, local_files_only=True
    )
    model.eval()

    # 设置 layer_id
    for name, module in model.named_modules():
        if 'DeepseekV2MoE' in type(module).__name__:
            parts = name.split('.')
            for i, part in enumerate(parts):
                if part == 'layers' and i + 1 < len(parts):
                    module.layer_id = int(parts[i + 1])
                    break

    # 多模态数据: 在模型加载后预计算 inputs_embeds
    if args.calib_data == 'multimodal' and inputs is None:
        print("\n预计算多模态校准数据的 inputs_embeds...")
        inputs = get_multimodal_calib(
            args.nsamples, args.seed, model, device,
            data_path=args.calib_data_path,
            model_path=args.model,
            seqlen=args.seqlen,
        )

    # 统计激活频率
    print("统计全局专家激活频率...")
    activations = count_activation_frequency(
        model, inputs, num_layers, num_experts, start_layer, device,
        batch_size=args.eval_batch_size
    )
    print(f"激活概率范围: [{activations.min():.6f}, {activations.max():.6f}]")

    # 计算 p_tilde
    print("\n计算稳健化采样权重 p_tilde...")
    estimator = RGISShapleyEstimator(
        num_experts=total_experts, tau=args.tau_temp,
        lambda_mix=args.lambda_mix, use_self_normalized=args.use_self_normalized,
        device=device
    )
    p_tilde = estimator.compute_p_tilde(activations)
    print(f"p_tilde范围: [{p_tilde.min():.6f}, {p_tilde.max():.6f}]")

    # 释放模型
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"已释放 GPU {stats_rank} 内存")

    # 生成排列
    print(f"\n生成 {args.monte_carlo_samples} 个专家级排列...")
    permutations = estimator.sample_plackett_luce(
        p_tilde, num_samples=args.monte_carlo_samples
    )
    print(f"排列形状: {permutations.shape}")

    # 多GPU并行
    expected_perm_indices = set(range(args.monte_carlo_samples))
    completed_checkpoint_indices = set()
    if not args.disable_checkpoints:
        completed_checkpoint_indices, _, completed_samples = load_completed_checkpoints(
            args.checkpoint_dir, expected_perm_indices
        )
        print(
            f"Checkpoint resume: {len(completed_checkpoint_indices)}/"
            f"{args.monte_carlo_samples} permutations already complete "
            f"({completed_samples} samples) at {args.checkpoint_dir}"
        )

    mp.set_start_method('spawn', force=True)
    manager = mp.Manager()
    owen_values_shared = manager.dict()
    num_samples_shared = manager.dict()
    lock = manager.Lock()

    print(f"\n将 {args.monte_carlo_samples} 个排列分配到 {args.num_gpus} 个GPU...")
    compute_start_time = time.time()
    processes = []
    rank_to_process = {}  # 保存 rank -> process 映射
    assignments = build_rank_permutation_assignments(
        permutations=permutations,
        num_gpus=args.num_gpus,
        completed_checkpoint_indices=completed_checkpoint_indices,
    )
    resumed_mode = bool(completed_checkpoint_indices)
    for assignment in assignments:
        rank = assignment["rank"]
        rank_perms = assignment["permutations"]
        rank_global_indices = assignment["global_indices"]

        if rank_perms.shape[0] == 0:
            print(f"  GPU {rank}: 跳过（无排列分配）")
            continue

        if resumed_mode:
            print(f"  GPU {rank}: missing permutations {rank_global_indices}")
        else:
            print(
                f"  GPU {rank}: permutations {rank_global_indices[0]}-{rank_global_indices[-1]} "
                f"({rank_perms.shape[0]} total)"
            )

        p = mp.Process(
            target=worker,
            args=(rank, args, args.model, inputs, rank_perms, activations,
                  p_tilde, owen_values_shared, num_samples_shared, lock,
                  num_experts, start_layer, num_layers, args.output_dir,
                  None if args.disable_checkpoints else args.checkpoint_dir,
                  rank_global_indices, set())
        )
        p.start()
        processes.append(p)
        rank_to_process[rank] = p

    for p in processes:
        p.join()

    compute_elapsed = time.time() - compute_start_time
    print(f"\n所有 worker 完成, 计算总耗时 {compute_elapsed:.1f}s ({compute_elapsed/60:.1f}min)")

    # 检查 worker 是否成功
    failed_ranks = [rank for rank, p in rank_to_process.items() if p.exitcode != 0]
    if failed_ranks:
        print(f"\n错误: {len(failed_ranks)} 个 worker 失败 (GPU ranks: {failed_ranks})。")
        print(f"请检查 worker 日志文件: {args.output_dir}/worker_gpu*.log")
        raise RuntimeError(
            f"Worker 失败 (GPU ranks: {failed_ranks})，拒绝使用不完整的采样结果汇总。"
            f" 请排查失败原因后重新运行。"
        )

    # 汇总结果：加权平均
    print("\n汇总 Menu Value 结果...")
    if args.disable_checkpoints:
        flat_values = dict(owen_values_shared)
        num_samples_dict = dict(num_samples_shared)
        total_samples = sum(num_samples_dict.values())
        print(f"  各 GPU 处理的排列数: {num_samples_dict}")
    else:
        completed_indices, flat_values, total_samples = load_completed_checkpoints(
            args.checkpoint_dir, expected_perm_indices
        )
        missing = sorted(expected_perm_indices - completed_indices)
        print(f"  checkpoint目录: {args.checkpoint_dir}")
        print(f"  checkpoint排列数: {len(completed_indices)}/{args.monte_carlo_samples}")
        if missing:
            preview = missing[:20]
            raise RuntimeError(
                f"Checkpoint incomplete: missing {len(missing)} permutations, "
                f"first missing indices={preview}"
            )

    print(f"  总排列数: {total_samples}")

    if total_samples == 0:
        raise RuntimeError("所有 worker 都未成功处理任何排列，请检查日志")

    # 从扁平 key 重建嵌套结构，同时转换为标准 key 格式
    # 扁平 key: "L{layer}.E{expert}|{menu_key}"
    # 标准 key: "language.model.layers.{L}.mlp.experts.{E}"
    menu_values_std = {}
    for flat_key, val in flat_values.items():
        expert_part, menu_key = flat_key.split('|', 1)
        parts = expert_part.split('.')
        layer_id = int(parts[0][1:])
        expert_id = int(parts[1][1:])
        std_key = f"language.model.layers.{layer_id}.mlp.experts.{expert_id}"
        if std_key not in menu_values_std:
            menu_values_std[std_key] = {}
        menu_values_std[std_key][menu_key] = val / total_samples

    # 保存
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, get_menu_values_filename(model_spec.model_id))
    with open(output_file, 'w') as f:
        json.dump(menu_values_std, f, indent=4)
    print(f"Menu Value 保存到: {output_file}")

    # 同时保存兼容旧格式的 per-projection Owen Value（取单投影层 menu 的值）
    owen_compat = {}
    for expert_key, menus in menu_values_std.items():
        for proj in ['gate_proj', 'up_proj', 'down_proj']:
            if proj in menus:
                owen_compat[f"{expert_key}.{proj}"] = menus[proj]
    compat_file = os.path.join(args.output_dir, get_owen_values_filename(model_spec.model_id))
    with open(compat_file, 'w') as f:
        json.dump(owen_compat, f, indent=4)
    print(f"兼容旧格式 Owen Value 保存到: {compat_file}")

    # 统计摘要
    all_h_vals = []
    for menus in menu_values_std.values():
        all_h_vals.extend(menus.values())
    print(f"\nMenu Value 统计:")
    print(f"  总专家数: {len(menu_values_std)}")
    print(f"  总 (expert, menu) 条目: {len(all_h_vals)}")
    print(f"  H 值范围: [{min(all_h_vals):.6e}, {max(all_h_vals):.6e}]")
    print(f"  H 值均值: {np.mean(all_h_vals):.6e}")

    # 按 menu 类型统计
    from rgis_shapley import enumerate_menus, menu_to_key
    nonempty_menus = [m for m in enumerate_menus() if m]
    for menu in nonempty_menus:
        mk = menu_to_key(menu)
        vals = [menus.get(mk, 0) for menus in menu_values_std.values()]
        if vals:
            print(f"  menu '{mk}': mean={np.mean(vals):.6e}, std={np.std(vals):.6e}")


if __name__ == '__main__':
    main()
