"""
RGIS-IS Shapley值计算模块（方案B）
基于量化框架.md中的Phase I实现

coalition语义：S = high-bit专家集合，E\S = low-bit专家集合
V(S) = 1/PPL(S内专家4-bit, E\S内专家2-bit)
Progressive方向：从全high(S=E)逐步将专家降为low-bit

主要特性：
1. 路由先验的温度化与混合（tempered + mixture）
2. Plackett-Luce引导的重要性采样
3. 方案A的重要性权重计算（与论文一致）
4. 自归一化IS估计器（可选）
5. fake quantization实现4-bit/2-bit切换
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import time
from itertools import combinations
from typing import List, Dict, Tuple, Optional, FrozenSet
import copy

from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask


# ==================== Menu 工具函数 ====================

PROJ_NAMES_DEFAULT = ['gate_proj', 'up_proj', 'down_proj']


def enumerate_menus(proj_names=None):
    """
    枚举所有 2^n 个 menu（投影层子集）。
    每个 menu 是一个 frozenset，表示保留高精度的投影层集合。
    n=3 时共 8 个 menu：∅, {g}, {u}, {d}, {g,u}, {g,d}, {u,d}, {g,u,d}
    """
    if proj_names is None:
        proj_names = PROJ_NAMES_DEFAULT
    menus = []
    for r in range(len(proj_names) + 1):
        for combo in combinations(proj_names, r):
            menus.append(frozenset(combo))
    return menus


def menu_to_key(menu: FrozenSet[str]) -> str:
    """将 menu (frozenset) 转为 JSON 可序列化的字符串 key。空集返回空字符串。"""
    return ','.join(sorted(menu)) if menu else ''


def key_to_menu(key: str) -> FrozenSet[str]:
    """将字符串 key 转回 frozenset。"""
    return frozenset(key.split(',')) if key else frozenset()


def fake_quantize_weight(weight: torch.Tensor, bits: int, group_size: int = 128) -> torch.Tensor:
    """
    对权重做 round-to-nearest 对称 fake quantization（量化再反量化）。
    用于模拟 int{bits} 量化效果，不改变 tensor 形状和 dtype。
    直接在原始 dtype/device 上操作，避免不必要的 float() 转换和设备搬运。

    参数：
        weight: 原始权重，形状 [out, in]
        bits: 量化位宽（如 2 或 4）
        group_size: 分组大小，-1 表示按行量化

    返回：
        fake_quant_weight: 量化再反量化后的权重，与输入同 dtype/device
    """
    # 直接在原始 dtype 上操作（BF16/FP16 精度足够做 round-to-nearest）
    w = weight
    qmax = 2 ** (bits - 1) - 1

    if group_size > 0 and w.shape[1] >= group_size:
        out_features, in_features = w.shape
        num_groups = in_features // group_size
        w_grouped = w[:, :num_groups * group_size].reshape(out_features * num_groups, group_size)

        max_val = w_grouped.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
        scale = max_val / qmax

        w_q = (w_grouped / scale).round().clamp(-qmax, qmax)
        w_dq = (w_q * scale).reshape(out_features, num_groups * group_size)

        if in_features > num_groups * group_size:
            tail = w[:, num_groups * group_size:]
            max_tail = tail.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
            scale_tail = max_tail / qmax
            tail_q = (tail / scale_tail).round().clamp(-qmax, qmax)
            tail_dq = tail_q * scale_tail
            w_dq = torch.cat([w_dq, tail_dq], dim=1)
    else:
        max_val = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
        scale = max_val / qmax
        w_q = (w / scale).round().clamp(-qmax, qmax)
        w_dq = w_q * scale

    return w_dq


def get_expert_linear_modules(model, layer_id: int, expert_id: int):
    """
    获取指定专家的所有 Linear 层（gate_proj, up_proj, down_proj）。

    返回：
        dict: {子模块名: nn.Linear}
    """
    try:
        expert = model.language.model.layers[layer_id].mlp.experts[expert_id]
        linears = {}
        for name, module in expert.named_modules():
            if isinstance(module, nn.Linear):
                linears[name] = module
        return linears
    except (AttributeError, IndexError):
        return {}


class RGISShapleyEstimator:
    """RGIS-IS Shapley值估计器"""

    def __init__(
        self,
        num_experts: int,
        tau: float = 0.7,
        lambda_mix: float = 0.1,
        use_self_normalized: bool = True,
        early_stop_threshold: Optional[float] = None,
        bits_high: int = 4,
        bits_low: int = 2,
        group_size: int = 128,
        device: torch.device = torch.device("cuda:0")
    ):
        """
        参数：
            num_experts: 专家总数
            tau: 温度化参数，范围(0,1)，越小越平滑
            lambda_mix: 均匀混合比例，范围[0.05, 0.2]
            use_self_normalized: 是否使用自归一化IS
            early_stop_threshold: 早停阈值（相对于value_full），如0.5表示降到50%时停止
            bits_high: high-bit精度（如4）
            bits_low: low-bit精度（如2）
            group_size: 量化分组大小
            device: 计算设备
        """
        self.num_experts = num_experts
        self.tau = tau
        self.lambda_mix = lambda_mix
        self.use_self_normalized = use_self_normalized
        self.early_stop_threshold = early_stop_threshold
        self.bits_high = bits_high
        self.bits_low = bits_low
        self.group_size = group_size
        self.device = device
        self._loss_fct = torch.nn.CrossEntropyLoss()

    def compute_p_tilde(self, p_raw: torch.Tensor) -> torch.Tensor:
        """
        计算稳健化的采样权重 p_tilde

        公式：p_tilde_i = lambda*(1/N) + (1-lambda) * (p_i^tau) / (sum_j p_j^tau)

        参数：
            p_raw: 原始路由先验概率，形状 [num_experts]

        返回：
            p_tilde: 稳健化后的采样权重，形状 [num_experts]
        """
        N = len(p_raw)

        # 温度化
        p_tempered = torch.pow(p_raw, self.tau)

        # 归一化
        p_tempered_norm = p_tempered / p_tempered.sum()

        # 混合均匀分布
        uniform_prob = 1.0 / N
        p_tilde = self.lambda_mix * uniform_prob + (1 - self.lambda_mix) * p_tempered_norm

        return p_tilde

    def sample_plackett_luce(
        self,
        p_tilde: torch.Tensor,
        num_samples: int = 1
    ) -> torch.Tensor:
        """
        使用 p_tilde 作为采样权重生成 Plackett-Luce 排列。
        只生成排列，不计算概率。
        概率 Q(π) 在 estimate_shapley 中同样用 p_tilde 单独计算（IS 一致性要求）。

        参数：
            p_tilde: 稳健化采样权重，形状 [num_experts]
            num_samples: 采样数量

        返回：
            permutations: 排列，形状 [num_samples, num_experts]
        """
        num_experts = len(p_tilde)
        permutations = []

        for _ in range(num_samples):
            available = list(range(num_experts))
            permutation = []

            for _ in range(num_experts):
                current_probs = p_tilde[available]

                if current_probs.sum() == 0:
                    current_probs = torch.ones_like(current_probs)
                norm_probs = current_probs / current_probs.sum()

                chosen_idx = torch.multinomial(norm_probs, num_samples=1).item()
                chosen_expert = available.pop(chosen_idx)
                permutation.append(chosen_expert)

            permutations.append(torch.tensor(permutation))

        return torch.stack(permutations, dim=0)

    def compute_plackett_luce_probability(
        self,
        permutation: torch.Tensor,
        p_tilde: torch.Tensor
    ) -> float:
        """
        用 p_tilde 计算排列的 Plackett-Luce 概率 Q(π)。

        根据文档 3.2：Q(π) = ∏[p_tilde_π[t] / Σ(p_tilde_π[s], s=t..N)]

        IS 要求：采样分布和权重中的 Q 必须一致，都基于 p_tilde。

        参数：
            permutation: 排列，形状 [num_experts]
            p_tilde: 稳健化采样权重，形状 [num_experts]

        返回：
            probability: 排列概率 Q(π)（标量）
        """
        N = len(permutation)
        probability = 1.0

        for t in range(N):
            current_expert = permutation[t].item()
            p_current = p_tilde[current_expert].item()

            remaining_sum = sum(
                p_tilde[permutation[s].item()].item() for s in range(t, N)
            )

            if remaining_sum > 0:
                probability *= p_current / remaining_sum

        # clamp 防止为0
        return max(probability, 1e-10)

    def compute_importance_weights(
        self,
        probabilities: torch.Tensor
    ) -> torch.Tensor:
        """
        计算重要性权重，与论文原始实现完全一致。

        公式：w(pi) = (1/N) / Q(pi)，N 为专家总数

        注：理论上应为 w(pi) = (1/N!) / Q(pi)，但实现中省略了全局常数 1/N!，
        仅保留相对权重 (1/N) / Q(pi)。这不影响最终 Shapley 值的相对大小，
        因为所有权重都乘以相同的常数 N!/N = (N-1)!。

        参数：
            probabilities: 排列概率Q(pi)，形状 [num_samples]

        返回：
            weights: 重要性权重，形状 [num_samples]
        """
        N = self.num_experts
        weights = (1.0 / N) / probabilities
        return weights

    def estimate_shapley(
        self,
        model,
        inputs: List,
        permutations: torch.Tensor,
        p_tilde: torch.Tensor,
        num_experts_per_layer: int,
        start_layer: int,
        value_fn,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        估计Shapley值

        coalition语义：S = high-bit专家集合，E\S = low-bit专家集合
        V(S) = 1/PPL(S内专家bits_high-bit, E\S内专家bits_low-bit)
        Progressive方向：从全high(S=E)逐步将专家降为low-bit

        参数：
            model: FP16模型（通过fake quant模拟量化效果）
            inputs: 校准数据
            permutations: 排列，形状 [num_samples, num_experts]
            p_tilde: 稳健化采样权重，形状 [num_experts]，用于计算 Q(π)（与采样分布一致）
            num_experts_per_layer: 每层专家数
            start_layer: 起始层
            value_fn: 价值函数，接受(model, inputs)返回标量值

        返回：
            shapley_values: Shapley值，形状 [num_experts]
            shapley_values_sn: 自归一化Shapley值（如果启用），形状 [num_experts]
        """
        M = permutations.shape[0]
        N = self.num_experts

        # 初始化累积值
        shapley_sum = torch.zeros(N)
        weight_sum = 0.0
        if self.use_self_normalized:
            weighted_marginal_sum = torch.zeros(N)

        # 缓存所有专家的原始FP16权重
        print("缓存原始FP16权重...", flush=True)
        orig_weights = {}
        for global_id in range(N):
            layer_id = global_id // num_experts_per_layer + start_layer
            expert_id = global_id % num_experts_per_layer
            linears = get_expert_linear_modules(model, layer_id, expert_id)
            orig_weights[global_id] = {
                name: module.weight.data.clone()
                for name, module in linears.items()
            }

        def apply_fake_quant(global_id: int, bits: int):
            layer_id = global_id // num_experts_per_layer + start_layer
            expert_id = global_id % num_experts_per_layer
            linears = get_expert_linear_modules(model, layer_id, expert_id)
            for name, module in linears.items():
                module.weight.data = fake_quantize_weight(
                    orig_weights[global_id][name], bits, self.group_size
                )

        def restore_expert(global_id: int):
            layer_id = global_id // num_experts_per_layer + start_layer
            expert_id = global_id % num_experts_per_layer
            linears = get_expert_linear_modules(model, layer_id, expert_id)
            for name, module in linears.items():
                module.weight.data = orig_weights[global_id][name].clone()

        # 初始状态：所有专家施加 high-bit fake quant（S = E）
        print(f"初始化：所有专家施加 {self.bits_high}-bit fake quant...", flush=True)
        for global_id in range(N):
            apply_fake_quant(global_id, self.bits_high)

        # 评估 V(S=E)：全 high-bit 基准价值
        with torch.no_grad():
            value_full = value_fn(model, inputs)
        print(f"全 {self.bits_high}-bit 基准价值: {value_full:.6f}")

        # 蒙特卡洛采样
        for m in range(M):
            print(f"\n排列 {m+1}/{M}")
            permutation = permutations[m]

            perm_prob = self.compute_plackett_luce_probability(permutation, p_tilde)
            weight = self.compute_importance_weights(torch.tensor([perm_prob]))[0].item()
            print(f"  排列概率: {perm_prob:.2e}, 重要性权重: {weight:.2e}")

            # 重置：所有专家恢复 high-bit（S = E）
            for global_id in range(N):
                apply_fake_quant(global_id, self.bits_high)

            value_with_expert = value_full

            # 遍历排列：逐步将专家从 high-bit 降为 low-bit
            for n in range(N):
                if self.early_stop_threshold is not None:
                    if value_with_expert < self.early_stop_threshold * value_full:
                        print(f"  n={n} 早停（价值降至 {value_with_expert/value_full:.2%}）")
                        break

                expert_id = permutation[n].item()
                layer_id = expert_id // num_experts_per_layer + start_layer
                expert_in_layer_id = expert_id % num_experts_per_layer

                # 将该专家从 high-bit 降为 low-bit
                apply_fake_quant(expert_id, self.bits_low)

                with torch.no_grad():
                    value_without_expert = value_fn(model, inputs)

                marginal_contribution = value_with_expert - value_without_expert
                shapley_sum[expert_id] += weight * marginal_contribution
                if self.use_self_normalized:
                    weighted_marginal_sum[expert_id] += weight * marginal_contribution

                if n < 5 or n % 50 == 0:
                    print(f"  n={n}, expert={expert_id} (L{layer_id}.E{expert_in_layer_id}), "
                          f"价值={value_without_expert:.6f}, 边际={marginal_contribution:.6f}")

                value_with_expert = value_without_expert

            weight_sum += weight

        # 恢复原始FP16权重
        print("恢复原始FP16权重...")
        for global_id in range(N):
            restore_expert(global_id)

        shapley_values = shapley_sum / M
        shapley_values_sn = None
        if self.use_self_normalized:
            shapley_values_sn = weighted_marginal_sum / weight_sum if weight_sum > 0 else weighted_marginal_sum

        return shapley_values, shapley_values_sn

    def _capture_single_expert(self, model, inputs, expert_gid,
                               num_experts_per_layer, start_layer,
                               value_fn):
        """
        在当前模型状态（S_bg）下，对单个专家注册 hook 并搭载一次 value_fn 调用，
        同时完成端到端评估和该专家的输入/输出捕获。零额外前向开销。

        返回：
            cached_inp: Tensor on device [num_routed_tokens, hidden]，或 None
            ref_out:    Tensor on device [num_routed_tokens, hidden]，或 None
            value:      value_fn 的返回值（float）
        """
        layer_id = expert_gid // num_experts_per_layer + start_layer
        expert_id = expert_gid % num_experts_per_layer

        captured_inputs = []
        captured_outputs = []

        try:
            expert_module = model.language.model.layers[layer_id].mlp.experts[expert_id]
        except (AttributeError, IndexError):
            value = value_fn(model, inputs)
            return None, None, value

        def hook_fn(module, inp, out):
            x = inp[0] if isinstance(inp, tuple) else inp
            captured_inputs.append(x.detach())
            captured_outputs.append(out.detach())

        handle = expert_module.register_forward_hook(hook_fn)

        with torch.no_grad():
            value = value_fn(model, inputs)

        handle.remove()

        if captured_inputs:
            cached_inp = torch.cat(captured_inputs, dim=0)
            ref_out = torch.cat(captured_outputs, dim=0)
        else:
            cached_inp = None
            ref_out = None

        return cached_inp, ref_out, value

    def estimate_owen_value(
        self,
        model,
        inputs: List,
        permutations: torch.Tensor,
        p_tilde: torch.Tensor,
        num_experts_per_layer: int,
        start_layer: int,
        value_fn,
        proj_names: List[str] = None,
        use_activation_cache: bool = False,
    ) -> Dict[str, float]:
        """
        双层 Owen Value 估计：外层对专家排列 MC，内层对每个专家穷举 6 条投影层降级路径。
        直接输出所有 3N 个投影层的 Owen Value。

        参数：
            model: FP16模型
            inputs: 校准数据
            permutations: 专家级排列，形状 [M, N]
            p_tilde: 稳健化采样权重，形状 [N]
            num_experts_per_layer: 每层专家数
            start_layer: 起始层
            value_fn: 价值函数 (model, inputs) -> float
            proj_names: 投影层名称列表，默认 ['gate_proj', 'up_proj', 'down_proj']
            use_activation_cache: 是否启用加速一（激活缓存），用局部 MSE 代替内层端到端前向

        返回：
            owen_values: dict, key="L{layer}.E{expert}.{proj}", value=Owen Value
        """
        from itertools import permutations as iter_perms

        if proj_names is None:
            proj_names = ['gate_proj', 'up_proj', 'down_proj']

        M = permutations.shape[0]
        N = self.num_experts
        n_proj = len(proj_names)
        all_inner_perms = list(iter_perms(range(n_proj)))  # 3! = 6

        # 初始化：key = (global_expert_id, proj_name) -> 累积加权边际贡献
        owen_sum = {}
        for gid in range(N):
            for pn in proj_names:
                owen_sum[(gid, pn)] = 0.0

        # 缓存所有专家的原始 FP16 权重（按投影层名称索引）
        print("缓存原始FP16权重...")
        orig_weights = {}
        proj_module_cache = {}
        for global_id in range(N):
            layer_id = global_id // num_experts_per_layer + start_layer
            expert_id = global_id % num_experts_per_layer
            linears = get_expert_linear_modules(model, layer_id, expert_id)
            orig_weights[global_id] = {}
            proj_module_cache[global_id] = {}
            for name, module in linears.items():
                # 匹配投影层名称
                matched_proj = None
                for pn in proj_names:
                    if pn in name:
                        matched_proj = pn
                        break
                if matched_proj is None:
                    matched_proj = name
                orig_weights[global_id][matched_proj] = module.weight.data.detach().cpu().clone()
                proj_module_cache[global_id][matched_proj] = module

        def set_proj_bits(global_id: int, proj_name: str, bits: int):
            """设置单个投影层的量化位宽"""
            module = proj_module_cache[global_id][proj_name]
            module.weight.data = fake_quantize_weight(
                orig_weights[global_id][proj_name], bits, self.group_size
            )

        def set_expert_all_bits(global_id: int, bits: int):
            """设置专家所有投影层为同一位宽"""
            for pn in proj_names:
                if pn in proj_module_cache[global_id]:
                    set_proj_bits(global_id, pn, bits)

        def restore_expert(global_id: int):
            """恢复专家原始 FP16 权重"""
            for pn in proj_names:
                if pn in proj_module_cache[global_id]:
                    module = proj_module_cache[global_id][pn]
                    module.weight.data = orig_weights[global_id][pn].clone()

        # 初始状态：所有专家 high-bit
        print(f"初始化：所有专家施加 {self.bits_high}-bit fake quant...")
        for global_id in range(N):
            set_expert_all_bits(global_id, self.bits_high)

        # 评估 V(S=E)：全 high-bit 基准
        with torch.no_grad():
            value_full = value_fn(model, inputs)
        print(f"全 {self.bits_high}-bit 基准价值: {value_full:.6f}")

        # 预缓存第一个专家的激活（搭载在 value_full 评估上）的占位变量
        # 实际捕获在每条排列的循环中进行
        next_cached_inp = None
        next_ref_out = None

        # 蒙特卡洛外层采样
        for m in range(M):
            print(f"\n排列 {m+1}/{M}")
            permutation = permutations[m]

            perm_prob = self.compute_plackett_luce_probability(permutation, p_tilde)
            weight = self.compute_importance_weights(torch.tensor([perm_prob]))[0].item()
            print(f"  排列概率: {perm_prob:.2e}, IS权重: {weight:.2e}")

            # 重置：所有专家恢复 high-bit
            for global_id in range(N):
                set_expert_all_bits(global_id, self.bits_high)

            value_before_expert = value_full
            next_cached_inp = None
            next_ref_out = None

            # 对第一个专家：搭载捕获在一次前向上
            if use_activation_cache:
                first_gid = permutation[0].item()
                next_cached_inp, next_ref_out, _ = self._capture_single_expert(
                    model, inputs, first_gid,
                    num_experts_per_layer, start_layer, value_fn
                )

            # 沿排列逐个降级专家
            for n in range(N):
                # 提前截断
                if self.early_stop_threshold is not None:
                    if value_before_expert < self.early_stop_threshold * value_full:
                        print(f"  n={n} 早停（价值降至 {value_before_expert/value_full:.2%}）")
                        # 清理预取的缓存
                        if next_cached_inp is not None:
                            del next_cached_inp, next_ref_out
                            next_cached_inp = None
                            next_ref_out = None
                        break

                expert_gid = permutation[n].item()
                layer_id = expert_gid // num_experts_per_layer + start_layer
                expert_in_layer = expert_gid % num_experts_per_layer

                # === 内层穷举：6 条投影层降级路径 ===
                proj_marginals = {pn: 0.0 for pn in proj_names}

                # 取出当前专家的实时缓存（由上一轮搭载捕获，或首专家单独捕获）
                cached_inp = next_cached_inp
                ref_out = next_ref_out
                next_cached_inp = None
                next_ref_out = None

                use_cache_for_expert = (
                    use_activation_cache and cached_inp is not None
                )

                if use_cache_for_expert:
                    # [加速一] 局部 MSE 代替端到端前向
                    expert_module = model.language.model.layers[layer_id].mlp.experts[expert_in_layer]

                    for sigma in all_inner_perms:
                        set_expert_all_bits(expert_gid, self.bits_high)
                        mse_current = 0.0

                        for step, proj_idx in enumerate(sigma):
                            pn = proj_names[proj_idx]
                            if pn not in proj_module_cache[expert_gid]:
                                continue
                            set_proj_bits(expert_gid, pn, self.bits_low)
                            with torch.no_grad():
                                cur_out = expert_module(cached_inp)
                                mse_after = F.mse_loss(cur_out, ref_out).item()
                            marginal = mse_after - mse_current
                            proj_marginals[pn] += marginal
                            mse_current = mse_after

                    del cached_inp, ref_out

                    # === 归一化：MSE 尺度 → 1/PPL 尺度 ===
                    # 计算专家级 1/PPL 边际贡献
                    set_expert_all_bits(expert_gid, self.bits_low)

                    # 搭载捕获下一个专家的激活（零额外开销）
                    if n + 1 < N:
                        next_gid = permutation[n + 1].item()
                        next_cached_inp, next_ref_out, value_after_all_low = \
                            self._capture_single_expert(
                                model, inputs, next_gid,
                                num_experts_per_layer, start_layer, value_fn
                            )
                    else:
                        with torch.no_grad():
                            value_after_all_low = value_fn(model, inputs)

                    expert_marginal = value_before_expert - value_after_all_low
                    mse_total = sum(proj_marginals.values())

                    # 归一化到 1/PPL 尺度
                    if abs(mse_total) > 1e-12 and abs(expert_marginal) > 1e-12:
                        scale = expert_marginal / mse_total
                        for pn in proj_names:
                            proj_marginals[pn] *= scale
                    else:
                        # MSE 或专家边际贡献太小，说明该专家对性能无影响
                        # 将所有投影层边际贡献设为 0，保持量纲一致性
                        for pn in proj_names:
                            proj_marginals[pn] = 0.0

                    value_before_expert = value_after_all_low

                else:
                    # 原始路径：端到端前向传播
                    for sigma in all_inner_perms:
                        set_expert_all_bits(expert_gid, self.bits_high)
                        v_current = value_before_expert

                        for step, proj_idx in enumerate(sigma):
                            pn = proj_names[proj_idx]
                            if pn not in proj_module_cache[expert_gid]:
                                continue
                            set_proj_bits(expert_gid, pn, self.bits_low)
                            with torch.no_grad():
                                v_after = value_fn(model, inputs)
                            marginal = v_current - v_after
                            proj_marginals[pn] += marginal
                            v_current = v_after

                    # 将专家设为全 low-bit，搭载捕获下一个专家
                    set_expert_all_bits(expert_gid, self.bits_low)
                    if use_activation_cache and n + 1 < N:
                        next_gid = permutation[n + 1].item()
                        next_cached_inp, next_ref_out, value_before_expert = \
                            self._capture_single_expert(
                                model, inputs, next_gid,
                                num_experts_per_layer, start_layer, value_fn
                            )
                    else:
                        with torch.no_grad():
                            value_before_expert = value_fn(model, inputs)

                # 平均 6 条路径 → 条件 Shapley
                for pn in proj_names:
                    cond_shapley = proj_marginals[pn] / len(all_inner_perms)
                    owen_sum[(expert_gid, pn)] += weight * cond_shapley

                if n < 5 or n % 50 == 0:
                    print(f"  n={n}, expert={expert_gid} (L{layer_id}.E{expert_in_layer}), "
                          f"V_after={value_before_expert:.6f}"
                          + (" [cache]" if use_cache_for_expert else ""))

        # 恢复原始 FP16 权重
        print("恢复原始FP16权重...")
        for global_id in range(N):
            restore_expert(global_id)

        # 汇总 Owen Value
        owen_values = {}
        for global_id in range(N):
            layer_id = global_id // num_experts_per_layer + start_layer
            expert_id = global_id % num_experts_per_layer
            for pn in proj_names:
                key = f"L{layer_id}.E{expert_id}.{pn}"
                owen_values[key] = owen_sum[(global_id, pn)] / M

        return owen_values

    def estimate_menu_value(
        self,
        model,
        inputs: List,
        permutations: torch.Tensor,
        p_tilde: torch.Tensor,
        num_experts_per_layer: int,
        start_layer: int,
        value_fn,
        proj_names: List[str] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Menu-level value 估计：对每个 expert i 和每个非空 menu A，计算 H_i(A)。

        H_i^π(A) = V(background; i→A) - V(background; i→∅)

        相比 estimate_owen_value（6×3=18 次前向/expert），menu 方式只需
        1(baseline) + 7(非空menu) = 8 次前向/expert，约 2.4× 加速。

        参数：
            model, inputs, permutations, p_tilde, num_experts_per_layer,
            start_layer, value_fn: 同 estimate_owen_value
            proj_names: 投影层名称列表，默认 ['gate_proj', 'up_proj', 'down_proj']

        返回：
            menu_values: {expert_key: {menu_key: H_value}}
            expert_key = "L{layer}.E{expert}"
            menu_key = 逗号分隔的高精度投影层名（如 "gate_proj,up_proj"）
        """
        if proj_names is None:
            proj_names = PROJ_NAMES_DEFAULT

        menus = enumerate_menus(proj_names)
        empty_menu = frozenset()
        nonempty_menus = [m for m in menus if m != empty_menu]

        M = permutations.shape[0]
        N = self.num_experts

        # 累积器：(gid, menu_key) -> 加权 H 值之和
        menu_sum = {}
        for gid in range(N):
            for menu in nonempty_menus:
                menu_sum[(gid, menu_to_key(menu))] = 0.0

        # 缓存原始 FP16 权重和模块引用
        print("缓存原始FP16权重...")
        orig_weights = {}
        proj_module_cache = {}
        for global_id in range(N):
            layer_id = global_id // num_experts_per_layer + start_layer
            expert_id = global_id % num_experts_per_layer
            linears = get_expert_linear_modules(model, layer_id, expert_id)
            orig_weights[global_id] = {}
            proj_module_cache[global_id] = {}
            for name, module in linears.items():
                matched_proj = None
                for pn in proj_names:
                    if pn in name:
                        matched_proj = pn
                        break
                if matched_proj is None:
                    matched_proj = name
                orig_weights[global_id][matched_proj] = module.weight.data.detach().cpu().clone()
                proj_module_cache[global_id][matched_proj] = module

        def set_proj_bits(global_id: int, proj_name: str, bits: int):
            module = proj_module_cache[global_id][proj_name]
            module.weight.data = fake_quantize_weight(
                orig_weights[global_id][proj_name], bits, self.group_size
            )

        def set_expert_all_bits(global_id: int, bits: int):
            for pn in proj_names:
                if pn in proj_module_cache[global_id]:
                    set_proj_bits(global_id, pn, bits)

        def set_expert_menu(global_id: int, menu: FrozenSet[str]):
            """按 menu 设置专家：menu 内的投影层 high-bit，其余 low-bit"""
            for pn in proj_names:
                if pn in proj_module_cache[global_id]:
                    bits = self.bits_high if pn in menu else self.bits_low
                    set_proj_bits(global_id, pn, bits)

        def restore_expert(global_id: int):
            for pn in proj_names:
                if pn in proj_module_cache[global_id]:
                    module = proj_module_cache[global_id][pn]
                    module.weight.data = orig_weights[global_id][pn].clone()

        # 初始状态：所有专家 high-bit
        print(f"初始化：所有专家施加 {self.bits_high}-bit fake quant...")
        for global_id in range(N):
            set_expert_all_bits(global_id, self.bits_high)

        with torch.no_grad():
            value_full = value_fn(model, inputs)
        print(f"全 {self.bits_high}-bit 基准价值: {value_full:.6f}")

        # === 蒙特卡洛外层采样 ===
        total_start_time = time.time()
        dirty_experts = set()
        for m in range(M):
            perm_start_time = time.time()
            elapsed_total = perm_start_time - total_start_time
            print(f"\n排列 {m+1}/{M} (已用时 {elapsed_total:.1f}s)")
            permutation = permutations[m]

            perm_prob = self.compute_plackett_luce_probability(permutation, p_tilde)
            weight = self.compute_importance_weights(torch.tensor([perm_prob]))[0].item()
            print(f"  排列概率: {perm_prob:.2e}, IS权重: {weight:.2e}")

            # 重置所有专家到 high-bit
            for global_id in range(N):
                set_expert_all_bits(global_id, self.bits_high)

            value_before_expert = value_full

            # 沿排列逐个降级专家
            experts_evaluated = 0
            for n in range(N):
                if self.early_stop_threshold is not None:
                    if value_before_expert < self.early_stop_threshold * value_full:
                        print(f"  n={n} 早停（价值降至 {value_before_expert/value_full:.2%}）")
                        break

                expert_start = time.time()
                expert_gid = permutation[n].item()
                layer_id = expert_gid // num_experts_per_layer + start_layer
                expert_in_layer = expert_gid % num_experts_per_layer

                # baseline: expert 设为 ∅（全 low-bit）
                set_expert_all_bits(expert_gid, self.bits_low)
                with torch.no_grad():
                    v_base = value_fn(model, inputs)

                # 评估每个非空 menu
                for menu in nonempty_menus:
                    set_expert_menu(expert_gid, menu)
                    with torch.no_grad():
                        v_menu = value_fn(model, inputs)
                    H = v_menu - v_base
                    menu_sum[(expert_gid, menu_to_key(menu))] += weight * H

                # 保持 ∅ 状态继续降级
                set_expert_all_bits(expert_gid, self.bits_low)
                value_before_expert = v_base
                dirty_experts.add(expert_gid)
                experts_evaluated += 1

                expert_elapsed = time.time() - expert_start
                if n < 5 or n % 20 == 0:
                    # ETA 估算
                    perm_elapsed = time.time() - perm_start_time
                    avg_per_expert = perm_elapsed / experts_evaluated
                    remaining_this_perm = avg_per_expert * (N - n - 1)
                    remaining_future_perms = avg_per_expert * N * (M - m - 1)
                    eta_seconds = remaining_this_perm + remaining_future_perms
                    eta_min = eta_seconds / 60
                    print(f"  n={n}, expert={expert_gid} (L{layer_id}.E{expert_in_layer}), "
                          f"V={value_before_expert:.6f}, "
                          f"耗时={expert_elapsed:.2f}s, ETA≈{eta_min:.1f}min")

            perm_elapsed = time.time() - perm_start_time
            print(f"  排列 {m+1} 完成, 评估了 {experts_evaluated}/{N} 个专家, "
                  f"耗时 {perm_elapsed:.1f}s")

        # 恢复原始 FP16 权重
        total_elapsed = time.time() - total_start_time
        print(f"\n全部排列完成, 总耗时 {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
        print("恢复原始FP16权重...")
        for global_id in range(N):
            restore_expert(global_id)

        # 汇总 menu value
        menu_values = {}
        for global_id in range(N):
            layer_id = global_id // num_experts_per_layer + start_layer
            expert_id = global_id % num_experts_per_layer
            expert_key = f"L{layer_id}.E{expert_id}"
            menu_values[expert_key] = {}
            for menu in nonempty_menus:
                mk = menu_to_key(menu)
                menu_values[expert_key][mk] = menu_sum[(global_id, mk)] / M

        return menu_values

    # ==================== 加速 Phase I 的辅助函数 ====================

    @staticmethod
    def _prepare_attention_mask(lang_model, batch_size, seq_length, inputs_embeds):
        """
        复刻 DeepseekV2Model.forward 中的 attention_mask 准备逻辑。
        - flash_attention_2 且无 padding → None
        - 标准 attention → 4D causal mask

        参数：
            lang_model: DeepseekV2ForCausalLM（即 model.language）
            batch_size: int
            seq_length: int
            inputs_embeds: [B, seq_len, hidden_dim]，用于推断 dtype/device

        返回：
            attention_mask: None 或 [B, 1, seq_len, seq_len] 的 4D causal mask
        """
        use_flash = getattr(lang_model.model, '_use_flash_attention_2', False)
        if use_flash:
            # flash attention: 无 padding 时传 None（与原始逻辑一致）
            return None
        else:
            # 标准 attention: 构造 4D causal mask
            return _prepare_4d_causal_attention_mask(
                None,  # 无外部 attention_mask（校准数据无 padding）
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length=0,
            )

    @staticmethod
    def _forward_prefix(model, inputs_embeds, attention_mask, position_ids,
                        end_layer: int):
        """
        运行 transformer 的前 end_layer 层（层 0 到 end_layer-1），返回中间 hidden_states。
        不修改模型，不计算 loss。严格等价于完整前向的前半段。

        参数：
            model: DeepseekV2ForCausalLM（即 model.language）
            inputs_embeds: [B, seq_len, hidden_dim]
            attention_mask: 预计算的 4D causal mask 或 None
            position_ids: [1, seq_len]
            end_layer: 前缀终止层（不含），即运行层 0..end_layer-1

        返回：
            hidden_states: [B, seq_len, hidden_dim]，层 end_layer 的输入
        """
        hidden_states = inputs_embeds
        for idx in range(end_layer):
            layer = model.model.layers[idx]
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
            )
            hidden_states = layer_outputs[0]
        return hidden_states

    @staticmethod
    def _forward_moe_only(layer, mlp_input, residual):
        """
        只运行 decoder layer 的 MoE/MLP 部分并加回残差。
        输入为 _forward_layer_attn_only 的输出。

        返回：
            hidden_states: MoE 输出 + 残差，即该层的完整输出
        """
        moe_out = layer.mlp(mlp_input)
        return residual + moe_out

    @staticmethod
    def _forward_layer_attn_only(layer, hidden_states, attention_mask, position_ids):
        """
        只运行 decoder layer 的 attention + residual + post_attention_layernorm 部分。
        返回 MoE/MLP 的输入（layernorm 后）和残差（用于后续加回）。

        返回：
            mlp_input: post_attention_layernorm 的输出，即 MoE 的输入
            residual: attention 输出 + 原始残差，用于 MoE 输出后加回
        """
        # 复制 DeepseekV2DecoderLayer.forward 的前半段
        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)
        hidden_states, _, _ = layer.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
        )
        hidden_states = residual + hidden_states  # attention residual

        residual = hidden_states
        mlp_input = layer.post_attention_layernorm(hidden_states)
        return mlp_input, residual

    @staticmethod
    def _forward_suffix(model, hidden_states, attention_mask, position_ids,
                        start_layer: int, labels):
        """
        运行 transformer 的后半段（层 start_layer 到最后一层），加上 norm + lm_head + loss。
        严格等价于完整前向的后半段。

        参数：
            model: DeepseekV2ForCausalLM（即 model.language）
            hidden_states: [B, seq_len, hidden_dim]，层 start_layer 的输入
            attention_mask: 预计算的 4D causal mask 或 None
            position_ids: [1, seq_len]
            start_layer: 后缀起始层（含）
            labels: [B, seq_len]，用于计算 loss

        返回：
            ppl: float，困惑度
        """
        num_layers = len(model.model.layers)
        for idx in range(start_layer, num_layers):
            layer = model.model.layers[idx]
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
            )
            hidden_states = layer_outputs[0]

        hidden_states = model.model.norm(hidden_states)
        logits = model.lm_head(hidden_states).float()

        # 计算 loss（与 DeepseekV2ForCausalLM.forward 一致）
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_logits = shift_logits.view(-1, logits.shape[-1])
        shift_labels = shift_labels.view(-1).to(shift_logits.device)
        loss = self._loss_fct(shift_logits, shift_labels)

        n_tokens = (labels[..., 1:] != -100).sum().item()
        return loss.item(), max(n_tokens, 1)

    def _eval_from_moe_cached(self, lang_model, cached_attn_batches, layer_id, device):
        """
        从缓存的 layer attention 输出开始，运行 MoE + suffix，返回 1/PPL。
        cached_attn_batches: list of (mlp_input, residual, labels, position_ids, attn_mask)
        """
        total_loss = 0.0
        total_tokens = 0
        layer = lang_model.model.layers[layer_id]
        for mlp_input, residual, labels, position_ids, attn_mask in cached_attn_batches:
            hidden = self._forward_moe_only(layer, mlp_input, residual)
            loss, n_tok = self._forward_suffix(
                lang_model, hidden, attn_mask, position_ids,
                start_layer=layer_id + 1, labels=labels,
            )
            total_loss += loss * n_tok
            total_tokens += n_tok
        avg_loss = total_loss / max(total_tokens, 1)
        ppl = math.exp(min(avg_loss, 100))
        return 1.0 / ppl

    def _eval_all_menus_batched(
        self, lang_model, cached_attn_batches, layer_id,
        expert_gid, all_menus, set_expert_menu_fn, set_expert_all_bits_fn,
    ):
        """
        将 baseline(∅) + 7 个非空 menu 的 MoE 输出沿 batch 维拼接，
        suffix 只跑一次，返回每个 menu 的 1/PPL。

        all_menus: [frozenset(), menu1, menu2, ...] 共 8 个，第 0 个是 ∅
        cached_attn_batches: list of (mlp_input, residual, labels, position_ids, attn_mask)
        返回: list of float, 长度 = len(all_menus)
        """
        n_menus = len(all_menus)
        layer = lang_model.model.layers[layer_id]

        total_loss = [0.0] * n_menus
        total_tokens = [0] * n_menus

        menu_chunk_size = n_menus
        for mlp_input, residual, labels, position_ids, attn_mask in cached_attn_batches:
            B = labels.shape[0]

            for chunk_start in range(0, n_menus, menu_chunk_size):
                menu_chunk = all_menus[chunk_start:chunk_start + menu_chunk_size]
                chunk_n = len(menu_chunk)
                moe_hiddens = []

                for menu in menu_chunk:
                    if menu:  # 非空 menu
                        set_expert_menu_fn(expert_gid, menu)
                    else:  # ∅ = 全 low-bit
                        set_expert_all_bits_fn(expert_gid, self.bits_low)
                    h = self._forward_moe_only(layer, mlp_input, residual)
                    moe_hiddens.append(h)

                # [chunk_n * B, seq_len, hidden_dim]
                stacked_hidden = torch.cat(moe_hiddens, dim=0)
                stacked_labels = labels.repeat(chunk_n, 1)
                stacked_pos = position_ids

                if attn_mask is not None:
                    stacked_attn_mask = attn_mask.repeat(chunk_n, 1, 1, 1)
                else:
                    stacked_attn_mask = None

                # suffix: layers L+1..end → norm → lm_head
                h = stacked_hidden
                num_layers = len(lang_model.model.layers)
                for idx in range(layer_id + 1, num_layers):
                    ly = lang_model.model.layers[idx]
                    h = ly(h, attention_mask=stacked_attn_mask, position_ids=stacked_pos,
                           past_key_value=None, output_attentions=False, use_cache=False)[0]
                h = lang_model.model.norm(h)
                logits = lang_model.lm_head(h)

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = stacked_labels[..., 1:].contiguous()

                for local_idx in range(chunk_n):
                    mi = chunk_start + local_idx
                    sl = shift_logits[local_idx * B:(local_idx + 1) * B]
                    slb = shift_labels[local_idx * B:(local_idx + 1) * B]
                    mask = (slb != -100)
                    n_tok = mask.sum().item()
                    if n_tok > 0:
                        loss = self._loss_fct(sl.float().reshape(-1, sl.shape[-1]),
                                              slb.reshape(-1).to(sl.device))
                        total_loss[mi] += loss.item() * n_tok
                        total_tokens[mi] += n_tok

                del moe_hiddens, stacked_hidden, stacked_labels, stacked_attn_mask
                del h, logits, shift_logits, shift_labels

        values = []
        for mi in range(n_menus):
            avg = total_loss[mi] / max(total_tokens[mi], 1)
            ppl = math.exp(min(avg, 100))
            values.append(1.0 / ppl)
        return values

    def estimate_menu_value_fast(
        self,
        model,
        inputs: List,
        permutations: torch.Tensor,
        p_tilde: torch.Tensor,
        num_experts_per_layer: int,
        start_layer: int,
        value_fn,
        proj_names: List[str] = None,
        batch_size: int = 4,
    ) -> Dict[str, Dict[str, float]]:
        """
        前缀共享加速版 menu value 估计。
        对 expert i (layer L) 的 8 个 menu，prefix (layers 0..L-1) 和
        layer L attention 只算一次，仅重跑 MoE + suffix。
        同层连续 expert 共享 prefix + attention 缓存。

        接口与 estimate_menu_value 完全一致，可直接替换。
        """
        if proj_names is None:
            proj_names = PROJ_NAMES_DEFAULT

        menus = enumerate_menus(proj_names)
        empty_menu = frozenset()
        nonempty_menus = [m for m in menus if m != empty_menu]

        M = permutations.shape[0]
        N = self.num_experts
        device = self.device
        lang_model = model.language  # DeepseekV2ForCausalLM

        # 累积器
        menu_sum = {}
        for gid in range(N):
            for menu in nonempty_menus:
                menu_sum[(gid, menu_to_key(menu))] = 0.0

        # 缓存原始权重 + 模块引用
        print("缓存原始FP16权重...")
        orig_weights = {}
        proj_module_cache = {}
        current_bits_state = {}
        for global_id in range(N):
            layer_id = global_id // num_experts_per_layer + start_layer
            expert_id = global_id % num_experts_per_layer
            linears = get_expert_linear_modules(model, layer_id, expert_id)
            orig_weights[global_id] = {}
            proj_module_cache[global_id] = {}
            current_bits_state[global_id] = {}
            for name, module in linears.items():
                matched_proj = None
                for pn in proj_names:
                    if pn in name:
                        matched_proj = pn
                        break
                if matched_proj is None:
                    matched_proj = name
                orig_weights[global_id][matched_proj] = module.weight.data.detach().cpu().clone()
                proj_module_cache[global_id][matched_proj] = module
                current_bits_state[global_id][matched_proj] = None

        cached_orig_layer_id = None
        cached_orig_layer_weights = {}

        def ensure_layer_orig_weights_on_device(layer_id):
            nonlocal cached_orig_layer_id, cached_orig_layer_weights
            if cached_orig_layer_id == layer_id:
                return

            cached_orig_layer_weights = {}
            layer_offset = layer_id - start_layer
            layer_base_gid = layer_offset * num_experts_per_layer
            for expert_id in range(num_experts_per_layer):
                global_id = layer_base_gid + expert_id
                if global_id not in orig_weights:
                    continue
                cached_orig_layer_weights[global_id] = {}
                for proj_name, cpu_weight in orig_weights[global_id].items():
                    module = proj_module_cache[global_id][proj_name]
                    cached_orig_layer_weights[global_id][proj_name] = cpu_weight.to(
                        device=module.weight.device,
                        dtype=module.weight.dtype,
                        non_blocking=True,
                    )
            cached_orig_layer_id = layer_id

        def set_proj_bits(global_id, proj_name, bits):
            if current_bits_state[global_id][proj_name] == bits:
                return
            module = proj_module_cache[global_id][proj_name]
            layer_id = global_id // num_experts_per_layer + start_layer
            ensure_layer_orig_weights_on_device(layer_id)
            orig_weight = cached_orig_layer_weights[global_id][proj_name]
            module.weight.data = fake_quantize_weight(
                orig_weight, bits, self.group_size)
            current_bits_state[global_id][proj_name] = bits

        def set_expert_all_bits(global_id, bits):
            for pn in proj_names:
                if pn in proj_module_cache[global_id]:
                    set_proj_bits(global_id, pn, bits)

        def set_expert_menu(global_id, menu):
            for pn in proj_names:
                if pn in proj_module_cache[global_id]:
                    bits = self.bits_high if pn in menu else self.bits_low
                    set_proj_bits(global_id, pn, bits)

        def restore_expert(global_id):
            for pn in proj_names:
                if pn in proj_module_cache[global_id]:
                    module = proj_module_cache[global_id][pn]
                    layer_id = global_id // num_experts_per_layer + start_layer
                    ensure_layer_orig_weights_on_device(layer_id)
                    module.weight.data = cached_orig_layer_weights[global_id][pn]
                    current_bits_state[global_id][pn] = None

        # 预构建 batch 列表
        is_embeds = (inputs[0][0].dim() == 3)
        batches = []
        for i in range(0, len(inputs), batch_size):
            be = torch.cat([inputs[j][0] for j in range(i, min(i + batch_size, len(inputs)))], dim=0).contiguous()
            bl = torch.cat([inputs[j][1] for j in range(i, min(i + batch_size, len(inputs)))], dim=0).contiguous()
            batches.append((be, bl))

        # 初始化：所有专家 high-bit
        print(f"初始化：所有专家施加 {self.bits_high}-bit fake quant...")
        for global_id in range(N):
            set_expert_all_bits(global_id, self.bits_high)

        print("开始计算全 high-bit 基准价值...", flush=True)
        with torch.no_grad():
            value_full = value_fn(model, inputs)
        print(f"全 {self.bits_high}-bit 基准价值: {value_full:.6f}", flush=True)

        total_start_time = time.time()
        dirty_experts = set()
        for m in range(M):
            perm_start = time.time()
            elapsed_total = perm_start - total_start_time
            print(f"\n排列 {m+1}/{M} (已用时 {elapsed_total:.1f}s)", flush=True)
            permutation = permutations[m]

            perm_prob = self.compute_plackett_luce_probability(permutation, p_tilde)
            weight = self.compute_importance_weights(torch.tensor([perm_prob]))[0].item()
            print(f"  排列概率: {perm_prob:.2e}, IS权重: {weight:.2e}", flush=True)

            # 第一个排列直接复用初始化后的 all-high 状态，避免重复做一遍全量 fake quant。
            if m == 0:
                print("  复用初始化后的 all-high 状态，跳过首次全量重置", flush=True)
            else:
                for global_id in dirty_experts:
                    set_expert_all_bits(global_id, self.bits_high)
                dirty_experts.clear()

            value_before_expert = value_full
            cached_layer_id = -1
            cached_attn_batches = None
            experts_evaluated = 0

            for n in range(N):
                if self.early_stop_threshold is not None:
                    if value_before_expert < self.early_stop_threshold * value_full:
                        print(f"  n={n} 早停（价值降至 {value_before_expert/value_full:.2%}）", flush=True)
                        break

                expert_start = time.time()
                expert_gid = permutation[n].item()
                layer_id = expert_gid // num_experts_per_layer + start_layer
                expert_in_layer = expert_gid % num_experts_per_layer

                # 如果换层了，重新计算 prefix + attention 缓存
                if layer_id != cached_layer_id:
                    cached_attn_batches = []
                    with torch.no_grad():
                        for be_cpu, bl_cpu in batches:
                            be = be_cpu.to(device, non_blocking=True)
                            bl = bl_cpu.to(device, non_blocking=True)
                            if is_embeds:
                                embeds = be
                            else:
                                embeds = lang_model.model.embed_tokens(be)
                            B_cur = embeds.shape[0]
                            seq_len = embeds.shape[1]
                            pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
                            attn_mask = self._prepare_attention_mask(
                                lang_model, B_cur, seq_len, embeds)
                            hidden = self._forward_prefix(lang_model, embeds, attn_mask, pos_ids, layer_id)
                            layer_module = lang_model.model.layers[layer_id]
                            mlp_inp, res = self._forward_layer_attn_only(layer_module, hidden, attn_mask, pos_ids)
                            cached_attn_batches.append((mlp_inp, res, bl, pos_ids, attn_mask))
                    cached_layer_id = layer_id

                # baseline(∅) + 7 个非空 menu 一次性 batch 评估
                all_menus_with_empty = [frozenset()] + list(nonempty_menus)
                with torch.no_grad():
                    all_values = self._eval_all_menus_batched(
                        lang_model, cached_attn_batches, layer_id,
                        expert_gid, all_menus_with_empty,
                        set_expert_menu, set_expert_all_bits,
                    )
                v_base = all_values[0]
                for mi, menu in enumerate(nonempty_menus):
                    H = all_values[mi + 1] - v_base
                    menu_sum[(expert_gid, menu_to_key(menu))] += weight * H

                # 保持 ∅ 状态继续降级，作废当前层缓存（MoE 输出已变）
                set_expert_all_bits(expert_gid, self.bits_low)
                # 用 v_base 作为下一个 expert 的 value_before（与原版语义一致）
                value_before_expert = v_base
                dirty_experts.add(expert_gid)
                experts_evaluated += 1
                # 下一个 expert 若在不同层，prefix 需重算；同层则缓存仍有效
                if n + 1 < N:
                    next_gid = permutation[n + 1].item()
                    next_layer = next_gid // num_experts_per_layer + start_layer
                    if next_layer != layer_id:
                        cached_layer_id = -1  # 强制重算

                expert_elapsed = time.time() - expert_start
                if n < 5 or n % 20 == 0:
                    perm_elapsed = time.time() - perm_start
                    avg_per = perm_elapsed / experts_evaluated
                    eta = avg_per * (N - n - 1) + avg_per * N * (M - m - 1)
                    print(f"  n={n}, expert={expert_gid} (L{layer_id}.E{expert_in_layer}), "
                          f"V={value_before_expert:.6f}, "
                          f"耗时={expert_elapsed:.2f}s, ETA≈{eta/60:.1f}min [fast]", flush=True)

            perm_elapsed = time.time() - perm_start
            print(f"  排列 {m+1} 完成, 评估了 {experts_evaluated}/{N} 个专家, "
                  f"耗时 {perm_elapsed:.1f}s", flush=True)

        # 恢复原始 FP16 权重
        total_elapsed = time.time() - total_start_time
        print(f"\n全部排列完成, 总耗时 {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)", flush=True)
        print("恢复原始FP16权重...", flush=True)
        for global_id in range(N):
            restore_expert(global_id)

        # 释放缓存
        del cached_attn_batches

        # 汇总 menu value
        menu_values = {}
        for global_id in range(N):
            layer_id = global_id // num_experts_per_layer + start_layer
            expert_id = global_id % num_experts_per_layer
            expert_key = f"L{layer_id}.E{expert_id}"
            menu_values[expert_key] = {}
            for menu in nonempty_menus:
                mk = menu_to_key(menu)
                menu_values[expert_key][mk] = menu_sum[(global_id, mk)] / M

        return menu_values
