# Config-Owen: DeepSeek-VL2-Small

统一入口：`run_small.py`。

## Small 固定 GPTQ 流程

## 约定

- 仅 DeepSeek-VL2-Small，27 层、64 个 routed experts/层、hidden_size=2048。
- 使用 MC-MoE 的 `GPTQ.add_batch`、`static_fasterquant` 和 `quantizer_moe` 算术。
- 保留参考实现 `wbits > 3` 的块内补偿条件，不替换成另一份通用 GPTQ。
- group/block=128，damping=0.01，actorder=False，sym=False，perchannel=True，pack=True。
- scale/zero 搜索和 pack 模式均跟随参考实现，包括其非标准细节，不自行修正。
- Small MLA 的 kv_b_proj 直接参与矩阵收缩，Hessian 从 kv_a_layernorm 输出采集，避免漏掉普通 Linear hook。
- 每层按 attention、专家 gate/up、down 的依赖顺序收集 Hessian；两种候选从同一原始投影权重和同一 Hessian 出发。
- 候选生成的前层上下文为已量化的 4-bit anchor；生成后固定，不在菜单估值时重新校准。
- 非 router Linear/Conv2d 背景默认 GPTQ 4-bit，embeddings、norm、router 保持原始精度。
- 默认先处理视觉/投影背景，再冻结多模态 embeddings。该离线阶段可能耗时较长。
- 配置预算严格是 routed expert 参数加权平均，router/shared/attention/vision 不计入该预算。

## 环境

在已有兼容 CUDA PyTorch 的 GPU 环境中安装：

```text
pip install -r requirements_small_gptq.txt -r newest/VLMEvalKit/requirements.txt
```

PyTorch 与 xformers 必须是彼此兼容的 CUDA 构建；上面的依赖列表不能代替服务器环境验收。
不依赖 AutoRound，也不在缺少 CUDA 时回退到 RTN。
模型需为本地完整 Small 检查点，包含 processor/tokenizer 配置。

## 数据

默认 nsamples=128、seqlen=2048（每条最多保留 2048 个多模态语言 token）。
校准来源严格限制为 DocVQA、InfoVQA、ScienceQA、TextVQA、ChartQA，缺少任意一类或混入 POPE/MMVet 等来源会报错。
默认五类来源样本数为 26/26/26/25/25。
运行前从本地 LMUData 构建校准文件：

```text
python newest/data/build_multimodal_calib.py --lmudata /data/LMUData --output newest/data/multimodal_calib.jsonl --total_samples 128 --seed 0
```

构建器从本地 TSV 读取并提取嵌入图片，缺文件/图片时明确报错，不下载数据、不用别的来源补齐。
多模态 JSONL 每行应含 image_path 和 user/assistant conversations，格式沿用 newest/data 中的文件。
必须提供真实图片。旧 /.../LMUData/images/ 前缀可用 --image-root 替换。
无图片、无有效 assistant token、未被校准样本覆盖的目标层会直接报错，不静默跳过。

效用的数值稳定处理：同一次估值中的所有配置统一乘以相同的正系数，避免排列重要性权重溢出。
这保留配置间的相对关系及精确 MCKP 的最优组合。

## 运行

从仓库根目录，使用新的工作目录：

```text
python run_small.py all --model /models/deepseek-vl2-small --data /data/calibration.jsonl --image-root /data/LMUData/images --work-dir /data/small-run-01 --average-bits 2.6 --device cuda:0
```

all 默认依次运行 --seeds 0 1 2，每个种子使用独立子目录 seed_0、seed_1、seed_2。
全部完成后生成 evaluation_summary.json 和 evaluation_summary.csv，包含各次得分、均值和样本标准差。
默认五项 Avg 为 MMBench、MMStar、MMMU、AI2D、OCRBench 的百分制平均分；MME 单独汇总。
指定 --seeds 0 可只执行一次完整流程。

单次阶段命令使用同一个 --work-dir 和 --seed；处理 all 的某次输出时，将 --work-dir 指向对应 seed_N 子目录：

```text
python run_small.py bank --model /models/deepseek-vl2-small --data /data/calibration.jsonl --image-root /data/LMUData/images --work-dir /data/small-run-01
python run_small.py estimate --work-dir /data/small-run-01
python run_small.py solve --work-dir /data/small-run-01 --average-bits 2.6
python run_small.py assemble --work-dir /data/small-run-01
python run_small.py evaluate --work-dir /data/small-run-01 --benchmarks MME AI2D_TEST MMBench_DEV_EN_V11 MMStar MMMU_DEV_VAL OCRBench
```

estimate 默认使用 bank 中的校准张量；也可传 --data、--image-root 读取另一组估值数据。
估值只使用一个缓存/批处理实现，统一 assistant 标签、shift 和有效 token 加权。
排列采样使用 router weight mass 加正平滑，重要性概率在 log 域计算，不截断到 1e-10。
所有采样效用共同乘一个正的数值稳定系数，保持 MCKP 配置最优解；不将绝对效用值跨运行直接比较。
CASP 默认 0.5；排列默认 20。不提供额外前缀比例截断、自适应均值代填或断点实验指纹。
序列长度不同会使用 batch=1，保持同一条估值路径，避免错误 padding。

## 输出

- bank/manifest.json：完成标记、量化设置和候选模块清单。
- bank/candidate-*.pt：固定候选浮点值、整数码、scales、zeros、g_idx。
- bank/background.pt：同一份背景量化权重。
- bank/calibration.pt：固定背景计算的多模态 embeddings 和 labels。
- configuration_values.json：每个专家七个非空配置相对全低位基线的效用。
- assignment.json：完整 routed projection → 2/4 配置。
- assignment_summary.json：目标/实际 routed 平均位宽与精确 DP 结果。
- assembled/：同候选组装的数值模型和 processor。
- evaluation/metrics.json：单次各评测集得分，OCRBench 采用百分制。
- evaluation/：VLMEvalKit 推理结果和原始指标。
- evaluation_summary.json / evaluation_summary.csv：all 的各次得分、均值、样本标准差及五项 Avg。

评测固定从 bank 和 assignment 加载同样的权重，不重新运行 GPTQ。
没有自动续算；同一个目录保留结果不表示可以在不同输入之间混用。
assembled 保存浮点形式的量化数值权重。
