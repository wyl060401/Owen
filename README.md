# Config-Owen: DeepSeek-VL2-Small GPTQ

当前唯一主入口是 `run_small.py`，仅支持 DeepSeek-VL2-Small。
平均位宽预算按 routed expert 的 gate/up/down 参数加权计算。

流程：MC-MoE GPTQ 预生成固定 2/4-bit 候选 → 缓存/批处理估值 → 精确 MCKP → 同候选组装 → VLMEvalKit。
估值、组装和评测不再次量化；不提供普通/快速两条路径，也不自动恢复旧实验。

完整命令和依赖见 [GPTQ_WORKFLOW.md](GPTQ_WORKFLOW.md)。

```text
python run_small.py all --model /path/to/deepseek-vl2-small --data newest/data/multimodal_calib.jsonl --work-dir /path/to/new-run --average-bits 2.6
```

也可分别执行 `bank`、`estimate`、`solve`、`assemble`、`evaluate`。
其中 `bank` 输出必须是新目录；修改输入后请使用新的运行目录，没有实验指纹或自动续算。
默认校准样本 128、序列上限 2048；五类来源为 DocVQA、InfoVQA、ScienceQA、TextVQA、ChartQA。
先按 GPTQ_WORKFLOW.md 构建校准数据，再运行主入口。
all 默认运行种子 0、1、2，可通过 --seeds 指定；结果分别保存在 seed_0、seed_1、seed_2。
默认评测 MME、AI2D、MMBench、MMStar、MMMU、OCRBench，并输出 evaluation_summary.json 和 evaluation_summary.csv。

候选文件保存量化权重、整数码、scale、zero 和 g_idx；assembled 保存组装后的模型与 processor。

```text
python -X utf8 -m unittest discover -s tests -p test_small_gptq.py -v
```

CPU 测试验证参考 GPTQ 算术对齐、候选生成与组装、配置端点值、路由采样和精确预算。
运行 Small 校准和完整 benchmark 需要模型、图片和兼容 CUDA 环境。

代码目录：

- `pipeline/small_gptq/`：GPTQ 候选、配置估值、MCKP、模型组装和结果汇总。
- `newest/VLMEvalKit/`：Small 模型后端、六项评测与评分依赖。
- `newest/data/`：五类校准数据的构建脚本。
- `tests/`：CPU 回归测试。
