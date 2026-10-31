# Current pipeline

当前主流程见 [GPTQ_WORKFLOW.md](GPTQ_WORKFLOW.md)。
唯一入口 run_small.py；唯一模型 DeepSeek-VL2-Small；预算是 routed expert 的平均位宽。
固定 MC-MoE GPTQ 候选 → 配置效用 → 精确 MCKP → 同候选组装 → VLMEvalKit。
旧目录中的其他方法/模型文件保留用于参考，不参与当前 Small 入口。
