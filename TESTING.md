# Small GPTQ 验证

当前回归入口：

```text
python scripts/run_tests.py --mode unit
```

或 python -X utf8 -m unittest discover -s tests -p test_small_gptq.py -v。
测试覆盖参考核心 AST、2/4-bit 候选整数码和解码数值、微型模型真实 GPTQ 建库与组装、
缓存估值对照、路由统计、精确预算、CLI 参数、多模态 dtype、OCRBench 单位转换和多种子汇总。
真实 Small 验收按 GPTQ_WORKFLOW.md，从新的工作目录逐阶段运行。
没有自动恢复、实验指纹或 AutoRound 测试入口。
