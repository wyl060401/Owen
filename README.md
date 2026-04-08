# Owen Core Repo

This repository packages the core artifacts for the Owen mixed-precision DeepSeek-VL2-Tiny workflow:

- Owen precision-allocation configs
- VLMEvalKit integration code for loading exported quantized checkpoints
- a small benchmark summary for Owen vs. MC-MoE

This is a lightweight companion repo, not a full training or evaluation environment.

## Included Files

- `configs/`
  - Owen precision-allocation JSON files used during search / selection
- `eval/vlmeval/vlm/deepseek_vl2.py`
  - evaluation-side model wrapper
  - supports loading exported quantized checkpoints through `DEEPSEEK_VL2_TINY_MODEL_PATH`
  - includes task-specific decoding caps used during benchmarking
- `eval/vlmeval/vlm/mixedbit_gptq_loader.py`
  - mixed-bit GPTQ-format loader used by the restored evaluation path
  - includes support for 2/3/4/8-bit packed weights
- `RESULTS.md`
  - benchmark snapshot from the current Owen vs. MC-MoE comparison

## Upstream Dependencies

This repo assumes the following upstream codebases are available separately:

- `MoE-Mixed-Prec-main`
- `VLMEvalKit`
- `DeepSeek-VL2`

Model weights, datasets, logs, quantized checkpoint directories, and benchmark caches are intentionally excluded.

## Notes

- The evaluation path used here is the exported-checkpoint path, not the online AutoRound runtime path.
- The intended workflow is to export quantized checkpoints first, then point evaluation to the exported directory with `DEEPSEEK_VL2_TINY_MODEL_PATH`.
- Dataset downloads on the remote machine were moved to `LMUData=/root/autodl-tmp/LMUData` to avoid filling the system disk.

## Layout

```text
configs/
eval/vlmeval/vlm/deepseek_vl2.py
eval/vlmeval/vlm/mixedbit_gptq_loader.py
RESULTS.md
```
