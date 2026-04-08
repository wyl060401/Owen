# Owen Core Repo

This repository packages the core code path for the Owen mixed-precision
DeepSeek-VL2-Tiny workflow.

It is no longer just an eval-patch repo. The current version covers the main
steps we actually needed in practice:

1. Phase I Owen-value estimation
2. Phase II mixed-precision assignment / config generation
3. evaluation-side loading of exported quantized checkpoints
4. benchmark result snapshots against MC-MoE

This is still a trimmed repo. It intentionally excludes datasets, model weights,
quantized checkpoint directories, logs, caches, and large generated outputs.

## Included Components

### `configs/`

Owen precision-allocation JSON files used during search / selection.

### `pipeline/phase1/`

Core Owen-value estimation code.

- `cal_owen_value.py`
- `rgis_shapley.py`

These files cover the routed-expert valuation stage and the RGIS / Owen-value
estimation logic used to build the menu-value signal.

### `pipeline/phase2/`

Config generation and assignment logic.

- `mckp_solver_deepseek.py`
- `build_projection_quant_config.py`
- `build_expert_config_global.py`

These files cover the main path from value estimates to a usable projection- or
expert-level mixed-precision configuration.

### `pipeline/comparison/`

Comparison-side MC-MoE adaptation code kept for reproducibility.

- `mcmoe_collect_deepseek_factors.py`
- `mcmoe_solver_deepseek.py`

These are useful because a meaningful Owen repo also needs the comparison path
that was used in the head-to-head experiments.

### `eval/vlmeval/vlm/`

Evaluation-side loading code.

- `deepseek_vl2.py`
  - supports loading exported quantized checkpoints through
    `DEEPSEEK_VL2_TINY_MODEL_PATH`
  - includes the task-specific decoding caps used during benchmarking
- `mixedbit_gptq_loader.py`
  - restored mixed-bit GPTQ-format loader
  - includes 2/3/4/8-bit packed-weight support

### Root docs

- `README.md`
- `RESULTS.md`
- `PIPELINE.md`

## Upstream Dependencies

This repo assumes the following upstream codebases are available separately:

- `MoE-Mixed-Prec-main`
- `VLMEvalKit`
- `DeepSeek-VL2`

Several scripts here reference the larger project layout and are meant to be
read and reused with those upstream sources available.

## Notes

- The evaluation path used here is the exported-checkpoint path, not the online
  AutoRound runtime path.
- The intended workflow is to export quantized checkpoints first, then point
  evaluation to the exported directory with `DEEPSEEK_VL2_TINY_MODEL_PATH`.
- On the remote machine, dataset downloads were redirected to
  `LMUData=/root/autodl-tmp/LMUData` to avoid filling the system disk.

## Layout

```text
configs/
pipeline/
  phase1/
  phase2/
  comparison/
eval/vlmeval/vlm/
RESULTS.md
PIPELINE.md
```
