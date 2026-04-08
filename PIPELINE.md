# Pipeline Overview

This file summarizes the intended Owen workflow as captured in this repo.

## Phase I: Owen-value estimation

Main files:

- `pipeline/phase1/cal_owen_value.py`
- `pipeline/phase1/rgis_shapley.py`

Purpose:

- build expert- and projection-aware value signals
- estimate Owen / Shapley-style importance under the routed-expert setting
- produce menu-value style outputs that can be consumed by the assignment stage

Typical artifact:

- an Owen-value or menu-value JSON under a `shapley_results`-style directory

## Phase II: assignment and config generation

Main files:

- `pipeline/phase2/mckp_solver_deepseek.py`
- `pipeline/phase2/build_projection_quant_config.py`
- `pipeline/phase2/build_expert_config_global.py`

Purpose:

- solve the budgeted bit-assignment problem
- map value signals to projection-level or expert-level mixed-precision configs
- emit JSON configs usable by the quantization / evaluation stack

Typical artifact:

- a per-module bitwidth JSON config

## Comparison path

Main files:

- `pipeline/comparison/mcmoe_collect_deepseek_factors.py`
- `pipeline/comparison/mcmoe_solver_deepseek.py`

Purpose:

- collect MC-MoE-style factors
- produce the comparison-side allocation used in the experiments

## Evaluation path

Main files:

- `eval/vlmeval/vlm/deepseek_vl2.py`
- `eval/vlmeval/vlm/mixedbit_gptq_loader.py`

Purpose:

- load exported quantized checkpoints
- evaluate them through VLMEvalKit
- use `DEEPSEEK_VL2_TINY_MODEL_PATH` to point at the exported checkpoint

Important implementation detail:

- this repo uses the exported-checkpoint path for evaluation
- it does not depend on an online AutoRound runtime quantization path

## Benchmark notes

The current benchmark snapshot is in `RESULTS.md`.

Clear Owen wins observed in the current comparison snapshot include:

- BLINK
- VizWiz
- MME total

Several other tasks were close but still favored MC-MoE.
