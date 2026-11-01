# Config-Owen

Configuration-valued mixed-precision quantization for routed experts.

The workflow builds fixed GPTQ candidates, estimates configuration utilities,
allocates precision with an exact multiple-choice knapsack solver, and assembles
and evaluates the selected weights.

## Repository layout

```text
owen/
  quantization/     GPTQ implementation, candidate bank, and candidate generation
  estimation/       Configuration utility estimation and permutation sampling
  allocation/       Exact MCKP precision allocation
  models/           Model loading, assembly, and transformer backend
  evaluation/       Score normalization and multi-seed aggregation
  cli.py            Workflow orchestration
evaluation/         Benchmark runner and VLMEvalKit scoring support
data/               Multimodal calibration data builder
run.py              Command-line entry point
requirements.txt    Runtime dependencies
```

## Installation

Use a CUDA environment with compatible PyTorch and xformers builds:

```bash
pip install -r requirements.txt
```

Provide a local DeepSeek-VL2-Small checkpoint, including tokenizer and processor
configuration files. Model loading uses local files.

## Calibration data

Prepare local VLMEvalKit TSVs for DocVQA, InfoVQA, ScienceQA, TextVQA, and ChartQA:

```bash
python data/build_multimodal_calib.py \
  --lmudata /data/LMUData \
  --output /data/calibration.jsonl \
  --total_samples 128 --seed 0
```

The default source counts are 26, 26, 26, 25, and 25. The builder extracts embedded
images when needed and rejects missing datasets or invalid images. Each JSONL
record contains `source`, `image_path`, and user/assistant `conversations`.
Use a new output filename when rebuilding the dataset.

## Run the workflow

```bash
python run.py all \
  --model /models/deepseek-vl2-small \
  --data /data/calibration.jsonl \
  --work-dir /data/owen-run \
  --average-bits 2.6 \
  --device cuda:0
```

The `all` command runs seeds 0, 1, and 2 in independent `seed_0`, `seed_1`, and
`seed_2` directories. Use `--seeds 0` for one complete run or supply another list
with `--seeds`. Choose a fresh work directory for a new experiment.

Run individual stages with the same work directory and seed:

```bash
python run.py bank --model /models/deepseek-vl2-small --data /data/calibration.jsonl --work-dir /data/owen-single --seed 0
python run.py estimate --work-dir /data/owen-single --seed 0
python run.py solve --work-dir /data/owen-single --average-bits 2.6
python run.py assemble --work-dir /data/owen-single
python run.py evaluate --work-dir /data/owen-single
```

For a run produced by `all`, point `--work-dir` at its `seed_N` directory.
The estimator reuses calibration tensors saved in the candidate bank by default.
Pass `--data` to prepare valuation inputs from a JSONL file. Use `--image-root`
to replace an old `LMUData/images` prefix when relocating images.

## Defaults

| Parameter | Default |
| --- | --- |
| Calibration samples | 128 |
| Maximum sequence length | 2048 |
| Candidate precision | 2 and 4 bits |
| GPTQ group / block size | 128 / 128 |
| Damping | 0.01 |
| Activation ordering | Disabled |
| Permutations | 20 |
| CASP threshold | 0.5 |
| Estimation batch size | 4; 1 for unequal sequence lengths |

The precision budget is the parameter-weighted average across routed expert
gate, up, and down projections. Candidate generation follows the MC-MoE GPTQ
arithmetic. Estimation, assembly, and evaluation reuse the saved candidates.

## Evaluation and outputs

Default benchmarks are MME, AI2D, MMBench, MMStar, MMMU, and OCRBench. `Avg` is
the mean percentage score across the latter five benchmarks; MME is reported
separately. Override the selection with `--benchmarks`.

Each seed directory contains:

- `bank/`: fixed candidates, background weights, calibration tensors, and manifest.
- `configuration_values.json`: utilities for each expert configuration.
- `assignment.json`: selected precision for every routed projection.
- `assignment_summary.json`: target and achieved average precision.
- `assembled/`: assembled model weights and processor.
- `evaluation/`: predictions, benchmark reports, and `metrics.json`.

After all seeds finish, `evaluation_summary.json` and `evaluation_summary.csv`
contain per-seed scores, means, and sample standard deviations.
