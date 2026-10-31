#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
exec "${PYTHON:-python}" "$ROOT/run_small.py" all   --model "${MODEL:?Set MODEL to your local DeepSeek-VL2-Small checkpoint}"   --data "${CALIB_DATA_PATH:?Set CALIB_DATA_PATH to your JSONL}"   --work-dir "${WORK_DIR:-$ROOT/small_gptq_runs}"   --average-bits "${TARGET_AVG_BITS:-2.6}" "$@"
