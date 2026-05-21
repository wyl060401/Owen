#!/usr/bin/env bash
set -euo pipefail

# Current DeepSeek-VL2 pipeline:
#   Phase I  - compute expert menu values
#   Phase II - solve the global MCKP assignment from menu values
#   Phase III - optionally run VLMEvalKit with the emitted mixed-precision config
#
# This script does not replace the legacy run_owen_pipeline*.sh scripts.
# It is the active entrypoint for the menu-values + MCKP method.

export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/MoE-Mixed-Prec-main}"
PYTHON="${PYTHON:-python}"
MODEL_ID="${MODEL_ID:-deepseek_vl2_tiny}"

case "$MODEL_ID" in
    deepseek_vl2_tiny)
        MODEL_DIRNAME="deepseek-vl2-tiny"
        MODEL_FILE_STEM="deepseek-vl2-tiny"
        HF_MODEL="deepseek-ai/deepseek-vl2-tiny"
        ;;
    deepseek_vl2_small)
        MODEL_DIRNAME="deepseek-vl2-small"
        MODEL_FILE_STEM="deepseek-vl2-small"
        HF_MODEL="deepseek-ai/deepseek-vl2-small"
        ;;
    deepseek_vl2_large)
        MODEL_DIRNAME="deepseek-vl2"
        MODEL_FILE_STEM="deepseek-vl2"
        HF_MODEL="deepseek-ai/deepseek-vl2"
        ;;
    *)
        echo "Unsupported MODEL_ID=${MODEL_ID}; expected deepseek_vl2_tiny, deepseek_vl2_small, or deepseek_vl2_large"
        exit 1
        ;;
esac

if [ -z "${MODEL:-}" ]; then
    if [ -n "${DEEPSEEK_VL2_MODEL_PATH:-}" ]; then
        MODEL="$DEEPSEEK_VL2_MODEL_PATH"
    elif [ -n "${DEEPSEEK_VL2_SMALL_MODEL_PATH:-}" ] && [ "$MODEL_ID" = "deepseek_vl2_small" ]; then
        MODEL="$DEEPSEEK_VL2_SMALL_MODEL_PATH"
    elif [ -n "${DEEPSEEK_VL2_TINY_MODEL_PATH:-}" ] && [ "$MODEL_ID" = "deepseek_vl2_tiny" ]; then
        MODEL="$DEEPSEEK_VL2_TINY_MODEL_PATH"
    elif [ -d "$PROJECT_ROOT/models/$MODEL_DIRNAME" ]; then
        MODEL="$PROJECT_ROOT/models/$MODEL_DIRNAME"
    else
        MODEL="$HF_MODEL"
    fi
fi

WORK_DIR="$PROJECT_ROOT/01_shapley_estimation/rgis_implementation"
EVAL_DIR="$PROJECT_ROOT/04_evaluation"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/shapley_results}"
CONFIG_DIR="${CONFIG_DIR:-$PROJECT_ROOT/02_precision_assignment/configs}"
LOG_DIR="${LOG_DIR:-$EVAL_DIR/logs}"
CACHE_DIR="${CACHE_DIR:-$HF_HOME}"
MENU_VALUES_FILE="${MENU_VALUES_FILE:-$OUTPUT_DIR/${MODEL_FILE_STEM}_menu_values.json}"
MCKP_PREFIX="${MCKP_PREFIX:-${MODEL_ID}_menu_mckp}"
EVAL_WORK_DIR="${EVAL_WORK_DIR:-./outputs_${MODEL_ID}_menu_mckp}"

RUN_MENU="${RUN_MENU:-true}"
RUN_MCKP="${RUN_MCKP:-true}"
RUN_EVAL="${RUN_EVAL:-false}"

NSAMPLES="${NSAMPLES:-128}"
SEQLEN="${SEQLEN:-128}"
MONTE_CARLO_SAMPLES="${MONTE_CARLO_SAMPLES:-20}"
TAU_TEMP="${TAU_TEMP:-0.7}"
LAMBDA_MIX="${LAMBDA_MIX:-0.1}"
EARLY_STOP="${EARLY_STOP:-0.5}"
MAX_EXPERT_FRACTION_PER_PERMUTATION="${MAX_EXPERT_FRACTION_PER_PERMUTATION:-1.0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"
DISABLE_CHECKPOINTS="${DISABLE_CHECKPOINTS:-false}"
NUM_GPUS="${NUM_GPUS:-2}"
START_LAYER="${START_LAYER:-1}"
SEED="${SEED:-0}"

CALIB_DATA="${CALIB_DATA:-multimodal}"
CALIB_DATA_PATH="${CALIB_DATA_PATH:-$PROJECT_ROOT/data/multimodal_calib.jsonl}"
FIXED_BITS="${FIXED_BITS:-4}"

TARGET_AVG_BITS="${TARGET_AVG_BITS:-2.60}"
MCKP_SOLVER="${MCKP_SOLVER:-dp}"
EVAL_BENCHMARKS="${EVAL_BENCHMARKS:-MME AI2D_TEST MMBench_DEV_EN_V11 MMStar MMMU_DEV_VAL}"
EVAL_GPU="${EVAL_GPU:-0}"
QUANT_FORMAT="${QUANT_FORMAT:-auto_round:auto_gptq}"
# 当 QUANT_FORMAT=gptq 时使用纯 GPTQ 量化器（与 PMQ 同一量化器，activation ordering 开启）
MODEL_PRECISION="${MODEL_PRECISION:-mixed_precision_quant}"
if [ "$QUANT_FORMAT" = "gptq" ]; then
    MODEL_PRECISION="gptq_quant"
fi

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

target_tag() {
    "$PYTHON" - "$1" <<'PY'
import sys
print(f"{float(sys.argv[1]):.2f}".replace(".", "p"))
PY
}

echo "============================================================"
echo "DeepSeek-VL2 menu + MCKP pipeline"
echo "Project root:       $PROJECT_ROOT"
echo "Model id:           $MODEL_ID"
echo "Model path/ref:     $MODEL"
echo "Menu values:        $MENU_VALUES_FILE"
echo "MCKP prefix:        $MCKP_PREFIX"
echo "Target avg bits:    $TARGET_AVG_BITS"
echo "Expert prefix frac: $MAX_EXPERT_FRACTION_PER_PERMUTATION"
echo "Eval batch size:    $EVAL_BATCH_SIZE"
echo "Checkpoints:        disabled=$DISABLE_CHECKPOINTS, dir=${CHECKPOINT_DIR:-auto}"
echo "Run menu/MCKP/eval: $RUN_MENU / $RUN_MCKP / $RUN_EVAL"
echo "============================================================"

if [ ! -d "$PROJECT_ROOT" ]; then
    echo "PROJECT_ROOT does not exist: $PROJECT_ROOT"
    exit 1
fi

mkdir -p "$OUTPUT_DIR" "$CONFIG_DIR" "$LOG_DIR"

"$PYTHON" - <<PY
from deepseek_vl2_model import get_model_spec, infer_model_config, resolve_model_path
project_root = r"$PROJECT_ROOT"
model_id = "$MODEL_ID"
model_ref = r"$MODEL"
spec = get_model_spec(model_id)
resolved = resolve_model_path(model_ref, project_root)
cfg = infer_model_config(resolved)
print(f"Metadata: layers={cfg.get('num_layers') or spec.num_layers}, experts={cfg.get('num_experts') or spec.num_experts}")
print(f"Resolved: {resolved}")
PY

if [ "$CALIB_DATA" = "multimodal" ] && [ ! -f "$CALIB_DATA_PATH" ]; then
    echo "Building multimodal calibration data: $CALIB_DATA_PATH"
    "$PYTHON" "$PROJECT_ROOT/data/build_multimodal_calib.py" \
        --output "$CALIB_DATA_PATH" \
        --total_samples "$NSAMPLES" \
        --seed 42
fi

cd "$WORK_DIR"

if [ "$RUN_MENU" = "true" ]; then
    echo ""
    echo "Phase I: compute menu values"
    MENU_CMD=(
        "$PYTHON" cal_owen_value.py
        --model "$MODEL"
        --nsamples "$NSAMPLES"
        --seqlen "$SEQLEN"
        --monte_carlo_samples "$MONTE_CARLO_SAMPLES"
        --tau_temp "$TAU_TEMP"
        --lambda_mix "$LAMBDA_MIX"
        --early_stop_threshold "$EARLY_STOP"
        --max_expert_fraction_per_permutation "$MAX_EXPERT_FRACTION_PER_PERMUTATION"
        --eval_batch_size "$EVAL_BATCH_SIZE"
        --seed "$SEED"
        --num_gpus "$NUM_GPUS"
        --cache_dir "$CACHE_DIR"
        --output_dir "$OUTPUT_DIR"
        --start_layer "$START_LAYER"
        --prequant_background
        --fixed_bits "$FIXED_BITS"
    )
    if [ -n "$CHECKPOINT_DIR" ]; then
        MENU_CMD+=(--checkpoint_dir "$CHECKPOINT_DIR")
    fi
    if [ "$DISABLE_CHECKPOINTS" = "true" ]; then
        MENU_CMD+=(--disable_checkpoints)
    fi
    if [ "$CALIB_DATA" = "multimodal" ]; then
        MENU_CMD+=(--calib_data multimodal --calib_data_path "$CALIB_DATA_PATH")
    else
        MENU_CMD+=(--calib_data c4)
    fi
    "${MENU_CMD[@]}"
fi

if [ "$RUN_MCKP" = "true" ]; then
    echo ""
    echo "Phase II: solve MCKP from menu values"
    if [ ! -f "$MENU_VALUES_FILE" ]; then
        echo "Menu values file not found: $MENU_VALUES_FILE"
        exit 1
    fi
    "$PYTHON" mckp_solver_deepseek.py \
        --model_id "$MODEL_ID" \
        --model_path "$MODEL" \
        --menu_values_path "$MENU_VALUES_FILE" \
        --target_avg_bits $TARGET_AVG_BITS \
        --solver "$MCKP_SOLVER" \
        --output_dir "$CONFIG_DIR" \
        --output_prefix "$MCKP_PREFIX" \
        --start_layer "$START_LAYER" \
        --fixed_bits "$FIXED_BITS" \
        --quant_attention \
        --quant_shared_experts \
        --quant_dense_ffn
fi

if [ "$RUN_EVAL" = "true" ]; then
    echo ""
    echo "Phase III: VLMEvalKit first-slice evaluation"
    cd "$EVAL_DIR/VLMEvalKit"
    for target in $TARGET_AVG_BITS; do
        tag="$(target_tag "$target")"
        config_path="$CONFIG_DIR/${MCKP_PREFIX}_avg${tag}.json"
        if [ ! -f "$config_path" ]; then
            echo "Config not found for target ${target}: $config_path"
            exit 1
        fi
        log_file="$LOG_DIR/${MODEL_ID}_menu_mckp_avg${tag}.log"
        echo "Evaluating target=${target}, config=${config_path}"
        CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$PYTHON" run.py \
            --model_precision="$MODEL_PRECISION" \
            --quant_format="$QUANT_FORMAT" \
            --json_file="$config_path" \
            --calib_source="$CALIB_DATA" \
            --work-dir "$EVAL_WORK_DIR" \
            --data $EVAL_BENCHMARKS \
            --model "$MODEL_ID" \
            --mode all \
            --verbose \
            2>&1 | tee "$log_file"
    done
fi

echo ""
echo "Done."
