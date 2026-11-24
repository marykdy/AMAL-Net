#!/usr/bin/env bash
# Train one model/dataset/seed, then evaluate best-val checkpoints on test.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash run_one.sh amal liver 42
#   CUDA_VISIBLE_DEVICES=1 bash run_one.sh uctransnet pancreas 42 --epochs 300
#
# Environment variables shared with evaluate.py:
#   DATA_ROOT, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS,
#   UCTRANSNET_REPO, UCTRANSNET_IMPL, UCTRANSNET_BASE_CH,
#   NNUNET_IMPL, HD95=1
#
# Do not pass --out-dir in the extra arguments; the output root is RUNS_DIR.
set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "Usage: CUDA_VISIBLE_DEVICES=N bash run_one.sh MODEL DATASET SEED [train args...]" >&2
    exit 2
fi

MODEL="$1"
DATASET="$2"
SEED="$3"
shift 3
TRAIN_ARGS=("$@")

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNS_DIR="${RUNS_DIR:-$ROOT/runs}"
RUN_DIR="$RUNS_DIR/${DATASET}_${MODEL}_s${SEED}"
PYTHON="${PYTHON:-python3}"
export PYTHONUNBUFFERED=1

# train.py receives the full extra-argument list; evaluate.py only reads these
# from the environment.  Mirror the common options so a manual one-line run
# behaves the same as a launcher-driven run.
for ((i = 0; i < ${#TRAIN_ARGS[@]}; i++)); do
    if [[ $((i + 1)) -lt ${#TRAIN_ARGS[@]} ]]; then
        case "${TRAIN_ARGS[$i]}" in
            --uctransnet-repo) export UCTRANSNET_REPO="${TRAIN_ARGS[$((i + 1))]}" ;;
            --uctransnet-impl) export UCTRANSNET_IMPL="${TRAIN_ARGS[$((i + 1))]}" ;;
            --uctransnet-base-ch) export UCTRANSNET_BASE_CH="${TRAIN_ARGS[$((i + 1))]}" ;;
            --data-root)       export DATA_ROOT="${TRAIN_ARGS[$((i + 1))]}" ;;
            --image-size)      export IMAGE_SIZE="${TRAIN_ARGS[$((i + 1))]}" ;;
            --batch-size)      export BATCH_SIZE="${TRAIN_ARGS[$((i + 1))]}" ;;
            --num-workers)     export NUM_WORKERS="${TRAIN_ARGS[$((i + 1))]}" ;;
            --nnunet-impl)     export NNUNET_IMPL="${TRAIN_ARGS[$((i + 1))]}" ;;
            --base-channels)   export BASE_CHANNELS="${TRAIN_ARGS[$((i + 1))]}" ;;
            --top-k)           export TOP_K="${TRAIN_ARGS[$((i + 1))]}" ;;
            --enlargement)     export ENLARGEMENT="${TRAIN_ARGS[$((i + 1))]}" ;;
            --halo)            export HALO="${TRAIN_ARGS[$((i + 1))]}" ;;
            --num-prototypes)  export NUM_PROTOTYPES="${TRAIN_ARGS[$((i + 1))]}" ;;
            --dropout)         export DROPOUT="${TRAIN_ARGS[$((i + 1))]}" ;;
            --channel-gate)    export CHANNEL_GATE=1 ;;
        esac
    fi
done

mkdir -p "$RUN_DIR"

echo "[run_one] model=$MODEL dataset=$DATASET seed=$SEED"
echo "[run_one] run_dir=$RUN_DIR"
echo "[run_one] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[run_one] train args: ${TRAIN_ARGS[*]:-<none>}"

"$PYTHON" "$ROOT/train.py" \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --seed "$SEED" \
    --out-dir "$RUNS_DIR" \
    "${TRAIN_ARGS[@]}" 2>&1 | tee "$RUN_DIR/train.log"

CKPT_TUMOR="$RUN_DIR/ckpt/best_tumor_dice.pth"
CKPT_MIOU="$RUN_DIR/ckpt/best_miou.pth"

if [[ ! -f "$CKPT_TUMOR" ]]; then
    echo "[run_one] ERROR: no best_tumor_dice checkpoint at $CKPT_TUMOR" >&2
    exit 1
fi

HD95_ARGS=()
if [[ "${HD95:-0}" == "1" ]]; then
    HD95_ARGS+=(--hd95)
fi

"$PYTHON" "$ROOT/evaluate.py" \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --checkpoint "$CKPT_TUMOR" \
    --split test \
    --seed "$SEED" \
    --out "$RUN_DIR/eval_test_best_tumor.json" \
    "${HD95_ARGS[@]}" 2>&1 | tee "$RUN_DIR/eval_test_best_tumor.log"

if [[ -f "$CKPT_MIOU" ]]; then
    "$PYTHON" "$ROOT/evaluate.py" \
        --model "$MODEL" \
        --dataset "$DATASET" \
        --checkpoint "$CKPT_MIOU" \
        --split test \
        --seed "$SEED" \
        --out "$RUN_DIR/eval_test_best_miou.json" \
        "${HD95_ARGS[@]}" 2>&1 | tee "$RUN_DIR/eval_test_best_miou.log"
fi

echo "[run_one] done: $RUN_DIR"
