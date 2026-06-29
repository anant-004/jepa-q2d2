#!/bin/bash
# Diagnostic sweep: lambda_sigreg ∈ {0.01, 0.05, 0.1, 0.5, 1.0}
# 5K steps each, GPUs 1 & 2, eval every 500 steps.
# Shorter audio (5s) for faster iterations.
#
# Usage: bash paper/scripts/sweep_sigreg.sh

set -euo pipefail

DATA_DIR="/local_data/data/librilight/librilight_9k"
OUTPUT_BASE="/local_data/checkpoints/sigreg_sweep"
STRIDES="4,4,4,5,3"
STEPS=5000
BATCH=16
LR="5e-5"
MAX_SEC=5
EVAL_EVERY=500
SAVE_EVERY=2500

mkdir -p "$OUTPUT_BASE"

LAMBDAS=(0.01 0.05 0.1 0.5 1.0)
GPUS=(1 2)

echo "=== SIGReg Diagnostic Sweep ==="
echo "Lambdas: ${LAMBDAS[*]}"
echo "GPUs: ${GPUS[*]}"
echo "Steps: $STEPS, Batch: $BATCH, Audio: ${MAX_SEC}s"
echo ""

run_one() {
    local GPU=$1
    local LAMBDA=$2
    local NAME="sigreg_lambda${LAMBDA}"
    local OUT="$OUTPUT_BASE/$NAME"
    mkdir -p "$OUT"

    echo "[GPU $GPU] Launching $NAME (lambda=$LAMBDA)"
    CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -m paper.scripts.train_sigreg_stage1 \
        --data_dir "$DATA_DIR" \
        --output_dir "$OUT" \
        --run_name "$NAME" \
        --strides "$STRIDES" \
        --stage1_steps $STEPS \
        --batch_size $BATCH \
        --lr $LR \
        --max_seconds $MAX_SEC \
        --lambda_sigreg $LAMBDA \
        --eval_every $EVAL_EVERY \
        --save_every $SAVE_EVERY \
        --no_compile \
        > "$OUT/train.log" 2>&1
}

# Round 1: lambdas 0-1 on GPUs 1,2
run_one ${GPUS[0]} ${LAMBDAS[0]} &
run_one ${GPUS[1]} ${LAMBDAS[1]} &
wait
echo "Round 1 done (lambda=${LAMBDAS[0]}, ${LAMBDAS[1]})"

# Round 2: lambdas 2-3
run_one ${GPUS[0]} ${LAMBDAS[2]} &
run_one ${GPUS[1]} ${LAMBDAS[3]} &
wait
echo "Round 2 done (lambda=${LAMBDAS[2]}, ${LAMBDAS[3]})"

# Round 3: lambda 4
run_one ${GPUS[0]} ${LAMBDAS[4]}
echo "Round 3 done (lambda=${LAMBDAS[4]})"

echo ""
echo "=== Sweep Complete ==="
echo "Results:"
for LAMBDA in "${LAMBDAS[@]}"; do
    NAME="sigreg_lambda${LAMBDA}"
    LOG="$OUTPUT_BASE/$NAME/train.log"
    echo ""
    echo "--- lambda=$LAMBDA ---"
    grep -E "\[SIGReg Stage 1\] Gaussianity" "$LOG" 2>/dev/null | tail -1 || echo "  (no eval yet)"
    grep -E "\[SIGReg Stage 1\] step" "$LOG" 2>/dev/null | tail -1 || echo "  (no steps logged)"
done
