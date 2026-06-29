#!/bin/bash
# Round 2+3: lambdas 0.1, 0.5, 1.0
cd ~/koe

run_one() {
    local GPU=$1
    local LAMBDA=$2
    local NAME="sigreg_lambda${LAMBDA}"
    local OUT="/local_data/checkpoints/sigreg_sweep/$NAME"
    mkdir -p "$OUT"
    echo "[GPU $GPU] Launching $NAME"
    PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python -m paper.scripts.train_sigreg_stage1 \
        --data_dir /local_data/data/librilight/librilight_9k \
        --output_dir "$OUT" \
        --run_name "$NAME" \
        --strides 4,4,4,5,3 \
        --stage1_steps 5000 \
        --batch_size 16 \
        --lr 5e-5 \
        --max_seconds 5 \
        --lambda_sigreg $LAMBDA \
        --eval_every 500 \
        --save_every 2500 \
        --no_compile \
        > "$OUT/train.log" 2>&1
}

# Round 2: 0.1 + 0.5 parallel
run_one 1 0.1 &
run_one 2 0.5 &
wait
echo "Round 2 done (0.1, 0.5)"

# Round 3: 1.0
run_one 1 1.0
echo "Round 3 done (1.0)"

echo "=== ALL DONE ==="
