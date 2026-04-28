#!/usr/bin/env bash
# Pre-quant TTT sweep on 1xH100. Each config: 2000 training steps + full pre-quant-TTT phase + no eval TTT.
# Ranks configs by post-ptt val_bpb.
set -euo pipefail

cd /workspace/parameter-golf
test -f /workspace/pg_modified.py || { echo "ERROR: /workspace/pg_modified.py not found; run pack_record.py + scp first"; exit 1; }

LRS=(1e-4 3e-4 1e-3)
EPOCHS_LIST=(12 18 24)
FREEZES=(0 1)

mkdir -p logs/sweep

for lr in "${LRS[@]}"; do
  for ep in "${EPOCHS_LIST[@]}"; do
    for fz in "${FREEZES[@]}"; do
      run_id="sweep_lr${lr}_ep${ep}_fz${fz}"
      log="logs/sweep/${run_id}.log"
      # Treat as done only if the success marker is present; partial logs get re-run.
      if [ -f "$log" ] && grep -q "timed_eval: pre-quantization post-ptt" "$log"; then
        echo "skip ${run_id} (complete)"
        continue
      fi
      [ -f "$log" ] && { echo "rerun ${run_id} (incomplete log)"; mv "$log" "${log}.partial.$(date +%s)"; }
      echo "=== starting ${run_id} ==="
      set +e
      SEED=42 \
      VOCAB_SIZE=8192 \
      QK_GAIN_INIT=5.25 \
      TTT_ENABLED=0 \
      PRE_QUANT_TTT_ENABLED=1 \
      PRE_QUANT_TTT_EPOCHS=${ep} \
      PRE_QUANT_TTT_LR=${lr} \
      PRE_QUANT_TTT_CHUNK_TOKENS=65536 \
      PRE_QUANT_TTT_FREEZE_BLOCKS=${fz} \
      ITERATIONS=2000 \
      MAX_WALLCLOCK_SECONDS=900 \
      VAL_LOSS_EVERY=1000 \
      RUN_ID=${run_id} \
      torchrun --standalone --nproc_per_node=1 /workspace/pg_modified.py \
        2>&1 | tee logs/sweep/${run_id}.log
      rc=${PIPESTATUS[0]}
      set -e
      if [ $rc -ne 0 ]; then
        echo "WARN: ${run_id} exited rc=$rc; continuing to next config"
      fi
    done
  done
done

echo "=== sweep complete; ranked results: ==="
for f in logs/sweep/*.log; do
  name=$(basename "$f" .log)
  bpb=$(grep "timed_eval: pre-quantization post-ptt" "$f" | tail -1 | grep -oE 'val_bpb: [0-9.]+' | awk '{print $2}')
  [ -z "$bpb" ] && bpb="FAIL"
  echo "${bpb} ${name}"
done | sort -n
