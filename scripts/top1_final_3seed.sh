#!/usr/bin/env bash
# Train seeds 0 and 1234 with the WINNING knob config from the sweep,
# matched-step against PR #1855's stop steps (4932 and 4917).
# Computes 3-seed mean and writes submission.json.
#
# Pass the winning knobs as args:
#   ./top1_final_3seed.sh AWQ_LITE_GROUP_TOP_K=2 LQER_TOP_K=4 LQER_GAIN_SELECT=1
#
# Each seed: ~10 min on 8xH100 = $4. Total ~$8 + the seed-42 base already done.

set -euo pipefail

cd /workspace/parameter-golf

if [ $# -eq 0 ]; then
  echo "usage: $0 KNOB1=val [KNOB2=val ...]"
  echo "example: $0 AWQ_LITE_GROUP_TOP_K=2 LQER_TOP_K=4 LQER_GAIN_SELECT=1"
  exit 1
fi

mkdir -p artifacts/final logs

run_seed() {
  local seed="$1"
  shift
  local out=artifacts/final/seed${seed}
  local log=logs/final_seed${seed}.log
  if [ -f "$log" ] && grep -q 'quantized_ttt_phased' "$log"; then
    echo "skip seed=$seed (complete)"
    return 0
  fi
  [ -f "$log" ] && mv "$log" "${log}.partial.$(date +%s)"
  mkdir -p "$out"

  # NOTE: organic 600s wallclock cap (NO FORCE_STOP_STEP) — compliance
  # safety. PR #1908 exceeded 600s with FORCE_STOP_STEP=4945 (used 601153ms).
  echo "=== seed=$seed starting: $(date -u) ==="
  DATA_DIR=./data \
  DATA_PATH=./data/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved \
  TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
  ARTIFACT_DIR=$out \
  RUN_ID=final_seed${seed} \
  SEED=${seed} VOCAB_SIZE=8192 CASEOPS_ENABLED=1 \
  ITERATIONS=20000 MAX_WALLCLOCK_SECONDS=600 \
  PHASED_TTT_ENABLED=1 PHASED_TTT_PREFIX_DOCS=2500 PHASED_TTT_NUM_PHASES=3 \
  EMBED_BITS=7 MATRIX_LR=0.026 MIN_LR=0.1 \
  MLP_CLIP_SIGMAS=11.5 ATTN_CLIP_SIGMAS=13.0 EMBED_CLIP_SIGMAS=14.0 \
  GRAD_CLIP_NORM=0.3 TTT_CHUNK_SIZE=48 WARMUP_STEPS=20 MUON_BACKEND_STEPS=5 \
  GLOBAL_TTT_MOMENTUM=0.9 WARMDOWN_FRAC=0.85 BETA2=0.99 \
  TTT_BETA2=0.99 TTT_WEIGHT_DECAY=0.5 TTT_LORA_RANK=80 \
  SPARSE_ATTN_GATE_SCALE=0.5 \
  GPTQ_RESERVE_SECONDS=0.5 GPTQ_CALIBRATION_BATCHES=16 VAL_LOSS_EVERY=0 \
  GATED_ATTN_QUANT_GATE=1 SPARSE_ATTN_GATE_ENABLED=1 GATE_WINDOW=12 \
  SMEAR_GATE_ENABLED=1 \
  LQER_ENABLED=1 LQER_ASYM_ENABLED=1 LQER_RANK=4 LQER_FACTOR_BITS=4 LQER_ASYM_GROUP=64 \
  AWQ_LITE_ENABLED=1 AWQ_LITE_BITS=8 AWQ_LITE_GROUP_SIZE=64 \
  FUSED_CE_ENABLED=1 COMPRESSOR=pergroup NCCL_NET=Socket \
  "$@" \
  torchrun --standalone --nproc_per_node=8 train_gpt_pr1908.py 2>&1 | tee "$log"
}

# Seed-42 should already exist as artifacts/repro_seed42/ — but if user changed
# knobs, retrain it for matched config.
run_seed 42 "$@"
run_seed 0 "$@"
run_seed 1234 "$@"

echo ""
echo "=== Per-seed post-TTT BPB ==="
python3 - <<'PY'
import re, glob, json, os
results = {}
for f in sorted(glob.glob("logs/final_seed*.log")):
    seed = re.search(r'seed(\d+)', f).group(1)
    txt = open(f).read()
    m = re.search(r'quantized_ttt_phased[^v]*val_bpb:([0-9]+\.[0-9]+)', txt)
    bpb = float(m.group(1)) if m else None
    bb = re.findall(r'Serialized model quantized\S*:\s*([0-9]+) bytes', txt)
    bytes_ = int(bb[-1]) if bb else None
    results[seed] = {"val_bpb": bpb, "artifact_bytes": bytes_, "log": f}
    print(f"seed {seed}: val_bpb={bpb}  artifact_bytes={bytes_}")

vs = [r["val_bpb"] for r in results.values() if r["val_bpb"] is not None]
if len(vs) == 3:
    import statistics
    mean = sum(vs)/3
    std = statistics.stdev(vs)
    print(f"\n3-seed mean: {mean:.5f}  std: {std:.5f}")
    print(f"vs PR #1908 (1.06081): delta = {mean - 1.06081:+.5f}")
    print(f"vs PR #1855 (1.06108): delta = {mean - 1.06108:+.5f}")
    with open("artifacts/final/3seed_results.json", "w") as f:
        json.dump({"seed_results": results, "mean": mean, "std": std,
                   "vs_pr1908": mean - 1.06081, "vs_pr1855": mean - 1.06108}, f, indent=2)
    print("\nWrote artifacts/final/3seed_results.json")
PY
