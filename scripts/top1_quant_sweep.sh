#!/usr/bin/env bash
# Quant-config sweep using QUANTIZE_ONLY=1 against the seed-42 base saved by
# top1_repro_pr1908.sh (artifacts/repro_seed42/final_model.pt).
#
# Strategy: PR #1908 left these knobs conservatively set. Try aggressive variants:
#   PR1908 baseline:  AWQ_TOP_K=1 LQER_TOP_K=3 LQER_GAIN_SELECT=0
#   sweep01: AWQ_TOP_K=2 LQER_TOP_K=3 LQER_GAIN_SELECT=0
#   sweep02: AWQ_TOP_K=2 LQER_TOP_K=4 LQER_GAIN_SELECT=0
#   sweep03: AWQ_TOP_K=1 LQER_TOP_K=3 LQER_GAIN_SELECT=1
#   sweep04: AWQ_TOP_K=2 LQER_TOP_K=4 LQER_GAIN_SELECT=1
#   sweep05: AWQ_TOP_K=3 LQER_TOP_K=4 LQER_GAIN_SELECT=1
#   sweep06: AWQ_TOP_K=2 LQER_RANK=6 LQER_TOP_K=3 LQER_GAIN_SELECT=1
#   sweep07: AWQ_TOP_K=2 LQER_TOP_K=5 LQER_GAIN_SELECT=1
#   sweep08: AWQ_TOP_K=4 LQER_TOP_K=4 LQER_GAIN_SELECT=1
#
# Each config takes ~2 min on 8xH100 (Hessian collect + GPTQ + serialize + TTT eval).
# Total sweep ~20 min, ~$8.
# Each config produces final_model.int6.ptz; we record post-TTT val_bpb and artifact size.

set -euo pipefail

cd /workspace/parameter-golf
mkdir -p artifacts/sweep logs/sweep

BASE=artifacts/repro_seed42/final_model.pt
test -f "$BASE" || { echo "ERROR: missing $BASE — run top1_repro_pr1908.sh first"; exit 1; }
echo "Base checkpoint: $(stat -c%s "$BASE" 2>/dev/null || stat -f%z "$BASE") bytes"

run_config() {
  local tag="$1"; shift
  local out=artifacts/sweep/${tag}
  local log=logs/sweep/${tag}.log
  if [ -f "$log" ] && grep -q 'quantized_ttt_phased' "$log"; then
    echo "skip ${tag} (complete)"
    return 0
  fi
  [ -f "$log" ] && mv "$log" "${log}.partial.$(date +%s)"
  mkdir -p "$out"
  cp "$BASE" "$out/final_model.pt"

  echo "=== ${tag} starting: $(date -u) ==="
  DATA_DIR=./data \
  DATA_PATH=./data/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved \
  TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
  ARTIFACT_DIR=$out \
  RUN_ID=$tag \
  SEED=42 VOCAB_SIZE=8192 CASEOPS_ENABLED=1 \
  ITERATIONS=20000 MAX_WALLCLOCK_SECONDS=0 FORCE_STOP_STEP=4945 \
  QUANTIZE_ONLY=1 \
  PHASED_TTT_ENABLED=1 PHASED_TTT_PREFIX_DOCS=2500 PHASED_TTT_NUM_PHASES=3 \
  EMBED_BITS=7 \
  MLP_CLIP_SIGMAS=11.5 ATTN_CLIP_SIGMAS=13.0 EMBED_CLIP_SIGMAS=14.0 \
  TTT_CHUNK_SIZE=48 GLOBAL_TTT_MOMENTUM=0.9 \
  TTT_BETA2=0.99 TTT_WEIGHT_DECAY=0.5 TTT_LORA_RANK=80 \
  SPARSE_ATTN_GATE_SCALE=0.5 \
  GPTQ_RESERVE_SECONDS=0.5 GPTQ_CALIBRATION_BATCHES=16 VAL_LOSS_EVERY=0 \
  GATED_ATTN_QUANT_GATE=1 SPARSE_ATTN_GATE_ENABLED=1 GATE_WINDOW=12 \
  SMEAR_GATE_ENABLED=1 \
  LQER_ENABLED=1 LQER_ASYM_ENABLED=1 LQER_FACTOR_BITS=4 LQER_ASYM_GROUP=64 \
  AWQ_LITE_ENABLED=1 AWQ_LITE_BITS=8 AWQ_LITE_GROUP_SIZE=64 \
  FUSED_CE_ENABLED=1 COMPRESSOR=pergroup NCCL_NET=Socket \
  "$@" \
  torchrun --standalone --nproc_per_node=8 train_gpt_pr1908.py 2>&1 | tee "$log"
}

# baseline (sanity)
run_config baseline AWQ_LITE_GROUP_TOP_K=1 LQER_RANK=4 LQER_TOP_K=3 LQER_GAIN_SELECT=0

# Target winners — additive aggressive knobs
run_config s01_awqk2          AWQ_LITE_GROUP_TOP_K=2 LQER_RANK=4 LQER_TOP_K=3 LQER_GAIN_SELECT=0
run_config s02_awqk2_lqk4     AWQ_LITE_GROUP_TOP_K=2 LQER_RANK=4 LQER_TOP_K=4 LQER_GAIN_SELECT=0
run_config s03_lqgs           AWQ_LITE_GROUP_TOP_K=1 LQER_RANK=4 LQER_TOP_K=3 LQER_GAIN_SELECT=1
run_config s04_combo          AWQ_LITE_GROUP_TOP_K=2 LQER_RANK=4 LQER_TOP_K=4 LQER_GAIN_SELECT=1
run_config s05_aggressive     AWQ_LITE_GROUP_TOP_K=3 LQER_RANK=4 LQER_TOP_K=4 LQER_GAIN_SELECT=1
run_config s06_lqr6           AWQ_LITE_GROUP_TOP_K=2 LQER_RANK=6 LQER_TOP_K=3 LQER_GAIN_SELECT=1
run_config s07_lqk5           AWQ_LITE_GROUP_TOP_K=2 LQER_RANK=4 LQER_TOP_K=5 LQER_GAIN_SELECT=1
run_config s08_max            AWQ_LITE_GROUP_TOP_K=4 LQER_RANK=4 LQER_TOP_K=4 LQER_GAIN_SELECT=1

echo ""
echo "=== Ranked results (lower BPB = better; bytes must be < 16,000,000) ==="
for f in logs/sweep/*.log; do
  tag=$(basename "$f" .log)
  bpb=$(grep -oE 'quantized_ttt_phased[^v]*val_bpb:[0-9]+\.[0-9]+' "$f" | grep -oE '[0-9]+\.[0-9]+' | tail -1)
  bytes=$(grep -oE 'Serialized model quantized\S*:\s*[0-9]+ bytes' "$f" | grep -oE '[0-9]+' | tail -1)
  preq=$(grep -oE 'pre-quantization post-ema[^v]*val_bpb:[0-9]+\.[0-9]+' "$f" | grep -oE '[0-9]+\.[0-9]+' | tail -1)
  q=$(grep -oE 'diagnostic quantized[^v]*val_bpb:[0-9]+\.[0-9]+' "$f" | grep -oE '[0-9]+\.[0-9]+' | tail -1)
  [ -z "$bpb" ] && bpb="FAIL"
  [ -z "$bytes" ] && bytes="?"
  [ -z "$preq" ] && preq="-"
  [ -z "$q" ] && q="-"
  printf "%-10s  preq=%s  q=%s  ttt=%s  bytes=%s  %s\n" "" "$preq" "$q" "$bpb" "$bytes" "$tag"
done | sort -k4 -n
echo ""
echo "Pick the lowest 'ttt' (post-TTT BPB) with bytes < 16,000,000."
echo "Then run: ./scripts/top1_final_3seed.sh AWQ_LITE_GROUP_TOP_K=<X> LQER_TOP_K=<Y> LQER_GAIN_SELECT=<Z>"
