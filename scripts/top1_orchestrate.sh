#!/usr/bin/env bash
# End-to-end orchestrator: repro → sweep → 3-seed final → STOP POD.
# Designed to run unattended on the pod. Stops the pod (via runpodctl
# from within the container) at the end regardless of success/failure.
#
# Pre-conditions: bootstrap already done (data + tokenizer + scripts).

set -uo pipefail

POD_ID="${POD_ID:-8xbstrspahdpgl}"
ORCH_LOG=/workspace/_orchestrate.log
exec >>"$ORCH_LOG" 2>&1

cd /workspace/parameter-golf

echo "============================================================"
echo "=== top1_orchestrate.sh START: $(date -u) ==="
echo "============================================================"

stop_pod_and_exit() {
  local rc=${1:-0}
  echo ""
  echo "=== Pipeline finished (rc=$rc) at $(date -u) ==="
  echo "=== Final balance / pod stop ==="
  if command -v runpodctl >/dev/null 2>&1; then
    runpodctl pod stop "$POD_ID" 2>&1 || echo "WARN: pod stop failed"
  else
    echo "WARN: runpodctl not on pod; pod will keep running until manually stopped"
  fi
  exit $rc
}

trap 'stop_pod_and_exit $?' EXIT

# Pre-flight: kill anything stale, clear leftover bg state
echo "=== Pre-flight cleanup ==="
pkill -9 -f 'torchrun' 2>/dev/null || true
pkill -9 -f 'train_gpt_pr1908' 2>/dev/null || true
sleep 3
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

# Stage 1: repro PR #1908 seed 42 — produces artifacts/repro_seed42/final_model.pt
echo ""
echo "============================================================"
echo "=== STAGE 1: repro seed 42 (target ~1.05957 ±0.0009) ==="
echo "============================================================"
rm -rf artifacts/repro_seed42 logs/repro_seed42.log
bash scripts/top1_repro_pr1908.sh
RC=$?
if [ $RC -ne 0 ] || ! grep -q quantized_ttt_phased logs/repro_seed42.log 2>/dev/null; then
  echo "STAGE 1 FAILED (rc=$RC) — aborting"
  exit 1
fi
REPRO_BPB=$(grep -oE 'quantized_ttt_phased[^v]*val_bpb:[0-9]+\.[0-9]+' logs/repro_seed42.log | grep -oE '[0-9]+\.[0-9]+' | tail -1)
echo "STAGE 1 OK: repro val_bpb=$REPRO_BPB"

# Stage 2: knob sweep using seed-42 base via QUANTIZE_ONLY=1
echo ""
echo "============================================================"
echo "=== STAGE 2: quant-knob sweep ==="
echo "============================================================"
bash scripts/top1_quant_sweep.sh

# Pick winner from sweep — lowest BPB with bytes < 16,000,000
echo ""
echo "=== Picking winner ==="
WINNER_LINE=$(python3 - <<'PY'
import re, glob, os
best = None
for f in glob.glob("logs/sweep/*.log"):
    tag = os.path.basename(f).replace(".log","")
    txt = open(f).read()
    m = re.search(r'quantized_ttt_phased[^v]*val_bpb:([0-9]+\.[0-9]+)', txt)
    bb = re.findall(r'Serialized model quantized\S*:\s*([0-9]+) bytes', txt)
    if not m or not bb: continue
    bpb = float(m.group(1))
    bytes_ = int(bb[-1])
    if bytes_ >= 16_000_000:
        continue
    if best is None or bpb < best[1]:
        best = (tag, bpb, bytes_)
if best is None:
    print("FAIL")
else:
    print(f"{best[0]}\t{best[1]}\t{best[2]}")
PY
)
echo "winner: $WINNER_LINE"
if echo "$WINNER_LINE" | grep -q '^FAIL'; then
  echo "STAGE 2 FAILED — no winner"
  exit 2
fi

WINNER_TAG=$(echo "$WINNER_LINE" | cut -f1)
WINNER_BPB=$(echo "$WINNER_LINE" | cut -f2)

# Decode winner tag to env vars (matches scripts/top1_quant_sweep.sh definitions)
case "$WINNER_TAG" in
  baseline)        WINNER_KNOBS="AWQ_LITE_GROUP_TOP_K=1 LQER_RANK=4 LQER_TOP_K=3 LQER_GAIN_SELECT=0" ;;
  s01_awqk2)       WINNER_KNOBS="AWQ_LITE_GROUP_TOP_K=2 LQER_RANK=4 LQER_TOP_K=3 LQER_GAIN_SELECT=0" ;;
  s02_awqk2_lqk4)  WINNER_KNOBS="AWQ_LITE_GROUP_TOP_K=2 LQER_RANK=4 LQER_TOP_K=4 LQER_GAIN_SELECT=0" ;;
  s03_lqgs)        WINNER_KNOBS="AWQ_LITE_GROUP_TOP_K=1 LQER_RANK=4 LQER_TOP_K=3 LQER_GAIN_SELECT=1" ;;
  s04_combo)       WINNER_KNOBS="AWQ_LITE_GROUP_TOP_K=2 LQER_RANK=4 LQER_TOP_K=4 LQER_GAIN_SELECT=1" ;;
  s05_aggressive)  WINNER_KNOBS="AWQ_LITE_GROUP_TOP_K=3 LQER_RANK=4 LQER_TOP_K=4 LQER_GAIN_SELECT=1" ;;
  s06_lqr6)        WINNER_KNOBS="AWQ_LITE_GROUP_TOP_K=2 LQER_RANK=6 LQER_TOP_K=3 LQER_GAIN_SELECT=1" ;;
  s07_lqk5)        WINNER_KNOBS="AWQ_LITE_GROUP_TOP_K=2 LQER_RANK=4 LQER_TOP_K=5 LQER_GAIN_SELECT=1" ;;
  s08_max)         WINNER_KNOBS="AWQ_LITE_GROUP_TOP_K=4 LQER_RANK=4 LQER_TOP_K=4 LQER_GAIN_SELECT=1" ;;
  *) echo "Unknown winner tag: $WINNER_TAG"; exit 3 ;;
esac
echo "WINNER_KNOBS: $WINNER_KNOBS"

# Always run 3-seed final — PR #1908 author admits 600s overshoot, so even a
# compliant 3-seed at PR #1908 quality could take #1. Worth the $12.
BASELINE_BPB=$(grep -oE 'quantized_ttt_phased[^v]*val_bpb:[0-9]+\.[0-9]+' logs/sweep/baseline.log 2>/dev/null | grep -oE '[0-9]+\.[0-9]+' | tail -1 || echo "?")
echo "Baseline (PR #1908 default) seed-42 BPB: $BASELINE_BPB"
echo "Winner ($WINNER_TAG) seed-42 BPB:        $WINNER_BPB"

# Stage 3: 3-seed final with organic 600s wallclock
echo ""
echo "============================================================"
echo "=== STAGE 3: 3-seed final with $WINNER_KNOBS ==="
echo "============================================================"
bash scripts/top1_final_3seed.sh $WINNER_KNOBS

echo ""
echo "============================================================"
echo "=== ALL STAGES DONE: $(date -u) ==="
echo "============================================================"
cat artifacts/final/3seed_results.json 2>/dev/null || echo "(no 3seed_results.json)"
