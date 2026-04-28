#!/usr/bin/env bash
# Pack final 3-seed results into a submission record directory under records/.
# Run after top1_final_3seed.sh succeeds with mean < 1.06081.
#
# Usage: ./top1_pack_submission.sh <record_subdir_name> "<title>"
# Example: ./top1_pack_submission.sh 2026-04-29_PR1908Base_AggressiveAWQLQER \
#                                    "PR1908 base + AWQ_TOP_K=2 + LQER_GAIN_SELECT + LQER_TOP_K=4"

set -euo pipefail

if [ $# -lt 2 ]; then
  echo "usage: $0 <subdir_name> <title>"
  exit 1
fi

cd /workspace/parameter-golf
SUBDIR="$1"
TITLE="$2"
DEST="records/track_10min_16mb/${SUBDIR}"
mkdir -p "$DEST"

# Copy logs
cp logs/final_seed42.log "$DEST/train_seed42.log"
cp logs/final_seed0.log "$DEST/train_seed0.log"
cp logs/final_seed1234.log "$DEST/train_seed1234.log"

# Copy code (this IS PR #1908's train_gpt.py — we credit them)
cp train_gpt_pr1908.py "$DEST/train_gpt.py"

# requirements.txt — copy what's known to work
cat > "$DEST/requirements.txt" <<'EOF'
torch==2.9.1+cu128
numpy
brotli
sentencepiece
huggingface-hub
EOF

# Compute mean from results.json
python3 - <<PY > "$DEST/submission.json"
import json, statistics
with open("artifacts/final/3seed_results.json") as f:
    r = json.load(f)
seeds = sorted(int(k) for k in r["seed_results"].keys())
seed_results = {}
for s in seeds:
    sr = r["seed_results"][str(s)]
    seed_results[str(s)] = {
        "val_bpb": sr["val_bpb"],
        "artifact_bytes": sr["artifact_bytes"],
    }
out = {
    "author": "Aayush C Baniya",
    "github_id": "AayushBaniya2006",
    "val_bpb": r["mean"],
    "val_bpb_std": r["std"],
    "seeds": seeds,
    "seed_results": seed_results,
    "compliance": {
        "artifact_bytes_max": max(s["artifact_bytes"] for s in seed_results.values()),
        "train_seconds_cap": 600,
        "eval_seconds_cap": 600,
    },
    "lineage": {
        "base": "PR #1908 (codex/awq-stepmatched, romeerp/parameter-golf)",
        "delta": "AWQ_LITE_GROUP_TOP_K + LQER_TOP_K + LQER_GAIN_SELECT knob increases",
    },
}
print(json.dumps(out, indent=2))
PY

# Stub README — user MUST edit before submitting
cat > "$DEST/README.md" <<EOF
# ${TITLE}

3-seed mean val_bpb: see \`submission.json\`. Stages on PR #1908 (romeerp's
activation-aware GPTQ) by enabling additional knobs that PR #1908 left at
defaults.

## What changed vs PR #1908

- \`AWQ_LITE_GROUP_TOP_K\`: protect more salient column groups at int8
- \`LQER_TOP_K\`: more LQER-corrected tensors
- \`LQER_GAIN_SELECT=1\`: select LQER tensors by actual gain (vs error-norm)

(Edit this README with the actual winning knob values before submitting.)

## Lineage

- PR #1908 (romeerp) — activation-aware GPTQ mixed precision base
- PR #1855 (codemath3000) — full architectural stack
- PR #1797 (dexhunter) — Smear Gate + LQER asymmetric
- PR #1787 (nprime06) — Polar Express NS, sparse attn gate, MIN_LR floor

## Reproducing

3-seed mean uses organic 600s wallclock cap (no FORCE_STOP_STEP) for
compliance safety.

\`\`\`bash
# After ./top1_bootstrap.sh on an 8xH100 pod:
./scripts/top1_final_3seed.sh \\
  AWQ_LITE_GROUP_TOP_K=<X> \\
  LQER_TOP_K=<Y> \\
  LQER_GAIN_SELECT=<Z>
\`\`\`
EOF

echo ""
echo "=== Packed: $DEST ==="
ls -la "$DEST"
echo ""
echo "=== Validate ==="
python3 scripts/validate_submission.py "$DEST" || \
  echo "Validation failed — fix issues before submitting"
echo ""
echo "Next steps:"
echo "  1. Edit $DEST/README.md with actual winning knobs"
echo "  2. git add $DEST && git commit && git push"
echo "  3. gh pr create against openai/parameter-golf:main"
