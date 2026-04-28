#!/usr/bin/env bash
# Bootstrap an 8xH100 pod for the PR #1908 base + knob-tweak sprint.
# Installs lrzip (system), Python deps, FA3, fetches CaseOps SP8192 dataset.
# Idempotent — safe to re-run.

set -euo pipefail

echo "=== top1_bootstrap.sh start: $(date -u) ==="

cd /workspace
[ ! -d parameter-golf ] && git clone https://github.com/AayushBaniya2006/parameter-golf.git
cd parameter-golf
git remote remove fork 2>/dev/null || true
git remote add fork https://github.com/AayushBaniya2006/parameter-golf.git
git fetch fork
git checkout -B sprint/top10-crazy fork/sprint/top10-crazy
git pull fork sprint/top10-crazy

echo "=== System deps (lrzip for PR #1855 pergroup compression) ==="
if ! command -v lrzip >/dev/null 2>&1; then
  apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq lrzip
fi
echo "lrzip: $(lrzip -V 2>&1 | head -1)"

echo "=== Python deps ==="
python3 -m pip install --upgrade pip --quiet
python3 -m pip install --quiet brotli sentencepiece huggingface-hub
python3 -m pip install --quiet flash_attn_3 --no-deps \
  --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/

echo "=== Fetch CaseOps dataset (romeerp/parameter-golf-caseops-v1) ==="
mkdir -p data/datasets data/tokenizers
DATA_DIR=./data/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved
TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model
if [ ! -d "$DATA_DIR" ] || [ ! -f "$TOKENIZER_PATH" ]; then
  python3 - <<'PY'
from huggingface_hub import snapshot_download
import os, glob, shutil, sys

# romeerp's repo layout (verified 2026-04-28):
#   datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/*.bin
#   datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model
repo_id = "romeerp/parameter-golf-caseops-v1"
local = snapshot_download(
    repo_id=repo_id, repo_type="dataset",
    local_dir="./data/_caseops_dl",
    allow_patterns=["datasets/**/*.bin", "datasets/**/*.model"],
)
print(f"downloaded to: {local}")

target_dir = "./data/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved"
target_tok = "./data/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model"
os.makedirs(target_dir, exist_ok=True)
os.makedirs("./data/tokenizers", exist_ok=True)

src_data = os.path.join(local, "datasets", "datasets",
                        "fineweb10B_sp8192_lossless_caps_caseops_v1_reserved")
src_tok = os.path.join(local, "datasets", "tokenizers",
                       "fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model")

if os.path.isdir(src_data):
    for src in sorted(glob.glob(os.path.join(src_data, "*.bin"))):
        dst = os.path.join(target_dir, os.path.basename(src))
        if not os.path.exists(dst):
            shutil.move(src, dst)
if os.path.exists(src_tok) and not os.path.exists(target_tok):
    shutil.move(src_tok, target_tok)

n_train = len(glob.glob(target_dir + "/fineweb_train_*.bin"))
n_val = len(glob.glob(target_dir + "/fineweb_val_*.bin"))
print(f"train shards: {n_train}  val shards: {n_val}  tokenizer: {os.path.exists(target_tok)}")
if n_val < 1 or not os.path.exists(target_tok):
    sys.exit("ERROR: dataset incomplete after download — inspect ./data/_caseops_dl/")
PY
fi

echo "=== Verify dataset ==="
N_TRAIN=$(ls data/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_train_*.bin 2>/dev/null | wc -l)
N_VAL=$(ls data/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_val_*.bin 2>/dev/null | wc -l)
echo "train shards: $N_TRAIN  val shards: $N_VAL"
if [ "$N_VAL" -lt 1 ]; then
  echo "ERROR: no val shards found. Inspect ./data/_caseops_dl/ and adjust paths."
  exit 1
fi

echo "=== Verify train_gpt_pr1908.py is in place ==="
test -f train_gpt_pr1908.py && echo "ok: $(wc -l <train_gpt_pr1908.py) lines"

echo "=== Verify GPU ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "=== top1_bootstrap.sh complete: $(date -u) ==="
