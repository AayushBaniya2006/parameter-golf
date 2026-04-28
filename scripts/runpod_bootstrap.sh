#!/usr/bin/env bash
# One-shot RunPod bootstrap for the top10-crazy worktree.
# Usage: ssh <pod> bash -s < scripts/runpod_bootstrap.sh
set -euo pipefail

cd /workspace
if [ ! -d parameter-golf ]; then
  git clone https://github.com/AayushBaniya2006/parameter-golf.git
fi
cd parameter-golf
git fetch origin
git checkout sprint/top10-crazy || git checkout -b sprint/top10-crazy origin/sprint/top10-crazy
git pull

python3 -m pip install --upgrade pip >/dev/null
python3 -m pip install brotli sentencepiece
python3 -m pip install flash_attn_3 --no-deps \
  --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/

if [ ! -d data/datasets/fineweb10B_sp8192 ]; then
  MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
    python3 data/cached_challenge_fineweb.py --variant sp8192
fi

echo "=== bootstrap complete ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "PyTorch: $(python3 -c 'import torch;print(torch.__version__)')"
echo "Val tokens: $(python3 -c "
import glob,numpy as np
files=sorted(glob.glob('data/datasets/fineweb10B_sp8192/fineweb_val_*.bin'))
tot=0
for f in files:
  hdr=np.fromfile(f,dtype='<i4',count=256);tot+=int(hdr[2])
print(tot)
")"
