#!/usr/bin/env bash
# Reproduce PR #1908 seed-42 result (~1.05957 post-TTT) to validate setup.
# This MUST land within seed-42 single-seed noise (~0.0009 BPB) of 1.05957
# before we trust the sweep. Runs on 8xH100, takes ~10 min, costs ~$4.
# Output: artifacts/repro_seed42/

set -euo pipefail

cd /workspace/parameter-golf
mkdir -p artifacts/repro_seed42 logs

echo "=== top1_repro_pr1908.sh: matched-step seed 42 (FORCE_STOP_STEP=4945) ==="
echo "=== Expected post-TTT BPB: ~1.05957 ±0.0009 ==="
echo "=== Started: $(date -u) ==="

DATA_DIR=./data \
DATA_PATH=./data/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved \
TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
ARTIFACT_DIR=artifacts/repro_seed42 \
RUN_ID=repro_seed42 \
SEED=42 \
VOCAB_SIZE=8192 \
CASEOPS_ENABLED=1 \
ITERATIONS=20000 MAX_WALLCLOCK_SECONDS=0 FORCE_STOP_STEP=4945 \
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
LQER_ENABLED=1 LQER_ASYM_ENABLED=1 LQER_RANK=4 LQER_FACTOR_BITS=4 \
LQER_ASYM_GROUP=64 LQER_TOP_K=3 \
AWQ_LITE_ENABLED=1 AWQ_LITE_BITS=8 AWQ_LITE_GROUP_TOP_K=1 AWQ_LITE_GROUP_SIZE=64 \
FUSED_CE_ENABLED=1 COMPRESSOR=pergroup NCCL_NET=Socket \
torchrun --standalone --nproc_per_node=8 train_gpt_pr1908.py 2>&1 | tee logs/repro_seed42.log

echo "=== Finished: $(date -u) ==="
echo "=== Final post-TTT BPB ==="
grep -E "post-ttt|Post-TTT|val_bpb" logs/repro_seed42.log | tail -10
