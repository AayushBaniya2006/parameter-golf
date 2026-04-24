# Parameter Golf Top-10 Final Push — Pre-Quant TTT + Eval-Time TTT on PR #1493 Trunk

**Date:** 2026-04-24
**Deadline:** 2026-04-30 (6 days)
**Author:** Aayush Baniya
**Branch:** `sprint/top10-crazy` (worktree at `.worktrees/top10-crazy/`)

## Problem statement

Our March 26 submission (PR #809, 0.29519 BPB) was closed as illegal: the hashed N-gram cache's non-renormalized mixing leaked target-adjacent token information. A month later the merged leaderboard has moved to #1184 Scylla at 0.9485 BPB, and the open PR frontier is at 1.00-1.08 BPB. We have 6 days and ~$1,004 RunPod credits to land a legal, merge-worthy record.

Our legitimate pre-N-gram baseline on the old SP1024 `frontier_lean` stack is 1.1460 BPB post-EMA (1.1600 BPB roundtrip). That stack is now obsolete — the modern trunk is SP8192 + depth recurrence + parallel residuals + legal score-first TTT (PR #1493, merged SOTA-minus-Scylla at 1.0810 BPB).

## Goal & target

- **Ship one legal record submission to `openai/parameter-golf` main leaderboard by 2026-04-30.**
- **Primary target:** val_bpb ≤ 1.066 (3-seed mean) → predicted landing slot **rank 3–7**.
- **Hard floor (fallback):** faithful reproduction of PR #1493 → 1.0810 BPB → rank 8–12 as open PRs get merged.
- **Delta to our legitimate baseline (1.1460):** −0.080 to −0.095 BPB.
- **Delta to current merged SOTA (0.9485 Scylla):** +0.12 BPB (not targeting #1; Scylla is a tokenizer-search win that needs 2-3 weeks of work to attack).

## Approach

Fork `records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT/train_gpt.py` verbatim. Apply one novel composition: add PR #1517-style **pre-quant AdamW TTT phase** between main training and GPTQ, while keeping #1493's **post-quant SGD eval-time TTT**. Both mechanisms are independently legal and merged elsewhere; their composition on #1493's exact trunk is unexplored.

## Section 1 — Baseline (inherited verbatim from PR #1493)

| Component | Setting |
|---|---|
| Tokenizer | SP8192 (`kevclark/parameter-golf` variant) |
| Architecture | 11 physical layers × 512d × 8H/4KV, MLP 4×, LeakyReLU(0.5)², Partial RoPE (16/64), tied embeddings, logit softcap 30 |
| Depth recurrence | Encoder `[0,1,2,3,4,5,3,4]`, decoder `[5,3,4,5,6,7,8,9,10]` → 17 virtual layers, activates at frac=0.35 |
| Parallel residuals | From layer 7, GPT-J style (attn and MLP share pre-residual input) |
| Pretraining optimizer | MuonEq-R (row-norm Muon + Newton-Schulz 5 steps) for 2D matrices, AdamW for embeddings/scalars |
| QK-Gain | 5.25, learnable per-head |
| Hyperparameters | WD=0.095, MLR=0.022, EMA=0.9965, warmdown=0.72 |
| Quantization | GPTQ SDClip int6 (k=12.85) for matrices, int8 (k=20.0) for embeddings, Brotli-11 + byte-shuffle |
| Post-quant eval TTT | SGD(lr=0.005, momentum=0.9), 3 epochs per 32K chunk, cosine LR decay, grad clip 1.0, score-first |
| Training envelope | 4,550 steps in 588s on 8×H100 SXM |
| Eval envelope | Sliding + TTT ≈ 500s |

**Nothing in this list changes.** We only insert a phase after EMA application and before GPTQ.

## Section 2 — Novel lever: pre-quant AdamW TTT phase

### 2.1 Mechanism

Between the end of main training (EMA-applied model) and GPTQ calibration, insert a short fine-tuning phase with a TTT-shaped protocol on **held-out training tokens**:

1. Reserve the last ~64K tokens of the training stream as a `pre_quant_ttt_chunk` (never seen during main training — enforced by loader offset).
2. Score the chunk under `torch.no_grad()` (sanity check only; not used).
3. Run 18 epochs of AdamW over the chunk:
   - lr = 3e-4
   - betas = (0.9, 0.95)
   - weight_decay = 0.0
   - cosine LR decay across epochs to 0
   - grad clip = 1.0
   - freeze first 1 block (mirror of PR #1517; keeps low-level features stable)
4. Hand the updated weights to GPTQ calibration as if they came straight from main training.

### 2.2 Code change (train_gpt.py)

Insert between `apply_ema()` and `run_gptq_calibration()`:

```python
if PRE_QUANT_TTT_ENABLED:
    pq_tokens = load_reserved_training_chunk(n=PRE_QUANT_TTT_CHUNK_TOKENS)  # 64K held out from train stream
    pq_opt = torch.optim.AdamW(
        [p for n,p in model.named_parameters() if not n.startswith("blocks.0.")],
        lr=PRE_QUANT_TTT_LR, betas=(0.9, 0.95), weight_decay=0.0,
    )
    pq_sched = torch.optim.lr_scheduler.CosineAnnealingLR(pq_opt, T_max=PRE_QUANT_TTT_EPOCHS)
    for epoch in range(PRE_QUANT_TTT_EPOCHS):
        for mb in batched(pq_tokens, bsz=PRE_QUANT_TTT_BSZ):
            loss = model(mb).loss
            pq_opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            pq_opt.step()
        pq_sched.step()
    dist.barrier()
```

### 2.3 Config (env vars)

| Env | Default |
|---|---|
| `PRE_QUANT_TTT_ENABLED` | 1 |
| `PRE_QUANT_TTT_EPOCHS` | 18 |
| `PRE_QUANT_TTT_LR` | 3e-4 |
| `PRE_QUANT_TTT_CHUNK_TOKENS` | 65536 |
| `PRE_QUANT_TTT_BSZ` | 32768 |
| `PRE_QUANT_TTT_FREEZE_BLOCKS` | 1 |

### 2.4 Expected stacking

| Phase | Expected BPB (model seed 42) |
|---|---|
| Main training + EMA (FP, pre-quant, no TTT) | ~1.10 |
| + pre-quant TTT (FP, still pre-quant) | ~1.04 |
| + GPTQ SDClip quantize (quantized artifact) | ~1.07 |
| + eval-time SGD TTT (final reported BPB) | **~1.065** |

If the stacking holds, 3-seed mean lands **1.060-1.066** → rank 3-7. If pre-quant TTT doesn't stack cleanly with eval-time TTT, we expect ~1.075 (rank 8-12, still on leaderboard).

## Section 3 — Legality verification

Per competition README and Issue #1017 (Track B legal eval-time adaptation):

| Rule | Status | Evidence |
|---|---|---|
| Artifact < 16,000,000 bytes | ✓ | Same quantization as #1493 (15.99 MB). Pre-quant TTT only updates existing weights, doesn't add parameters. |
| Training < 600 s on 8×H100 SXM | ✓ (tight) | Main training: 550-560s (reduced from 588s to make room). Pre-quant TTT: ~15-25s (18 epochs on 64K tokens ≈ 1.2M token passes total at H100 throughput). GPTQ: ~3s. **Total: 570-590s, needs profiling on Day 1.** |
| Evaluation < 600 s on 8×H100 SXM | ✓ | Unchanged from #1493 (~500s). |
| No validation data access during training | ✓ | Pre-quant TTT uses held-out **training** tokens only. Enforced by loader reserving last 64K tokens from training stream. |
| TTT must be strictly score-first at eval | ✓ | Post-quant eval TTT is #1493's loop verbatim (score chunk under no_grad, then update). |
| N-gram eval cache | N/A | We do not add one. |
| SLOT / per-sample eval adaptation | N/A | Not used. |
| ETLB / eval-time logit bias | N/A | Not used. |
| Paid prefix / val tokens baked into artifact | N/A | Not used. |
| p < 0.01 significance, 3 seeds | ✓ | Plan: seeds 42, 314, 999 (match #1493's seeds for apples-to-apples delta). |
| No unjustified external compute | ✓ | All training + TTT happens within allotted 600s training budget. |

**Legal-category precedents:**
- Pre-quant TTT on training data: merged in PR #1148 (Muon TTT, 1.1179), established in PR #1517 (1.0632, claimed).
- Post-quant eval TTT: merged in PR #549, #1394, #1413, #1477, #1493.

Both gates have been reviewed and approved; the novel element is their composition, not any individual mechanism.

## Section 4 — Timeline (6 days)

| Day | Work | Compute | Cost | Exit gate |
|---|---|---|---|---|
| **1 — 2026-04-25** | RunPod 1×H100 setup. Fetch SP8192 data (`MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf`). Install brotli, sentencepiece, flash_attn_3. Run unmodified #1493 `train_gpt.py` on 1-GPU smoke mode (256K val subset, reduced steps). Verify we reach within 0.003 BPB of their step-matched curve. | 1×H100 × 5h | $10 | Reproduction parity |
| **2 — 2026-04-26** | On 1×H100, add pre-quant TTT phase to `train_gpt.py`. Sweep {lr=1e-4, 3e-4, 1e-3} × {epochs=12, 18, 24} × {freeze=0, 1}, 8 configs total, ~20 min each. Pick winner by post-pre-quant-TTT BPB (pre-GPTQ). | 1×H100 × 6h | $12 | Picked (lr, epochs, freeze) |
| **3 — 2026-04-27** | First 8×H100 full run (600s train + 600s eval) with winning config, seed 42. Check: does pre-quant TTT + eval-time TTT stack? If post-quant eval BPB ≤ 1.075, continue. If > 1.075, drop pre-quant TTT (fall back to pure #1493 reproduction). | 8×H100 × 30 min | $11 | Efficacy gate |
| **4 — 2026-04-28** | Frozen config. 3-seed run: seeds 42, 314, 999. One run at a time (one pod, ~40 min each incl. setup). Collect train logs and metrics. | 8×H100 × 2 h | $43 | 3-seed mean ≤ 1.066 OR ≤ 1.08 |
| **5 — 2026-04-29** | Use `research/collapse_record.py` (or manual) to produce submission `train_gpt.py`. Write README following #1493 template. Fill `submission.json`. Run `research/validate_record.py`. Local smoke check on MLX (optional). | local | $0 | `validate_record.py` passes |
| **6 — 2026-04-30** | Open PR at `openai/parameter-golf` against `main`. Respond to any reviewer comments. Buffer for pod restart / unexpected debugging. | idle / 1×H100 on standby | $20 | PR open before midnight UTC |
| | **Subtotal** | | **$96** | |
| | + Contingency (debugging, re-runs, failed seeds) | | **$200** | |
| | **Total ceiling** | | **~$296** | Well under $1,004 budget |

## Section 5 — Validation protocol

Three gates before submission. Failure at any gate → fall back to the reproduction.

1. **Reproduction gate (Day 1):** On 1×H100 short run of unmodified #1493, TTT-BPB curve at step 2000, 3000, 4000 within 0.003 BPB of published logs (published at `records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT/train_seed*.log`).
2. **Efficacy gate (Day 3):** Single-seed 8×H100 full run, post-quant eval-TTT BPB ≤ 1.075. (Target is 1.065; 1.075 is the "still beating pure reproduction" threshold that justifies the added complexity.)
3. **Compliance gate (Day 5):** `validate_record.py` passes all checks. Artifact < 16,000,000 bytes on all 3 seeds. Train wallclock < 600s on all 3 seeds. Eval wallclock < 600s on all 3 seeds. No eval-time use of `tokens_np[i]` for `i ≥ current_score_position`.

## Section 6 — Risk register

| Risk | P(hit) | Impact | Mitigation | Fallback |
|---|---|---|---|---|
| #1493 reproduction fails (infra or data mismatch) | 15% | High | Follow their exact `MATCHED_FINEWEB_REPO_ID` and flash_attn_3 install command. | Read PR #1493 comments for known reproduction issues. |
| Pre-quant TTT diverges (loss spike) | 20% | Medium | Conservative lr=3e-4 per PR #1517. Grad clip 1.0. Freeze block 0. | Drop freeze to 2 blocks; halve lr; or skip phase entirely. |
| Pre-quant TTT + eval-time TTT double-count (overlapping adaptation hurts) | 30% | Medium | Post-quant TTT runs on quantized weights adapting to eval tokens; pre-quant TTT adapted FP weights to training. Different data, different precision — should compose. | If hurts, drop eval-time TTT, submit at ~1.063 (rank 5-8). |
| Training goes over 600s | 15% | High | Pre-budget: main 550s + pre-quant TTT 25s + GPTQ 3s + buffer 22s. Profile on Day 3. | Reduce pre-quant TTT to 12 epochs (saves ~8s) or reduce main training to 540s. |
| Pod crash / NCCL timeout mid-3-seed-run | 25% | Medium | Save post-EMA checkpoint before pre-quant TTT phase. Resume from there if crash. | Worst case: re-run one seed on fresh pod, ~$15. |
| Reviewer rejects on legality | 5% | High | Both mechanisms have merged precedent. Document legality section explicitly in PR README. | Engage in PR discussion. Worst case: drop pre-quant TTT and resubmit as reproduction. |
| Open PRs get merged, lowering our rank | Ongoing | Low | Not controllable. Focus on absolute BPB, not rank. | Even 1.08 reproduction stays top-12. |

**Hard fallback at any gate failure:** submit the unmodified #1493 reproduction with our 3-seed replication. Even if we're second-author on identical code, a clean independent 3-seed replication is publishable as a non-record and validates our pipeline.

## Section 7 — Deliverables

1. `records/track_10min_16mb/2026-04-30_SP8192_PreQuantTTT_EvalTTT_1.06X/` with:
   - `train_gpt.py` (collapsed, self-contained)
   - `README.md` (following #1493 template; explicit legality section)
   - `submission.json` (3-seed mean, std, per-seed details)
   - `train_seed42.log`, `train_seed314.log`, `train_seed999.log`
   - `requirements.txt`
2. PR opened at `openai/parameter-golf` with attribution to PR #1493 (trunk), PR #1517 (pre-quant TTT recipe), PR #549/1394/1413 (eval-time TTT framework).
3. Design doc (this file) committed to `sprint/top10-crazy` branch for reproducibility.

## Section 8 — Open questions (resolve on Day 1)

1. **Exact SP8192 data version?** The `kevclark/parameter-golf` HF repo should be pinned to a specific snapshot. Verify by file hash against #1493's first log line.
2. **Is the last 64K training tokens actually unseen during main training?** #1493's data loader may already consume the entire stream. If so, reserve from a different shard.
3. **Does freezing "block 0" mean the token embedding + first transformer layer, or just the first transformer layer?** Check PR #1517's code if available; default to "first transformer layer" (embeddings must train for GPTQ calibration stability).
4. **Does `torch.compile` survive the pre-quant TTT phase?** If not, disable compile just for that phase (matches #1493's pattern of disabling compile inside eval-time TTT inner loop).

## Section 9 — Non-goals

- **Not attempting Scylla-level tokenizer search.** 2-3 weeks minimum, 6 days available.
- **Not attempting GatedDeltaNet reproduction (PR #1698, 1.00995).** FLA library dependency and the risk that unmerged claims don't replicate make it too aggressive for 6 days.
- **Not adding any novel architecture.** #1493's trunk is frozen.
- **Not exploring novel quantization schemes.** GPTQ SDClip int6/int8 is already at frontier.
- **Not submitting to non-record track.** The user's goal is main leaderboard top 10.

## Success criteria

- ✅ PR opened at `openai/parameter-golf` by 2026-04-30 23:59 UTC.
- ✅ 3-seed mean val_bpb ≤ 1.075.
- ✅ All compliance gates pass (artifact size, train/eval wallclock, score-first, no n-gram/SLOT/ETLB).
- ✅ Reviewers accept as legitimate (may take time post-deadline).

**Stretch:** val_bpb ≤ 1.065 → rank 3-5.
