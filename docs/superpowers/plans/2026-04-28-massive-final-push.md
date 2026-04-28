# Massive Final Push Plan — Parameter Golf (2026-04-28 → 2026-04-30)

> **Two days to deadline. Two parallel tracks. One goal: land on the leaderboard.**

## Context

- **Deadline:** 2026-04-30 23:59 UTC (≈ 51 hours from now)
- **What we have:** A complete, validated, ready-to-PR submission at `submission/pr1797-repro-1.06136` (3-seed mean **1.06136 BPB**, beats merged SOTA by 33σ, beats PR #1797's claim by 5.8σ)
- **What we built:** A pre-quant TTT plan in `.worktrees/top10-crazy/` that targets ≤ 1.066 BPB (potentially better than the safety net)
- **Hard blocker:** outbound SSH from this Mac is firewalled (port 22 + RunPod high ports both time out). I cannot drive GPU runs from this session — but the user can via RunPod's web terminal or from any unblocked network.
- **Already running:** Pod `actt24lu2d0ofz` (1×H100, $2.99/hr) — burning ~$72/day. Either use it or stop it.

## Two-track strategy

```
TRACK A (SAFETY NET) ─── ship parked 1.06136 PR ──────────► top-10 LOCKED
                            (HTTPS only, I can do this)

TRACK B (UPSIDE SWING) ── pre-quant TTT GPU sprint ───────► top-3 to 6
                            (needs user's web terminal)
```

Run both in parallel. If B fails or runs late, A is already merged. If B succeeds with a better score, withdraw A and ship B.

---

## TRACK A — Safety-net PR (≤ 30 min, no GPU)

### Status

The branch `submission/pr1797-repro-1.06136` (commit `dba870a`) on local `main`'s tip has the complete record folder at `records/track_10min_16mb/2026-04-25_PR1797Reproduction_3Seed_1.06136/` — all 9 required files (README, submission.json, train_gpt.py LZMA-wrapped, 3 seed logs, lossless_caps.py, prepare_caseops_data.py, tokenizer model). Compliance pre-verified by the contributor.

### Steps

| # | Action | Tool | Time |
|---|---|---|---|
| A1 | Confirm branch ahead of `origin/main` and clean | `git status`, `git log` | 1 min |
| A2 | Push branch to fork via HTTPS | `git push -u fork submission/pr1797-repro-1.06136` | 2 min |
| A3 | Open PR on `openai/parameter-golf` (HTTPS via gh) | `gh pr create` with templated body crediting #1394→#1493→#1736→#1787→#1797 chain | 3 min |
| A4 | Record PR URL to `PR_URL_safety_net.txt` | local file | 1 min |

**I CAN execute A1–A4 autonomously now (HTTPS works).** I will pause for explicit OK before A2/A3 since these are visible-to-others actions.

---

## TRACK B — GPU sprint (≈ 7 hr GPU, ≈ $50, user-driven on pod)

### Phase B0: Decide pod strategy (now, 5 min)

Existing pod `actt24lu2d0ofz` is a 1×H100 NVL at $2.99/hr. Two options:
- **B0-keep:** Use it for dev/sweep (Phases B1–B4). Stop it before final 8×H100 runs to avoid double-billing.
- **B0-fresh:** Stop it now (saves money during planning), then provision fresh 1×H100 SXM when ready (~5 min provisioning, $1.50/hr).

User decision: I'll default to **keep** (already paid the boot cost) unless told otherwise.

### Phase B1: Bootstrap dev pod (10 min, $0.50)

User opens RunPod web terminal for `actt24lu2d0ofz` and runs:

```bash
cd /workspace
[ ! -d parameter-golf ] && git clone https://github.com/AayushBaniya2006/parameter-golf.git
cd parameter-golf
git fetch origin
git checkout sprint/top10-crazy 2>/dev/null || git checkout -b sprint/top10-crazy origin/sprint/top10-crazy
git pull
pip install --quiet brotli sentencepiece
pip install --quiet flash_attn_3 --no-deps --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/
[ ! -d data/datasets/fineweb10B_sp8192 ] && MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192
echo "BOOTSTRAP OK"
nvidia-smi --query-gpu=name --format=csv,noheader
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Branch `sprint/top10-crazy` (which I'll push from here in Phase B0.5) carries our `train_gpt_src.py` + scripts. Bootstrap script in repo at `.worktrees/top10-crazy/scripts/runpod_bootstrap.sh` — equivalent to the inline above.

### Phase B0.5: Push `sprint/top10-crazy` to fork (now, 2 min)

Currently the worktree branch hasn't been pushed. Without push, the pod can't `git checkout sprint/top10-crazy`. I'll push `sprint/top10-crazy` to fork in Track A's same wave.

### Phase B2: Reproduction gate (30 min, $1.50)

Verify the pod can reproduce the published PR #1493 baseline before adding our novel lever.

```bash
cp records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT/train_gpt.py /workspace/pg_1493_baseline.py
mkdir -p logs
SEED=42 VOCAB_SIZE=8192 QK_GAIN_INIT=5.25 \
  TTT_ENABLED=0 ITERATIONS=3000 MAX_WALLCLOCK_SECONDS=600 \
  VAL_LOSS_EVERY=500 RUN_ID=repro_1gpu_3k \
  torchrun --standalone --nproc_per_node=1 /workspace/pg_1493_baseline.py 2>&1 | tee logs/repro_1gpu_3k.log
grep "val_bpb" logs/repro_1gpu_3k.log
```

**Gate:** val_bpb at steps 500/1000/1500/2000/2500/3000 within ±0.01 of `records/.../train_seed42.log` reference curve.

If fails → infra issue. Stop and debug before continuing. Likely causes: SP8192 data mismatch, missing flash_attn_3, env var mismatch.

### Phase B3: Pack + smoke pre-quant TTT (20 min, $1)

```bash
# from this Mac (HTTPS-based, I can do this):
python3 .worktrees/top10-crazy/scripts/pack_record.py \
  .worktrees/top10-crazy/train_gpt_src.py /tmp/pg_modified.py
git -C .worktrees/top10-crazy add . && git -C .worktrees/top10-crazy commit -m "Pack modified train_gpt for pod"
git -C .worktrees/top10-crazy push fork sprint/top10-crazy
```

Then on pod (web terminal):

```bash
cd /workspace/parameter-golf && git pull
python3 .worktrees/top10-crazy/scripts/pack_record.py train_gpt_src.py /workspace/pg_modified.py
SEED=42 VOCAB_SIZE=8192 QK_GAIN_INIT=5.25 \
  TTT_ENABLED=0 PRE_QUANT_TTT_ENABLED=1 PRE_QUANT_TTT_EPOCHS=6 \
  PRE_QUANT_TTT_LR=3e-4 PRE_QUANT_TTT_CHUNK_TOKENS=32768 \
  ITERATIONS=500 MAX_WALLCLOCK_SECONDS=600 VAL_LOSS_EVERY=250 \
  RUN_ID=smoke_pqttt_1gpu \
  torchrun --standalone --nproc_per_node=1 /workspace/pg_modified.py 2>&1 | tee logs/smoke_pqttt_1gpu.log
```

**Smoke checks (must pass all):**
1. Log line `pre_quant_ttt:start epochs=6 lr=0.0003 chunk_tokens=32768 freeze_blocks=1` appears.
2. Each `pre_quant_ttt:epoch N/6 loss=X.XXXX` line shows decreasing loss.
3. `pre-quantization post-ptt val_bpb` is **lower** than `pre-quantization post-ema val_bpb`.
4. No NaN/Inf, no NCCL errors, no OOM.

### Phase B4: Hyperparameter sweep (3.5 hr, $11)

```bash
# on pod:
bash /workspace/parameter-golf/.worktrees/top10-crazy/scripts/run_ttt_sweep.sh
```

Sweep is now crash-tolerant (commit `7bfb81d`): single-config OOM/NaN logs WARN and continues. At end, ranking printed sorted ascending by `pre-quantization post-ptt` BPB. Pick the top entry.

### Phase B5: 8×H100 SXM provisioning + efficacy gate (35 min, $11)

User provisions a fresh 8×H100 SXM pod via RunPod web (or runpodctl from a network where SSH works). After bootstrap (same steps as B1):

```bash
# pack winner config + ship:
SEED=42 VOCAB_SIZE=8192 QK_GAIN_INIT=5.25 \
  TTT_ENABLED=1 TTT_LR=0.005 TTT_EPOCHS=3 \
  PRE_QUANT_TTT_ENABLED=1 \
  PRE_QUANT_TTT_EPOCHS=<winner> PRE_QUANT_TTT_LR=<winner> \
  PRE_QUANT_TTT_CHUNK_TOKENS=65536 PRE_QUANT_TTT_FREEZE_BLOCKS=<winner> \
  MAX_WALLCLOCK_SECONDS=600 RUN_ID=full_seed42 \
  torchrun --standalone --nproc_per_node=8 /workspace/pg_modified.py 2>&1 | tee logs/full_seed42.log
```

**Efficacy gate:** final `quantized_ttt val_bpb ≤ 1.075`.
- Pass → Phase B6
- Fail (1.075 < x ≤ 1.081) → ship Track A (safety net at 1.06136 wins)
- Fail (x > 1.081) → infra/code issue, stop and debug

### Phase B6: 3-seed run (75 min, $33)

```bash
for SEED in 314 999; do
  SEED=$SEED VOCAB_SIZE=8192 QK_GAIN_INIT=5.25 \
    TTT_ENABLED=1 TTT_LR=0.005 TTT_EPOCHS=3 \
    PRE_QUANT_TTT_ENABLED=1 \
    PRE_QUANT_TTT_EPOCHS=<winner> PRE_QUANT_TTT_LR=<winner> \
    PRE_QUANT_TTT_CHUNK_TOKENS=65536 PRE_QUANT_TTT_FREEZE_BLOCKS=<winner> \
    MAX_WALLCLOCK_SECONDS=600 RUN_ID=full_seed${SEED} \
    torchrun --standalone --nproc_per_node=8 /workspace/pg_modified.py \
      2>&1 | tee logs/full_seed${SEED}.log
done
```

(seed 42 already done in B5)

Then user runs `runpodctl send logs/full_seed*.log` (croc P2P transfer) or pushes the logs to a branch on the fork via `git push`. I can pull from there.

### Phase B7: Submission assembly (30 min, $0)

I do this back here:

```bash
cd .worktrees/top10-crazy
SCORE=<3-seed-mean as 5-digit, e.g. 10654>
DIR=records/track_10min_16mb/2026-04-30_SP8192_PreQuantTTT_EvalTTT_${SCORE}
mkdir -p $DIR
python3 scripts/pack_record.py train_gpt_src.py $DIR/train_gpt.py
cp logs/full_seed42.log  $DIR/train_seed42.log
cp logs/full_seed314.log $DIR/train_seed314.log
cp logs/full_seed999.log $DIR/train_seed999.log
# Write submission.json + README from templates
python3 scripts/validate_submission.py $DIR --allow-fallback
```

### Phase B8: Second PR (5 min, $0)

```bash
git -C .worktrees/top10-crazy add records/...
git -C .worktrees/top10-crazy commit -m "Record: Pre-Quant TTT + ... — val_bpb $SCORE_DOT (3-seed)"
git -C .worktrees/top10-crazy push fork sprint/top10-crazy
gh pr create --repo openai/parameter-golf --base main \
  --head AayushBaniya2006:sprint/top10-crazy \
  --title "Record: Pre-Quant TTT + ... — val_bpb <X.XXXX>" \
  --body "<<EOF...EOF"
```

If Track A's PR (1.06136) was already merged and Track B's score is better, comment on Track A asking maintainers to prefer B.

### Phase B9: Cleanup (2 min, $0)

```bash
runpodctl pod stop actt24lu2d0ofz  # 1xH100 NVL dev pod
runpodctl pod stop <8xh100-id>     # final pod
```

Verify with `runpodctl pod list`.

---

## Time + cost ledger

| Phase | What | Wall time | Cost |
|---|---|---|---|
| A | Safety-net PR | 30 min | $0 |
| B0–B0.5 | Decide + push | 5 min | $0 |
| B1 | Bootstrap dev pod | 10 min | $0.50 |
| B2 | Reproduction gate | 30 min | $1.50 |
| B3 | Pre-quant TTT smoke | 20 min | $1.00 |
| B4 | Sweep (3.5 hr 1×H100) | 3.5 hr | $11 |
| B5 | 8×H100 efficacy | 35 min | $11 |
| B6 | 3-seed (8×H100) | 75 min | $33 |
| B7 | Submission assembly | 30 min | $0 |
| B8 | Second PR | 5 min | $0 |
| B9 | Cleanup | 2 min | $0 |
| **TOTAL** | | **~7.5 hr GPU + 1 hr orchestration** | **~$58** |

Budget remaining: $1,004 − $58 = $946 buffer.

---

## Critical decision points

1. **End of B2 (reproduction gate fail):** stop. Network/infra issue is unlikely to be fixable in our window. Ship Track A only.
2. **End of B3 (smoke fail):** if `import math` or chunk-loop bugs reappear, fix in worktree (5 min) and retry. If no fix found in 30 min, ship Track A only.
3. **End of B4 (sweep): ALL configs > 1.10 BPB:** something fundamental is wrong with pre-quant TTT composition. Ship Track A only.
4. **End of B5 (efficacy): val_bpb > 1.06136:** Track B is worse than the safety net. Don't open PR; B7–B8 are skipped.
5. **End of B6 (3-seed): mean > 1.06136:** Same as above. Track A wins.
6. **Approaching deadline (< 6 hr remaining):** if B is still in B4 or B5, abort B and rely on A.

---

## What I do autonomously (HTTPS only)

- Track A1, A2, A3, A4 — push branch + open PR (after explicit user OK)
- B0.5 — push `sprint/top10-crazy` to fork
- Phase B7 — assemble submission folder once seed logs land
- Phase B8 — open second PR (after explicit user OK)
- Pod state monitoring (`runpodctl pod list`)
- Pod stopping (`runpodctl pod stop`) for cleanup
- Memory updates as state changes

## What needs the user

- Phases B1–B6 — running commands on pod via RunPod web terminal (or any network where SSH works)
- Provisioning the 8×H100 SXM pod (web UI or `runpodctl pod create` from unblocked network)
- Confirming each PR creation (Track A, Track B)
- Sending logs back to me (via `runpodctl send` croc P2P, or via git push to a logs branch on fork)

## Communication protocol while user runs pod work

After each phase, user pastes a one-liner status + key log excerpt back here. I'll confirm gate pass/fail, advise next phase, update memory. Pattern:

> User: "B2 done, last 3 val_bpb lines: ..."
> Me: "Gate passed, proceed to B3 with `<exact command>`."

---

## Risk register (additions on top of the spec's)

| Risk | Mitigation |
|---|---|
| User's network is blocked → can't do GPU work | Use RunPod web terminal; or a different network (mobile hotspot, VPN); or a friend's machine |
| Pod IP/port changes mid-run | Reconnect from web terminal; runpodctl pod list still works |
| Seed logs lost when pod stops | Push logs to fork: `cd /workspace/parameter-golf && git checkout -b sprint/logs-$(date +%s) && git add logs/ && git commit -m "logs" && git push fork`. I pull from fork. |
| Time runs out mid-B6 | B5's seed 42 alone may be enough for a non-record submission with caveat |
| Both Track A and B fail to merge by deadline | Even unmerged PRs are timestamped. As long as PR opened before deadline, it's eligible per competition rules. |

---

## Memory snapshot

After each phase, I'll update `~/.claude/projects/.../memory/sprint_top10_crazy.md` with:
- Current phase + gate pass/fail
- Pod IDs (running + stopped)
- Cost-to-date
- Decision-point outcomes

So the sprint state survives any session/context boundary.
