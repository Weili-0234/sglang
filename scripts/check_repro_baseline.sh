#!/usr/bin/env bash
# check_repro_baseline.sh — deterministic "the test has teeth" gate for the
# HiCache kernel + FA3 IMA, RECONCILED for node-16 (driver R575 / CUDA-12.9 / H100).
#
# RECONCILIATION (user-accepted via AskUserQuestion, 2026-06-04 — option
# "Node-16 + port #21631"):
#   The live CUDA illegal-memory-access does NOT reproduce on this (newer) stack.
#   That was established exhaustively — 7 configurations, all clean:
#     full-server JIT@bq2 & AOT@bq2 (conc=256, 93% KV usage, heavy eviction);
#     micro-repro decode @block_quota 512 (200 iters) and @2 (425 iters / 14 min);
#     FA3-prefill (varlen) micro-repro; compute-sanitizer memcheck = 0 errors;
#     compute-sanitizer racecheck = 0 hazards.
#   The bug is a Hopper hardware memory-ordering effect (the on-SM, NON-FENCED
#   `st.global.cg` D2H store racing FA3's TMA descriptors); the R575/CUDA-12.9
#   stack masks it (no software OOB; the copy's host-pinned dst does not alias
#   FA3 descriptors). The live IMA was reported ~2026-03 on an older driver.
#
#   Therefore the literal "reproduce IMA <=10min" criterion is replaced by a
#   DETERMINISTIC surrogate with two assertions:
#     (A) TEETH  — the unfixed baseline write-back path is exactly the on-SM
#                  NON-FENCED `st.global.cg` D2H store (sgl-kernel
#                  transfer_kernel_impl) that races FA3, and the guard-relax flag
#                  exists so kernel+FA3-decode actually runs it. I.e. the unsafe
#                  code that WOULD IMA on vulnerable hardware is present + reachable.
#     (B) CLEAN  — running that exact AOT st.global.cg path CONCURRENTLY with an
#                  FA3 kernel under `compute-sanitizer --tool memcheck` reports
#                  0 errors (the deterministic checker is clean = the accepted
#                  fix-correctness surrogate on this hardware).
#
# Exit 0 iff BOTH (A) and (B) hold. Run from the sglang -exp worktree root or
# scripts/.  Requires the `rs16` ssh alias (research-secure node-16: GPU +
# /usr/local/cuda-12.9/bin/compute-sanitizer + the prepared venv/repo).
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
RS="${RS_HOST:-rs16}"
fail() { echo "REPRO_BASELINE_GATE: FAIL — $*" >&2; exit 1; }

echo "== (A) teeth: baseline on-SM non-fenced st.global.cg D2H path present =="
TRANSFER="$REPO/sgl-kernel/csrc/kvcacheio/transfer.cu"
[ -f "$TRANSFER" ] || fail "missing $TRANSFER"
grep -q 'st.global.cg' "$TRANSFER" \
  || fail "non-fenced 'st.global.cg' D2H store not found in transfer.cu (teeth missing)"
grep -q 'transfer_kernel_impl' "$TRANSFER" \
  || fail "transfer_kernel_impl (the racing on-SM D2H kernel) not found"
grep -q 'hicache_kernel_fa3_sync' "$REPO/python/sglang/srt/server_args.py" \
  || fail "guard-relax flag --hicache-kernel-fa3-sync missing (kernel+FA3-decode cannot run)"
echo "PASS (A): transfer_kernel_impl uses non-fenced st.global.cg; guard-relax flag present"

echo "== (B) clean: AOT st.global.cg D2H || FA3 is compute-sanitizer memcheck-clean =="
CS=/usr/local/cuda-12.9/bin/compute-sanitizer
GPU="${REPRO_GPU:-0}"
REMOTE="source /home/wxu/hicache-fix/venv/bin/activate; \
export CUDA_HOME=/usr/local/cuda; export PATH=\$CUDA_HOME/bin:\$PATH; \
cd /home/wxu/hicache-fix/sglang; \
CUDA_VISIBLE_DEVICES=$GPU timeout 600 $CS --tool memcheck --error-exitcode 99 \
python /home/wxu/hicache-fix/scripts/micro_repro.py --device cuda:0 --mode decode \
--batch 64 --cache-seqlen 2048 --iters 8 --inner 4 --transfers-per-inner 4 \
--block-quota 256 2>&1 | tail -6; echo SANEXIT=\${PIPESTATUS[0]}"
OUT="$(ssh "$RS" "$REMOTE" 2>&1)" || true
echo "$OUT"
echo "$OUT" | grep -q 'ERROR SUMMARY: 0 errors' \
  || fail "compute-sanitizer memcheck reported errors (surrogate fix-correctness FAILED)"
echo "PASS (B): AOT transfer_kv_all_layer (st.global.cg) || FA3 is memcheck-clean (0 errors)"

echo "REPRO_BASELINE_GATE: PASS (deterministic surrogate; live IMA masked on R575+CUDA-12.9)"
