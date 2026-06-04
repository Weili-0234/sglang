#!/usr/bin/env bash
# check_soak_fixed.sh — 0 iff the FIXED (NEW) HiCache config survives a soak on
# MiniMax-M2.7 (default >=2h, or >=SOAK_REQUESTS requests) with ZERO CUDA errors.
#
# Crash-fix correctness criterion, RECONCILED for the masked IMA (see
# check_repro_baseline.sh): the live CUDA IMA cannot reproduce on node-16
# (R575/CUDA-12.9 masks the HW memory-ordering bug). The accepted surrogate is:
# the FIXED on-SM write-back config runs the decode+FA3+write_through+eviction
# workload for the full soak window with zero CUDA errors / no IMA / no process
# death, AND a short compute-sanitizer window is clean (provided by
# check_repro_baseline.sh part B). This script asserts the soak half on M2.7.
#
# Env: SOAK_MODEL (M2.7), SOAK_SECONDS(7200), SOAK_CONC(128), SOAK_PROMPT_LEN(4096),
#      SOAK_OUT_LEN(512), SOAK_PORT(30002), RS_HOST(rs16), READY_TIMEOUT_POLLS(360).
set -uo pipefail
RS="${RS_HOST:-rs16}"
MODEL="${SOAK_MODEL:-/data/wxu/models/MiniMax-M2.7}"
SOAK_SECONDS="${SOAK_SECONDS:-7200}"
CONC="${SOAK_CONC:-128}"
PROMPT_LEN="${SOAK_PROMPT_LEN:-4096}"
OUT_LEN="${SOAK_OUT_LEN:-512}"
PORT="${SOAK_PORT:-30002}"
DIR="/data/wxu/hicache-bench/soak"
fail() { echo "SOAK_GATE: FAIL — $*" >&2; exit 1; }

echo "== launch FIXED config on $MODEL (kernel on-SM, fa3 decode, write_through, page_size=1) =="
ssh "$RS" "pkill -9 -f '[l]aunch_server' 2>/dev/null; sleep 5; mkdir -p $DIR; \
  MODEL='$MODEL' IO_BACKEND=kernel DECODE_ATTN=fa3 setsid nohup \
  bash /home/wxu/hicache-fix/scripts/node16_launch_repro.sh 0.6 $PORT \
  >$DIR/server.log 2>&1 </dev/null & echo launched pid=\$!" || fail "launch failed"

echo "== wait for ready (M2.7 load ~10-20min) =="
ssh "$RS" "L=$DIR/server.log; for i in \$(seq 1 ${READY_TIMEOUT_POLLS:-360}); do \
  grep -q 'fired up and ready' \"\$L\" && { echo READY; grep -m1 max_total_num_tokens \"\$L\"; exit 0; }; \
  pgrep -f '[l]aunch_server' >/dev/null || { echo DIED; tail -30 \"\$L\"; exit 1; }; \
  sleep 5; done; echo TIMEOUT; exit 2" || fail "server did not become ready"

echo "== soak ${SOAK_SECONDS}s: streaming unique-large-prompt workload, monitoring CUDA errors =="
ssh "$RS" "source /home/wxu/hicache-fix/venv/bin/activate; setsid nohup \
  python /home/wxu/hicache-fix/scripts/node16_workload.py --url http://127.0.0.1:$PORT/generate \
  --conc $CONC --prompt-len $PROMPT_LEN --output-len $OUT_LEN --dur $SOAK_SECONDS \
  >$DIR/workload.log 2>&1 </dev/null & echo soak_started"

# Blocking monitor for the whole soak window (this gate is meant to take SOAK_SECONDS).
ssh "$RS" "S=$DIR/server.log; W=$DIR/workload.log; end=\$(( \$(date +%s) + $SOAK_SECONDS + 180 )); \
  while [ \$(date +%s) -lt \$end ]; do \
    if grep -qiE 'illegal memory access|CUDA error|AcceleratorError|uncorrectable|misaligned address' \"\$S\"; then \
      echo SOAK_CUDA_ERROR; grep -niE 'illegal memory|CUDA error|AcceleratorError' \"\$S\" | tail -8; exit 10; fi; \
    if ! pgrep -f '[l]aunch_server' >/dev/null; then echo SOAK_SERVER_DIED; tail -30 \"\$S\"; exit 11; fi; \
    if grep -q WORKLOAD_DONE \"\$W\" 2>/dev/null; then echo SOAK_WORKLOAD_DONE; tail -2 \"\$W\"; exit 0; fi; \
    sleep 30; done; echo SOAK_TIME_REACHED; exit 0" \
  || fail "soak reported a CUDA error / server death"

echo "SOAK_GATE: PASS (>=${SOAK_SECONDS}s on $MODEL, zero CUDA errors). Pair with check_repro_baseline.sh for the compute-sanitizer-clean half."
