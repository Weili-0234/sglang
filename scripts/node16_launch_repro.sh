#!/usr/bin/env bash
# Launch a HiCache + FA3 server on node-16, configurable for every matrix arm.
# Base config = the validated crash/NEW config: guard relaxed (--hicache-kernel-fa3-sync),
# write_through (constant D2H), page_size=1. Env overrides select the arm:
#   MODEL        (default Qwen3-32B)   -- /data/wxu/models/MiniMax-M2.7 for soak
#   IO_BACKEND   (default kernel)      -- kernel=on-SM(NEW) | direct=copy-engine | staged=#21631
#   DECODE_ATTN  (default fa3)         -- flashinfer/triton => "ceiling" (safe non-fa3 overlap)
#   PAGE_SIZE    (default 1)
#   MEM_FRACTION (arg1, default 0.6)   PORT (arg2, default 30000)
#   EXTRA_ARGS   (default "")          -- e.g. SGLANG_HICACHE_FORCE_STREAM_SYNC handled via env
# Usage: [MODEL=..] [IO_BACKEND=..] [DECODE_ATTN=..] bash node16_launch_repro.sh [MEM_FRACTION] [PORT]
set -euo pipefail

MEM_FRACTION="${1:-0.6}"
PORT="${2:-30000}"
MODEL="${MODEL:-/data/wxu/models/Qwen3-32B}"
IO_BACKEND="${IO_BACKEND:-kernel}"
DECODE_ATTN="${DECODE_ATTN:-fa3}"
PAGE_SIZE="${PAGE_SIZE:-1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

source /home/wxu/hicache-fix/venv/bin/activate
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=9.0
cd /home/wxu/hicache-fix/sglang

echo "[launch] model=${MODEL} io=${IO_BACKEND} decode_attn=${DECODE_ATTN} page=${PAGE_SIZE} mem=${MEM_FRACTION} port=${PORT} disable_jit=${SGLANG_HICACHE_DISABLE_JIT:-0} extra='${EXTRA_ARGS}' $(date -Is)"
exec python -m sglang.launch_server \
  --model-path "${MODEL}" \
  --tp 8 \
  --mem-fraction-static "${MEM_FRACTION}" \
  --attention-backend fa3 \
  --decode-attention-backend "${DECODE_ATTN}" \
  --enable-hierarchical-cache \
  --hicache-io-backend "${IO_BACKEND}" \
  --hicache-write-policy write_through \
  --hicache-kernel-fa3-sync \
  --page-size "${PAGE_SIZE}" \
  --host 127.0.0.1 \
  --port "${PORT}" ${EXTRA_ARGS}
