#!/usr/bin/env bash
# Launch the HiCache kernel + FA3-decode crash-repro server on node-16.
# Config = the validated crash config: guard relaxed (--hicache-kernel-fa3-sync),
# write_through (constant D2H on prefill-commit + finish-commit), fa3 prefill+decode,
# page_size=1 (scattered per-token pages -> big on-SM transfer kernel).
# Usage: bash node16_launch_repro.sh [MEM_FRACTION] [PORT]
set -euo pipefail

MEM_FRACTION="${1:-0.6}"
PORT="${2:-30000}"

source /home/wxu/hicache-fix/venv/bin/activate
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=9.0
cd /home/wxu/hicache-fix/sglang

echo "[launch_repro] mem_fraction=${MEM_FRACTION} port=${PORT} $(date -Is)"
exec python -m sglang.launch_server \
  --model-path /data/wxu/models/Qwen3-32B \
  --tp 8 \
  --mem-fraction-static "${MEM_FRACTION}" \
  --attention-backend fa3 \
  --decode-attention-backend fa3 \
  --enable-hierarchical-cache \
  --hicache-io-backend kernel \
  --hicache-write-policy write_through \
  --hicache-kernel-fa3-sync \
  --page-size 1 \
  --host 127.0.0.1 \
  --port "${PORT}"
