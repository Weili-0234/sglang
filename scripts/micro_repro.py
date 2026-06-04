#!/usr/bin/env python3
"""Targeted micro-repro of the HiCache kernel + FA3 IMA.

Hypothesis (PLAN): the on-SM AOT D2H transfer (transfer_kv_all_layer ->
transfer_kernel_impl, ld.global.nc + st.global.cg to host-pinned memory, NO
device fence) running concurrently with an FA3 kernel corrupts FA3's TMA
descriptors -> CUDA illegal memory access.

FIDELITY (evaluator round 2):
 * Stream topology mirrors production: FA3 runs on the MAIN/forward stream
   (torch current stream) and the AOT D2H copy on a SEPARATE non-default
   write_stream (= torch.cuda.Stream(), like cache_controller.write_stream),
   with NO cross-stream event linking them -- the missing sync IS the bug.
 * --mode {decode,prefill}: prefill uses flash_attn_varlen_func (different
   varlen kernels / TMA-descriptor usage = the flagged residual race window).
 * --measure-overlap: empirically proves FA3 || copy actually overlap (refutes
   "they serialize, so there's no race").

Run natively, then under:
    compute-sanitizer --tool memcheck  python scripts/micro_repro.py ...
    compute-sanitizer --tool racecheck python scripts/micro_repro.py ...
"""
import argparse
import sys
import time

import torch

from sgl_kernel.flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from sgl_kernel.kvcacheio import transfer_kv_all_layer


def build_fa3_decode(args, device, dtype):
    b, hq, hkv, d = args.batch, args.nheads_q, args.nheads_kv, args.headdim
    pbs = args.page_block_size
    blocks_per_seq = (args.cache_seqlen + pbs - 1) // pbs
    num_blocks = b * blocks_per_seq
    q = torch.randn(b, 1, hq, d, device=device, dtype=dtype)
    k_cache = torch.randn(num_blocks, pbs, hkv, d, device=device, dtype=dtype)
    v_cache = torch.randn(num_blocks, pbs, hkv, d, device=device, dtype=dtype)
    cache_seqlens = torch.full((b,), args.cache_seqlen, dtype=torch.int32, device=device)
    page_table = torch.arange(num_blocks, dtype=torch.int32, device=device).reshape(
        b, blocks_per_seq
    )
    return dict(q=q, k_cache=k_cache, v_cache=v_cache,
                cache_seqlens=cache_seqlens, page_table=page_table)


def build_fa3_prefill(args, device, dtype):
    b, hq, hkv, d = args.prefill_batch, args.nheads_q, args.nheads_kv, args.headdim
    s = args.prefill_seqlen
    total = b * s
    q = torch.randn(total, hq, d, device=device, dtype=dtype)
    k = torch.randn(total, hkv, d, device=device, dtype=dtype)
    v = torch.randn(total, hkv, d, device=device, dtype=dtype)
    cu = torch.arange(0, (b + 1) * s, s, dtype=torch.int32, device=device)
    return dict(q=q, k=k, v=v, cu_seqlens=cu, max_seqlen=s)


def build_transfer(args, device, dtype):
    nl, total, ie = args.num_layers, args.total_items, args.item_elems
    dev_k = [torch.randn(total, ie, device=device, dtype=dtype) for _ in range(nl)]
    dev_v = [torch.randn(total, ie, device=device, dtype=dtype) for _ in range(nl)]
    host_k = [torch.zeros(total, ie, dtype=dtype).pin_memory() for _ in range(nl)]
    host_v = [torch.zeros(total, ie, dtype=dtype).pin_memory() for _ in range(nl)]
    mk = lambda ts: torch.tensor([t.data_ptr() for t in ts], dtype=torch.uint64, device=device)
    n = args.transfer_items
    return dict(
        keepalive=(dev_k, dev_v, host_k, host_v),
        src_k_ptrs=mk(dev_k), src_v_ptrs=mk(dev_v),
        dst_k_ptrs=mk(host_k), dst_v_ptrs=mk(host_v),
        src_indices=torch.randint(0, total, (n,), dtype=torch.int64, device=device),
        dst_indices=torch.arange(n, dtype=torch.int64, device=device),
        item_size_bytes=ie * dtype.itemsize,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--mode", choices=["decode", "prefill"], default="decode")
    ap.add_argument("--measure-overlap", action="store_true")
    # FA3 shape (Qwen3-32B-ish)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--nheads-q", type=int, default=64)
    ap.add_argument("--nheads-kv", type=int, default=8)
    ap.add_argument("--headdim", type=int, default=128)
    ap.add_argument("--cache-seqlen", type=int, default=4096)
    ap.add_argument("--page-block-size", type=int, default=256)
    ap.add_argument("--prefill-batch", type=int, default=8)
    ap.add_argument("--prefill-seqlen", type=int, default=4096)
    # transfer shape
    ap.add_argument("--num-layers", type=int, default=64)
    ap.add_argument("--total-items", type=int, default=16384)
    ap.add_argument("--transfer-items", type=int, default=4096)
    ap.add_argument("--item-elems", type=int, default=1024)
    ap.add_argument("--block-quota", type=int, default=2)
    # loop
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--inner", type=int, default=8)
    ap.add_argument("--transfers-per-inner", type=int, default=8)
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16
    torch.cuda.set_device(device)

    fa = build_fa3_decode(args, device, dtype) if args.mode == "decode" else build_fa3_prefill(args, device, dtype)
    tr = build_transfer(args, device, dtype)
    torch.cuda.synchronize()

    # Production topology: FA3 on the MAIN (current) stream, copy on a SEPARATE
    # non-default write_stream, NO event linking them.
    # Production overlap needs two NON-default streams: torch's current stream is
    # the LEGACY default (stream 0) which blocks/serializes other streams (verified
    # ratio 1.0). sglang's forward runs on a non-default stream, so mirror that with
    # a created compute_stream || write_stream, NO linking event (the missing sync IS the bug).
    compute_stream = torch.cuda.Stream(device=device)
    write_stream = torch.cuda.Stream(device=device)
    print(f"[micro_repro] mode={args.mode} compute_stream={compute_stream} write_stream={write_stream} "
          f"block_quota={args.block_quota}", flush=True)

    def launch_fa3():
        with torch.cuda.stream(compute_stream):
            if args.mode == "decode":
                flash_attn_with_kvcache(
                    fa["q"], fa["k_cache"], fa["v_cache"],
                    cache_seqlens=fa["cache_seqlens"], page_table=fa["page_table"], causal=True,
                )
            else:
                flash_attn_varlen_func(
                    fa["q"], fa["k"], fa["v"],
                    cu_seqlens_q=fa["cu_seqlens"], cu_seqlens_k=fa["cu_seqlens"],
                    max_seqlen_q=fa["max_seqlen"], max_seqlen_k=fa["max_seqlen"], causal=True,
                )

    def launch_transfer():
        with torch.cuda.stream(write_stream):
            transfer_kv_all_layer(
                tr["src_k_ptrs"], tr["dst_k_ptrs"], tr["src_v_ptrs"], tr["dst_v_ptrs"],
                tr["src_indices"], tr["dst_indices"],
                item_size=tr["item_size_bytes"], num_layers=args.num_layers,
                block_quota=args.block_quota,
            )

    # warmup
    launch_fa3(); launch_transfer(); torch.cuda.synchronize()

    if args.measure_overlap:
        def timed(fn, n=100):
            torch.cuda.synchronize(); t = time.time()
            for _ in range(n):
                fn()
            torch.cuda.synchronize()
            return (time.time() - t) / n * 1000.0  # ms/iter wall-clock (captures all streams)
        t_fa3 = timed(lambda: launch_fa3())
        t_cp = timed(lambda: launch_transfer())
        def both():
            launch_fa3()
            launch_transfer()
        t_both = timed(both)
        serial = t_fa3 + t_cp
        ratio = t_both / serial if serial else 0
        print(f"[micro_repro] OVERLAP CHECK: fa3={t_fa3:.3f}ms copy={t_cp:.3f}ms "
              f"concurrent={t_both:.3f}ms serial_sum={serial:.3f}ms ratio={ratio:.2f} "
              f"=> {'OVERLAP' if t_both < 0.9*serial else 'SERIALIZED/NO-OVERLAP'}", flush=True)
        return

    t0 = time.time()
    for it in range(args.iters):
        for _ in range(args.inner):
            launch_fa3()
            for _ in range(args.transfers_per_inner):
                launch_transfer()
        try:
            torch.cuda.synchronize()
        except RuntimeError as e:
            print(f"[micro_repro] *** CUDA ERROR at iter {it} (t={time.time()-t0:.1f}s): {e}", flush=True)
            print("CRASH_REPRODUCED", flush=True)
            sys.exit(42)
        if it % 50 == 0:
            print(f"[micro_repro] iter {it} ok (t={time.time()-t0:.1f}s)", flush=True)
    print(f"[micro_repro] completed {args.iters} iters mode={args.mode} NO crash (t={time.time()-t0:.1f}s)", flush=True)
    print("NO_CRASH", flush=True)


if __name__ == "__main__":
    main()
