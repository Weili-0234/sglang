#!/usr/bin/env python3
"""Continuous streaming load to maximize D2H-write-back  ∥  FA3-decode overlap.

Why this shape (see PLAN root-cause analysis):
  For write_through HiCache the on-SM D2H KV back-up kernel fires only at
  (1) prefill-commit (prompt KV) and (2) request-finish (output KV) -- it is
  BURSTY, not continuous, and eviction adds no D2H. To race the FA3 decode
  persistent kernel we need prefill-commits happening WHILE a large batch
  decodes. So: keep `conc` requests in flight continuously (immediate
  replacement), each with a UNIQUE large random prompt (cache miss => full
  prompt D2H, no `.backuped` skip) and a short-ish fixed output (fast turnover
  => high prefill-commit rate) with ignore_eos (deterministic decode length).

A server IMA crash shows up here as a spike in `errs` (in-flight requests get
connection resets / 5xx) and in the server log as "illegal memory access".
"""
import argparse
import random
import threading
import time

import requests


def worker(args, deadline, stats, wid):
    rng = random.Random(wid * 100003 + 7)
    sess = requests.Session()
    n = 0
    while time.time() < deadline:
        # Unique random prompt -> guaranteed radix cache miss -> full prompt D2H.
        ids = [rng.randint(5, args.vocab) for _ in range(args.prompt_len)]
        payload = {
            "input_ids": ids,
            "sampling_params": {
                "max_new_tokens": args.output_len,
                "ignore_eos": True,
                "temperature": 0.0,
            },
        }
        try:
            r = sess.post(args.url, json=payload, timeout=args.req_timeout)
            if r.status_code != 200:
                stats["errs"] += 1
                stats["last_err"] = f"HTTP {r.status_code}: {r.text[:200]}"
            else:
                stats["ok"] += 1
        except Exception as e:  # connection reset on crash, timeouts, etc.
            stats["errs"] += 1
            stats["last_err"] = repr(e)[:200]
        n += 1
    stats["per_worker"][wid] = n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:30000/generate")
    ap.add_argument("--conc", type=int, default=256)
    ap.add_argument("--prompt-len", type=int, default=4096)
    ap.add_argument("--output-len", type=int, default=512)
    ap.add_argument("--dur", type=int, default=300)
    ap.add_argument("--vocab", type=int, default=150000)
    ap.add_argument("--req-timeout", type=int, default=600)
    ap.add_argument("--report-interval", type=int, default=10)
    args = ap.parse_args()

    print(
        f"WORKLOAD start conc={args.conc} prompt_len={args.prompt_len} "
        f"output_len={args.output_len} dur={args.dur}s url={args.url}",
        flush=True,
    )
    stats = {"ok": 0, "errs": 0, "last_err": "", "per_worker": {}}
    start = time.time()
    deadline = start + args.dur
    threads = [
        threading.Thread(target=worker, args=(args, deadline, stats, i), daemon=True)
        for i in range(args.conc)
    ]
    for t in threads:
        t.start()
    while any(t.is_alive() for t in threads):
        time.sleep(args.report_interval)
        print(
            f"[t={int(time.time() - start)}s] ok={stats['ok']} errs={stats['errs']} "
            f"last_err={stats['last_err']}",
            flush=True,
        )
    for t in threads:
        t.join()
    print(
        f"WORKLOAD_DONE ok={stats['ok']} errs={stats['errs']} last_err={stats['last_err']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
