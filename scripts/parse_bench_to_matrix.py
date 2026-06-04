#!/usr/bin/env python3
"""parse_bench_to_matrix.py — extract a results/matrix.jsonl row from a
sglang.bench_serving output log.

Usage:
  python3 scripts/parse_bench_to_matrix.py <bench.log> --arm NEW --model Qwen3-32B \
      --config "kernel/on-SM page_first p1 in4096/out512/conc256" [--out results/matrix.jsonl]
"""
import argparse
import json
import re
import sys


def grab(txt, pat):
    m = re.search(pat, txt)
    return float(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("benchlog")
    ap.add_argument("--arm", required=True)
    ap.add_argument("--model", default="Qwen3-32B")
    ap.add_argument("--config", default="")
    ap.add_argument("--out", default="results/matrix.jsonl")
    args = ap.parse_args()

    txt = open(args.benchlog).read()
    row = {
        "arm": args.arm,
        "model": args.model,
        "output_tok_s": grab(txt, r"Output token throughput \(tok/s\):\s*([\d.]+)"),
        "total_tok_s": grab(txt, r"Total token throughput \(tok/s\):\s*([\d.]+)"),
        "req_per_s": grab(txt, r"Request throughput \(req/s\):\s*([\d.]+)"),
        "itl_p50_ms": grab(txt, r"Median ITL \(ms\):\s*([\d.]+)"),
        "itl_p99_ms": grab(txt, r"P99 ITL \(ms\):\s*([\d.]+)"),
        "ttft_p50_ms": grab(txt, r"Median TTFT \(ms\):\s*([\d.]+)"),
        "ttft_p99_ms": grab(txt, r"P99 TTFT \(ms\):\s*([\d.]+)"),
        "duration_s": grab(txt, r"Benchmark duration \(s\):\s*([\d.]+)"),
        "config": args.config,
    }
    if row["output_tok_s"] is None:
        print(f"ERROR: could not parse output throughput from {args.benchlog}", file=sys.stderr)
        sys.exit(1)
    with open(args.out, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row))


if __name__ == "__main__":
    main()
