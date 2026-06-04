#!/usr/bin/env python3
"""check_perf_gates.py — deterministic PERF gate for the HiCache write-back fix.

Reads results/matrix.jsonl (one JSON object per arm) and asserts ALL of
SUCCESS_CONDITION criterion 2 hold for the winning fix `NEW` vs the baselines:

  NEW.output_tok_s  >  #21631.output_tok_s * (1 + MARGIN)
  NEW.output_tok_s  >  PR20611.output_tok_s
  NEW.output_tok_s  >= 0.98 * ceiling.output_tok_s
  NEW.itl_p99_ms    <= #21631.itl_p99_ms  AND  <= PR20611.itl_p99_ms

GATE RELAXATION (user-approved 2026-06-04 via AskUserQuestion, "Relax gate to
NEW's real wins"): the ORIGINAL HARD GATE was MARGIN=0.02 (+2% beyond noise).
That proved UNREACHABLE because the on-SM D2H write-back is only ~1% of e2e on
these compute-bound runs (Qwen3-32B run-noise ~1%; M2.7 MoE run-noise +-13%),
so the real 3-6x write-back micro-bench advantage (page_size=1) is e2e-hidden.
Measured: NEW(on-SM,bq1)=3706 vs #21631=3640 tok/s = +1.8% (0.2% short of +2%),
but NEW BEATS #21631 on ITL_p99 (468<=490), is >=0.98x ceiling, AND is
correctness-equivalent (Strategy B/GDC is the DEFAULT FA3 build -> no IMA).
=> MARGIN relaxed to 0.0 (NEW strictly faster than #21631 = "tie-or-better",
here +1.8%), keeping the ITL_p99<=both, >PR20611, and >=0.98x ceiling checks.
Pass `--margin 0.02` to re-assert the original strict gate.

Each matrix.jsonl row:
  {"arm": "NEW"|"#21631"|"PR20611"|"direct"|"ceiling", "model": "...",
   "output_tok_s": float, "itl_p50_ms": float, "itl_p99_ms": float,
   "ttft_p99_ms": float, "config": "..."}
Arms may be repeated per model; gates are checked per model (default model =
the one with a NEW row). Exit 0 iff every gate passes.

Usage: python3 scripts/check_perf_gates.py results/matrix.jsonl [--model NAME] [--margin 0.0]
"""
import argparse
import json
import sys
from collections import defaultdict

MARGIN = 0.0  # user-approved relaxation (was 0.02); see GATE RELAXATION note above


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("matrix", nargs="?", default="results/matrix.jsonl")
    ap.add_argument("--model", default=None)
    ap.add_argument("--margin", type=float, default=MARGIN)
    args = ap.parse_args()

    rows = load(args.matrix)
    if not rows:
        print(f"PERF_GATE: FAIL — no rows in {args.matrix}", file=sys.stderr)
        sys.exit(1)

    by_model = defaultdict(dict)
    for r in rows:
        by_model[r.get("model", "default")][r["arm"]] = r

    models = [args.model] if args.model else sorted(by_model)
    all_ok = True
    for model in models:
        arms = by_model.get(model, {})
        print(f"\n=== model={model} arms={sorted(arms)} ===")
        missing = [a for a in ("NEW", "#21631") if a not in arms]
        if missing:
            print(f"  FAIL — missing required arm(s): {missing}")
            all_ok = False
            continue

        def tok(a):
            return arms[a]["output_tok_s"]

        def itl(a):
            return arms[a]["itl_p99_ms"]

        checks = []
        # HARD GATE: strict +margin over #21631
        checks.append((
            f"NEW.tok({tok('NEW'):.1f}) > #21631.tok({tok('#21631'):.1f}) * {1+args.margin:.2f}"
            f" = {tok('#21631')*(1+args.margin):.1f}",
            tok("NEW") > tok("#21631") * (1 + args.margin),
        ))
        if "PR20611" in arms:
            checks.append((
                f"NEW.tok({tok('NEW'):.1f}) > PR20611.tok({tok('PR20611'):.1f})",
                tok("NEW") > tok("PR20611"),
            ))
        if "ceiling" in arms:
            checks.append((
                f"NEW.tok({tok('NEW'):.1f}) >= 0.98 * ceiling.tok({tok('ceiling'):.1f})"
                f" = {0.98*tok('ceiling'):.1f}",
                tok("NEW") >= 0.98 * tok("ceiling"),
            ))
        checks.append((
            f"NEW.itl_p99({itl('NEW'):.2f}) <= #21631.itl_p99({itl('#21631'):.2f})",
            itl("NEW") <= itl("#21631"),
        ))
        if "PR20611" in arms:
            checks.append((
                f"NEW.itl_p99({itl('NEW'):.2f}) <= PR20611.itl_p99({itl('PR20611'):.2f})",
                itl("NEW") <= itl("PR20611"),
            ))
        for desc, ok in checks:
            print(f"  [{'PASS' if ok else 'FAIL'}] {desc}")
            all_ok = all_ok and ok

    if all_ok:
        print("\nPERF_GATE: PASS")
        sys.exit(0)
    print("\nPERF_GATE: FAIL", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
