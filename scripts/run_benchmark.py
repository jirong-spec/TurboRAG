#!/usr/bin/env python3
"""run_benchmark.py — End-to-end TTFT + accuracy comparison.

Modes
─────
  sim        GPU-only TTFT simulation, no model download needed (fast)
  full       Real Qwen2.5-0.5B inference with accuracy measurement
  longbench  LongBench dataset evaluation (TTFT / VRAM / F1)
             Requires: pip install datasets

Examples
────────
  python scripts/run_benchmark.py --mode sim --tokens 512 --layers 24
  python scripts/run_benchmark.py --mode full --store ./kv_store
  python scripts/run_benchmark.py --mode longbench \\
      --dataset qasper \\
      --model Qwen/Qwen2.5-3B-Instruct \\
      --max-samples 20 --max-length 32768
  python scripts/run_benchmark.py --mode longbench \\
      --dataset qasper,2wikimqa \\
      --ttft-only --max-samples 50
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["sim", "full", "longbench"], default="sim")

    # ── sim mode ──────────────────────────────────────────────────── #
    ap.add_argument("--tokens",  type=int, default=512)
    ap.add_argument("--layers",  type=int, default=24)
    ap.add_argument("--warmup",  type=int, default=3)
    ap.add_argument("--iters",   type=int, default=20)

    # ── full mode ─────────────────────────────────────────────────── #
    ap.add_argument("--model",      default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--store",      default="./kv_store")
    ap.add_argument("--corpus",     default=None)
    ap.add_argument("--queries",    default=None)
    ap.add_argument("--schemes",    default="fp16,turbo_prod,turbo_mse,polar")
    ap.add_argument("--new-tokens", type=int, default=64)
    ap.add_argument("--lib",        default=None)

    # ── longbench mode ────────────────────────────────────────────── #
    ap.add_argument(
        "--dataset",
        default="qasper",
        help=(
            "Comma-separated list of LongBench subsets to run.\n"
            "  qasper     single-doc QA on scientific papers (~12K tok)\n"
            "  2wikimqa   multi-hop QA                       (~5K tok)\n"
            "  gov_report long-document summarization        (~8K tok)\n"
            "Example: --dataset qasper,2wikimqa"
        ),
    )
    ap.add_argument(
        "--max-samples", type=int, default=20,
        help="Number of LongBench test items to evaluate (default: 20)",
    )
    ap.add_argument(
        "--max-length", type=int, default=32768,
        help="Context truncation in tokens (Qwen2.5-3B limit: 32768)",
    )
    ap.add_argument(
        "--ttft-only", action="store_true",
        help="Skip answer generation; measure TTFT + VRAM only (no F1)",
    )
    # Note: --model, --store, --schemes, --new-tokens, --lib are shared with full mode

    args = ap.parse_args()

    # ── dispatch ──────────────────────────────────────────────────── #
    if args.mode == "sim":
        from tq_backend.ttft_sim import run_ttft_sim
        run_ttft_sim(
            num_tokens=args.tokens,
            num_layers=args.layers,
            warmup=args.warmup,
            iters=args.iters,
        )

    elif args.mode == "full":
        from tq_backend.benchmark import run_benchmark
        run_benchmark(
            model_name=args.model,
            store_dir=args.store,
            corpus_path=args.corpus,
            queries_path=args.queries,
            schemes=args.schemes.split(","),
            max_new_tokens=args.new_tokens,
            lib_path=args.lib,
        )

    else:  # longbench
        from tq_backend.longbench_eval import run_longbench_benchmark
        datasets = [d.strip() for d in args.dataset.split(",") if d.strip()]
        for ds in datasets:
            print(f"\n{'='*60}")
            print(f"  Dataset: {ds}")
            print(f"{'='*60}")
            run_longbench_benchmark(
                dataset_name=ds,
                model_name=args.model,
                store_dir=args.store,
                schemes=args.schemes.split(","),
                max_samples=args.max_samples,
                max_length=args.max_length,
                max_new_tokens=args.new_tokens,
                ttft_only=args.ttft_only,
                lib_path=args.lib,
            )


if __name__ == "__main__":
    main()
