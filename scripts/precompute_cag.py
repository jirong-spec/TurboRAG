#!/usr/bin/env python3
"""precompute_cag.py — Offline: compress and store KV for a document corpus.

Usage:
  python scripts/precompute_cag.py --corpus data/corpus.jsonl --store ./kv_store
  python scripts/precompute_cag.py --text "Your document text here" --doc-id my_doc
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tq_backend.model_runner import TQModelRunner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",   default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--store",   default="./kv_store")
    ap.add_argument("--corpus",  default=None, help="JSONL file: {id, text} per line")
    ap.add_argument("--text",    default=None, help="Single document text")
    ap.add_argument("--doc-id",  default="doc_0", help="Doc ID when --text is used")
    ap.add_argument("--schemes", default="fp16,turbo_prod,turbo_mse")
    ap.add_argument("--lib",     default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    corpus: dict[str, str] = {}
    if args.corpus:
        with open(args.corpus) as f:
            for line in f:
                rec = json.loads(line)
                corpus[rec["id"]] = rec["text"]
    elif args.text:
        corpus[args.doc_id] = args.text
    else:
        print("Error: provide --corpus or --text", file=sys.stderr)
        sys.exit(1)

    schemes = args.schemes.split(",")
    print(f"Corpus: {len(corpus)} docs  |  Schemes: {schemes}")

    runner = TQModelRunner(args.model, args.store, args.lib)
    runner.precompute_corpus(corpus, schemes=schemes, overwrite=args.overwrite)
    print("Done.")


if __name__ == "__main__":
    main()
