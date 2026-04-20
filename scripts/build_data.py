#!/usr/bin/env python3
"""build_data.py — Build a CAG corpus from arbitrary documents and (optionally) precompute KV caches.

Supported input formats: .txt, .md, .jsonl ({"id","text"} or {"doc_id","text"}), .csv (id,text columns), .pdf
Chunking strategies: fixed (token count), sentence, paragraph

Usage examples
──────────────
# Chunk a folder of text files → corpus.jsonl only
python scripts/build_data.py --input-dir data/docs/ --output-dir data/ --corpus-only

# Full pipeline: chunk + compress KV for all schemes
python scripts/build_data.py \
    --input-dir data/docs/ \
    --output-dir data/ \
    --store ./kv_store \
    --model qwen2.5-0.5b \
    --quant-type turbo_prod,turbo_mse

# JSONL input with sentence chunking
python scripts/build_data.py \
    --input-dir data/ \
    --formats jsonl \
    --chunking sentence \
    --output-dir data/ \
    --corpus-only
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Iterator

# ── optional deps ─────────────────────────────────────────────────────────── #

try:
    from tqdm import tqdm as _tqdm
    def _progress(iterable, **kw):
        return _tqdm(iterable, **kw)
except ImportError:
    def _progress(iterable, **kw):
        desc = kw.get("desc", "")
        items = list(iterable)
        if desc:
            print(desc)
        return items

try:
    import pypdf
    _HAS_PYPDF = True
except ImportError:
    _HAS_PYPDF = False

# ── model shorthand resolver ───────────────────────────────────────────────── #

_MODEL_SHORTHANDS: dict[str, str] = {
    "qwen2.5-0.5b":  "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen2.5-1.5b":  "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2.5-3b":    "Qwen/Qwen2.5-3B-Instruct",
    "qwen2.5-7b":    "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-14b":   "Qwen/Qwen2.5-14B-Instruct",
    "llama3.2-1b":   "meta-llama/Llama-3.2-1B-Instruct",
    "llama3.2-3b":   "meta-llama/Llama-3.2-3B-Instruct",
    "llama3.1-8b":   "meta-llama/Llama-3.1-8B-Instruct",
    "mistral-7b":    "mistralai/Mistral-7B-Instruct-v0.3",
    "phi3-mini":     "microsoft/Phi-3-mini-4k-instruct",
    "gemma2-2b":     "google/gemma-2-2b-it",
    "gemma2-9b":     "google/gemma-2-9b-it",
}

def _resolve_model(name: str) -> str:
    return _MODEL_SHORTHANDS.get(name.lower(), name)


# ── loaders ───────────────────────────────────────────────────────────────── #

def _load_txt(path: Path) -> list[tuple[str, str]]:
    return [(path.stem, path.read_text(encoding="utf-8", errors="replace"))]


def _load_md(path: Path) -> list[tuple[str, str]]:
    return [(path.stem, path.read_text(encoding="utf-8", errors="replace"))]


def _load_jsonl(path: Path) -> list[tuple[str, str]]:
    records = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            doc_id = obj.get("id") or obj.get("doc_id") or f"{path.stem}_{i}"
            text   = obj.get("text") or obj.get("content") or ""
            if text:
                records.append((str(doc_id), text))
    return records


def _load_csv(path: Path) -> list[tuple[str, str]]:
    records = []
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            doc_id = row.get("id") or row.get("doc_id") or f"{path.stem}_{i}"
            text   = row.get("text") or row.get("content") or ""
            if text:
                records.append((str(doc_id), text))
    return records


def _load_pdf(path: Path) -> list[tuple[str, str]]:
    if not _HAS_PYPDF:
        print(f"  [SKIP] {path.name}: pypdf not installed (pip install pypdf)")
        return []
    reader = pypdf.PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n\n".join(p for p in pages if p.strip())
    return [(path.stem, text)] if text.strip() else []


_LOADERS = {
    ".txt":  _load_txt,
    ".md":   _load_md,
    ".jsonl": _load_jsonl,
    ".csv":  _load_csv,
    ".pdf":  _load_pdf,
}

# ── chunkers ──────────────────────────────────────────────────────────────── #

def _chunk_fixed(text: str, chunk_size: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    step = max(1, chunk_size - overlap)
    chunks = []
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    return chunks


def _chunk_sentence(text: str, chunk_size: int, overlap: int) -> list[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return []
    chunks, buf, buf_words = [], [], 0
    for sent in sentences:
        w = len(sent.split())
        if buf and buf_words + w > chunk_size:
            chunks.append(" ".join(buf))
            # keep overlap sentences
            kept, kept_words = [], 0
            for s in reversed(buf):
                sw = len(s.split())
                if kept_words + sw <= overlap:
                    kept.insert(0, s)
                    kept_words += sw
                else:
                    break
            buf, buf_words = kept, kept_words
        buf.append(sent)
        buf_words += w
    if buf:
        chunks.append(" ".join(buf))
    return chunks


def _chunk_paragraph(text: str, chunk_size: int, overlap: int) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    if not paragraphs:
        return _chunk_fixed(text, chunk_size, overlap)
    chunks, buf, buf_words = [], [], 0
    for para in paragraphs:
        w = len(para.split())
        if buf and buf_words + w > chunk_size:
            chunks.append("\n\n".join(buf))
            buf, buf_words = [], 0
        buf.append(para)
        buf_words += w
    if buf:
        chunks.append("\n\n".join(buf))
    return chunks


_CHUNKERS = {
    "fixed":     _chunk_fixed,
    "sentence":  _chunk_sentence,
    "paragraph": _chunk_paragraph,
}

# ── doc-id helpers ────────────────────────────────────────────────────────── #

def _safe_stem(name: str) -> str:
    return re.sub(r'[^\w-]', '_', name)[:64]


def _make_doc_id(base_id: str, chunk_idx: int, total: int) -> str:
    stem = _safe_stem(base_id)
    return stem if total == 1 else f"{stem}_c{chunk_idx:03d}"


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


# ── scan ──────────────────────────────────────────────────────────────────── #

def scan_input_dir(
    input_dir: Path,
    formats: list[str],
) -> Iterator[tuple[str, str]]:
    """Yield (base_doc_id, text) for every loadable file in input_dir."""
    exts = {f if f.startswith(".") else f".{f}" for f in formats}
    files = sorted(p for p in input_dir.rglob("*") if p.suffix.lower() in exts)
    if not files:
        print(f"No files with extensions {exts} found in {input_dir}", file=sys.stderr)
        return
    for fpath in files:
        loader = _LOADERS.get(fpath.suffix.lower())
        if loader is None:
            continue
        try:
            records = loader(fpath)
        except Exception as exc:
            print(f"  [WARN] {fpath.name}: {exc}", file=sys.stderr)
            continue
        for doc_id, text in records:
            if text.strip():
                yield doc_id, text


# ── build ─────────────────────────────────────────────────────────────────── #

def build_data(
    input_dir: Path,
    output_dir: Path,
    formats: list[str],
    chunking: str,
    chunk_size: int,
    overlap: int,
    store_dir: Path | None,
    model: str | None,
    quant_schemes: list[str],
    lib_path: str | None,
    overwrite: bool,
    corpus_only: bool,
    max_docs: int | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    chunker = _CHUNKERS[chunking]

    # ── phase 1: scan + chunk → corpus dict ── #
    corpus: dict[str, str] = {}
    seen_ids: set[str] = set()

    raw_records = list(scan_input_dir(input_dir, formats))
    if not raw_records:
        print("No documents found. Exiting.")
        return

    print(f"Loaded {len(raw_records)} raw document(s). Chunking with strategy={chunking!r} ...")

    for base_id, text in raw_records:
        chunks = chunker(text, chunk_size, overlap)
        if not chunks:
            continue
        for i, chunk in enumerate(chunks):
            doc_id = _make_doc_id(base_id, i, len(chunks))
            # deduplicate: append content hash suffix if id already taken
            if doc_id in seen_ids:
                doc_id = f"{doc_id}_{_content_hash(chunk)}"
            seen_ids.add(doc_id)
            corpus[doc_id] = chunk
        if max_docs and len(corpus) >= max_docs:
            break

    if max_docs:
        corpus = dict(list(corpus.items())[:max_docs])

    print(f"Produced {len(corpus)} chunk(s).")

    # ── phase 2: write corpus.jsonl ── #
    corpus_path = output_dir / "corpus.jsonl"
    with corpus_path.open("w", encoding="utf-8") as f:
        for doc_id, text in corpus.items():
            f.write(json.dumps({"id": doc_id, "text": text}, ensure_ascii=False) + "\n")
    print(f"Wrote corpus → {corpus_path}  ({len(corpus)} docs)")

    if corpus_only:
        _write_build_manifest(output_dir, corpus, schemes=[], elapsed=0.0)
        return

    # ── phase 3: load model + precompute KV ── #
    if not model:
        print("No --model specified; skipping KV precomputation. Use --corpus-only or add --model.")
        return
    if not store_dir:
        print("No --store specified; skipping KV precomputation.")
        return

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tq_backend.model_runner import TQModelRunner

    resolved = _resolve_model(model)
    print(f"Loading model: {resolved}")
    t0 = time.time()
    runner = TQModelRunner(resolved, str(store_dir), lib_path)
    runner.precompute_corpus(corpus, schemes=quant_schemes, overwrite=overwrite)
    elapsed = time.time() - t0
    print(f"KV precomputation done in {elapsed:.1f}s")

    _write_build_manifest(output_dir, corpus, schemes=quant_schemes, elapsed=elapsed)


def _write_build_manifest(
    output_dir: Path,
    corpus: dict[str, str],
    schemes: list[str],
    elapsed: float,
) -> None:
    manifest = {
        "num_docs": len(corpus),
        "doc_ids":  list(corpus.keys()),
        "schemes":  schemes,
        "elapsed_s": round(elapsed, 2),
        "built_at":  time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    p = output_dir / "build_manifest.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2))
    tmp.replace(p)
    print(f"Wrote build manifest → {p}")


# ── CLI ───────────────────────────────────────────────────────────────────── #

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input-dir",  required=True,
                    help="Directory to scan for documents")
    ap.add_argument("--output-dir", default="data/",
                    help="Where to write corpus.jsonl and build_manifest.json")
    ap.add_argument("--formats",    default="txt,md,jsonl,csv,pdf",
                    help="Comma-separated file extensions to load (default: txt,md,jsonl,csv,pdf)")
    ap.add_argument("--chunk-size", type=int, default=512,
                    help="Target chunk size in words (default: 512)")
    ap.add_argument("--overlap",    type=int, default=64,
                    help="Overlap in words between consecutive chunks (default: 64)")
    ap.add_argument("--chunking",   default="fixed",
                    choices=list(_CHUNKERS),
                    help="Chunking strategy: fixed | sentence | paragraph (default: fixed)")
    ap.add_argument("--model",      default=None,
                    help="Model name or shorthand (e.g. qwen2.5-0.5b). Skips KV step if omitted.")
    ap.add_argument("--store",      default=None,
                    help="KV store directory for precomputed caches")
    ap.add_argument("--quant-type", default="fp16,turbo_prod",
                    help="Comma-separated quantization schemes (default: fp16,turbo_prod)")
    ap.add_argument("--lib",        default=None,
                    help="Path to libturboquant.so (auto-detected if omitted)")
    ap.add_argument("--overwrite",  action="store_true",
                    help="Re-compress even if KV already exists in the store")
    ap.add_argument("--corpus-only", action="store_true",
                    help="Write corpus.jsonl only; skip model load and KV precomputation")
    ap.add_argument("--max-docs",   type=int, default=None,
                    help="Limit total number of chunks written to corpus")
    args = ap.parse_args()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    store_dir  = Path(args.store) if args.store else None

    if not input_dir.is_dir():
        ap.error(f"--input-dir not found: {input_dir}")

    formats = [f.strip().lstrip(".") for f in args.formats.split(",")]
    quant_schemes = [s.strip() for s in args.quant_type.split(",")]

    build_data(
        input_dir    = input_dir,
        output_dir   = output_dir,
        formats      = formats,
        chunking     = args.chunking,
        chunk_size   = args.chunk_size,
        overlap      = args.overlap,
        store_dir    = store_dir,
        model        = args.model,
        quant_schemes = quant_schemes,
        lib_path     = args.lib,
        overwrite    = args.overwrite,
        corpus_only  = args.corpus_only,
        max_docs     = args.max_docs,
    )


if __name__ == "__main__":
    main()
