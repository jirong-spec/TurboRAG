#!/usr/bin/env python3
# Claude's version — config-driven RAG benchmark with TurboQuant KV simulation
"""rag_turbo_demo.py

Bring-your-own-data RAG benchmark powered by TurboQuant.
All dataset / model / output settings live in ../config.yaml.

Pipeline
────────
1. Load corpus from KaggleHub or a local CSV (config.yaml → dataset)
2. Build Q&A pairs or load your own (config.yaml → qa)
3. BM25 retrieval over all questions → retrieval recall
4. Ollama LLM inference on a sample   → LLM accuracy
5. TurboQuant KV simulation on every retrieved context
6. Write output/rag_results.md  +  output/rag_results.json

Usage
─────
    python3 rag_turbo_demo.py                   # uses ../config.yaml
    python3 rag_turbo_demo.py --config my.yaml  # custom config path
    python3 rag_turbo_demo.py --llm-sample 100  # override one field

Bring your own data
───────────────────
Edit config.yaml:
  1. Set dataset.source to "local" and dataset.local_path to your CSV, or
     set dataset.kaggle_handle / dataset.kaggle_file for a Kaggle dataset.
  2. Set dataset.text_columns to the column(s) that form the RAG corpus.
  3. Set qa.auto_generate: false and supply your own JSONL at qa.path, or
     configure qa.templates to match your dataset's structured columns.
  4. Adjust llm.model, llm.sample_size, retrieval.top_k as needed.
  5. Run: python3 rag_turbo_demo.py
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

try:
    import requests
except ImportError:
    sys.exit("pip install requests")

# Optional Kaggle support
try:
    import kagglehub
    from kagglehub import KaggleDatasetAdapter
    _KAGGLE_OK = True
except ImportError:
    _KAGGLE_OK = False

try:
    import pandas as pd
except ImportError:
    sys.exit("pip install pandas")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from turboquant_wrapper import TurboQuantWrapper

DEVICE = "cuda"
CHARS_PER_TOK = 3.5


# ─────────────────────────────────────────────────────────────────── #
# Config loading                                                       #
# ─────────────────────────────────────────────────────────────────── #

def load_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)


def _cfg(cfg: dict, *keys, default=None):
    node = cfg
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node


# ─────────────────────────────────────────────────────────────────── #
# Dataset loading                                                      #
# ─────────────────────────────────────────────────────────────────── #

def load_dataframe(cfg: dict) -> "pd.DataFrame":
    source = _cfg(cfg, "dataset", "source", default="kaggle")
    nrows  = _cfg(cfg, "dataset", "nrows",  default=5000)

    if source == "local":
        local_path = _cfg(cfg, "dataset", "local_path")
        if not local_path:
            sys.exit("config.yaml: dataset.local_path must be set when source=local")
        p = Path(local_path)
        if not p.is_absolute():
            p = ROOT / p
        if not p.exists():
            sys.exit(f"Local dataset not found: {p}")
        print(f"Loading   : {p} (nrows={nrows})")
        return pd.read_csv(p, nrows=nrows)

    # Kaggle
    if not _KAGGLE_OK:
        sys.exit("pip install kagglehub  (required for source=kaggle)")
    handle = _cfg(cfg, "dataset", "kaggle_handle")
    file   = _cfg(cfg, "dataset", "kaggle_file")
    if not handle:
        sys.exit("config.yaml: dataset.kaggle_handle must be set")
    print(f"Fetching  : {handle} / {file} …")
    return kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        handle,
        file or "",
        pandas_kwargs={"nrows": nrows},
    )


def build_corpus(df: "pd.DataFrame", text_cols: list[str]) -> list[str]:
    available = [c for c in text_cols if c in df.columns]
    if not available:
        sys.exit(f"None of text_columns {text_cols!r} found. Available: {df.columns.tolist()}")
    combined = df[available[0]].fillna("").astype(str)
    for col in available[1:]:
        combined = combined + ". " + df[col].fillna("").astype(str)
    return combined.str.strip().tolist()


# ─────────────────────────────────────────────────────────────────── #
# QA generation / loading                                              #
# ─────────────────────────────────────────────────────────────────── #

def generate_qa_pairs(
    df: "pd.DataFrame",
    name_col: str,
    templates: dict[str, str],
    n: int = 5_000,
) -> list[dict]:
    pairs: list[dict] = []
    for _, row in df.iterrows():
        name = row.get(name_col, "")
        if not name or pd.isna(name):
            continue
        for field, tmpl in templates.items():
            val = row.get(field)
            if pd.notna(val) and str(val).strip() not in ("nan", ""):
                pairs.append({
                    "question":    tmpl.format(name=str(name).strip()),
                    "answer":      str(val).strip(),
                    "source_name": str(name).strip(),
                    "field":       field,
                })
        if len(pairs) >= n:
            break
    return pairs[:n]


def load_or_generate_qa(df: "pd.DataFrame", cfg: dict) -> list[dict]:
    qa_path = ROOT / _cfg(cfg, "qa", "path", default="data/qa_pairs.jsonl")
    auto    = _cfg(cfg, "qa", "auto_generate", default=True)

    if qa_path.exists():
        print(f"QA file   : {qa_path} (cached)")
        with qa_path.open() as f:
            return [json.loads(l) for l in f if l.strip()]

    if not auto:
        sys.exit(
            f"qa.auto_generate is false but {qa_path} does not exist.\n"
            "Either set auto_generate: true or provide the JSONL file."
        )

    name_col   = _cfg(cfg, "qa", "name_column", default="name")
    templates  = _cfg(cfg, "qa", "templates", default={})
    nrows      = _cfg(cfg, "dataset", "nrows", default=5000)

    if not templates:
        sys.exit(
            "config.yaml: qa.templates is empty. "
            "Define at least one field→template mapping."
        )
    if name_col not in df.columns:
        sys.exit(
            f"config.yaml: qa.name_column='{name_col}' not found. "
            f"Available: {df.columns.tolist()}"
        )

    print(f"Generating QA pairs → {qa_path} …")
    pairs = generate_qa_pairs(df, name_col, templates, n=nrows)
    qa_path.parent.mkdir(parents=True, exist_ok=True)
    with qa_path.open("w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"QA file   : saved {len(pairs)} pairs")
    return pairs


# ─────────────────────────────────────────────────────────────────── #
# BM25                                                                 #
# ─────────────────────────────────────────────────────────────────── #

def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    tokens.extend(re.findall(r'[\d,.]+\s*[%]?|[A-Za-z]+', text))
    cjk = re.sub(r'[^\u4e00-\u9fff]', '', text)
    tokens.extend(cjk[i:i + 2] for i in range(len(cjk) - 1))
    return tokens


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.docs = docs
        self.k1, self.b = k1, b
        self.N   = len(docs)
        self.df: dict[str, int]     = defaultdict(int)
        self.tf: list[Counter[str]] = []
        self.dl: list[int]          = []
        for doc in docs:
            toks = _tokenize(doc)
            self.dl.append(len(toks))
            c = Counter(toks)
            self.tf.append(c)
            for t in c:
                self.df[t] += 1
        self.avgdl = sum(self.dl) / max(self.N, 1)

    def _score(self, qtoks: list[str], idx: int) -> float:
        tf, dl = self.tf[idx], self.dl[idx]
        s = 0.0
        for t in qtoks:
            if t not in self.df:
                continue
            idf = math.log((self.N - self.df[t] + 0.5) / (self.df[t] + 0.5) + 1)
            tfn = tf[t] * (self.k1 + 1) / (
                tf[t] + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            s += idf * tfn
        return s

    def retrieve(self, query: str, k: int = 5) -> list[tuple[int, float, str]]:
        qtoks = _tokenize(query)
        scored = sorted(
            ((i, self._score(qtoks, i)) for i in range(self.N)),
            key=lambda x: -x[1],
        )
        return [(i, s, self.docs[i]) for i, s in scored[:k] if s > 0]


# ─────────────────────────────────────────────────────────────────── #
# Ollama                                                               #
# ─────────────────────────────────────────────────────────────────── #

def ollama_available(url: str, model: str) -> bool:
    try:
        r = requests.get(f"{url}/api/tags", timeout=5)
        names = [m["name"] for m in r.json().get("models", [])]
        return any(model.split(":")[0] in n for n in names)
    except Exception:
        return False


def ollama_generate(prompt: str, url: str, model: str, timeout: int) -> tuple[str, float]:
    try:
        t0 = time.perf_counter()
        r  = requests.post(
            f"{url}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=timeout,
        )
        r.raise_for_status()
        ms = (time.perf_counter() - t0) * 1000
        return r.json().get("response", "").strip(), ms
    except Exception as exc:
        return f"[unavailable: {exc}]", 0.0


def build_prompt(system: str, question: str, context: str) -> str:
    return f"{system}\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"


# ─────────────────────────────────────────────────────────────────── #
# TurboQuant KV simulation                                            #
# ─────────────────────────────────────────────────────────────────── #

@dataclass
class KVResult:
    scheme: str; num_tokens: int
    fp16_mb: float; quant_mb: float; compression: float
    pack_us: float; kv_mse: float


def _bench(fn, warmup=3, iters=20) -> float:
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


def simulate_kv(tq: TurboQuantWrapper, num_tokens: int) -> list[KVResult]:
    cfg = tq.default_config()
    H, D = cfg.num_kv_heads, cfg.head_dim
    k = torch.randn(num_tokens, H, D, device=DEVICE, dtype=torch.float16).contiguous()
    v = torch.randn(num_tokens, H, D, device=DEVICE, dtype=torch.float16).contiguous()
    s = torch.arange(num_tokens, dtype=torch.int32, device=DEVICE)
    fp16_b = tq.fp16_bytes(num_tokens, cfg)
    results: list[KVResult] = []

    lay = tq.make_layout_for(cfg)
    pool = tq.alloc_page_pool(num_tokens, lay, cfg)
    ok = torch.empty_like(k); ov = torch.empty_like(v)
    pu = _bench(lambda: tq.pack(k, v, s, pool, lay, cfg))
    tq.pack(k, v, s, pool, lay, cfg); torch.cuda.synchronize()
    tq.dequant(pool, s, ok, ov, lay, cfg); torch.cuda.synchronize()
    qb = tq.quant_bytes(num_tokens, lay, cfg)
    results.append(KVResult("turbo_prod", num_tokens, fp16_b/1024**2, qb/1024**2,
                             fp16_b/qb, pu,
                             TurboQuantWrapper.compute_mse(torch.cat([ok,ov]), torch.cat([k,v]))))
    try:
        ml = tq.make_mse_layout_for(cfg)
        mp = tq.alloc_mse_pool(num_tokens, ml, cfg)
        mk = torch.empty_like(k); mv = torch.empty_like(v)
        mpu = _bench(lambda: tq.mse_pack(k, v, s, mp, ml, cfg))
        tq.mse_pack(k, v, s, mp, ml, cfg); torch.cuda.synchronize()
        tq.mse_dequant(mp, s, mk, mv, ml, cfg); torch.cuda.synchronize()
        mb = tq.mse_bytes(num_tokens, ml, cfg)
        results.append(KVResult("turbo_mse", num_tokens, fp16_b/1024**2, mb/1024**2,
                                 fp16_b/mb, mpu,
                                 TurboQuantWrapper.compute_mse(torch.cat([mk,mv]), torch.cat([k,v]))))
    except Exception as exc:
        warnings.warn(f"turbo_mse skipped: {exc}", stacklevel=2)
    return results


# ─────────────────────────────────────────────────────────────────── #
# Report                                                               #
# ─────────────────────────────────────────────────────────────────── #

def _avg(lst: list[KVResult], attr: str) -> float:
    return sum(getattr(r, attr) for r in lst) / len(lst) if lst else float("nan")


def write_report(
    cfg: dict,
    qa_pairs: list[dict],
    retrieval_hits: int,
    llm_rows: list[dict],
    kv_all: list[KVResult],
) -> None:
    out_dir   = ROOT / _cfg(cfg, "output", "dir",         default="output")
    stem      = _cfg(cfg, "output", "report_stem", default="rag_results")
    top_k     = _cfg(cfg, "retrieval", "top_k",    default=5)
    model     = _cfg(cfg, "llm", "model",          default="")
    handle    = _cfg(cfg, "dataset", "kaggle_handle", default="local")
    out_dir.mkdir(parents=True, exist_ok=True)

    total     = len(qa_pairs)
    recall    = retrieval_hits / total if total else 0
    llm_acc   = (sum(1 for r in llm_rows if r["correct"]) / len(llm_rows)
                 if llm_rows else None)

    prod_lst  = [r for r in kv_all if r.scheme == "turbo_prod"]
    mse_lst   = [r for r in kv_all if r.scheme == "turbo_mse"]

    lines = [
        "# TurboRAG — RAG Benchmark Report\n",
        f"- Dataset      : `{handle}`",
        f"- Corpus size  : {total} documents",
        f"- Questions    : {total}",
        f"- LLM model    : {model}",
        f"- BM25 top-k   : {top_k}",
        "",
        "---",
        "",
        "## Retrieval Recall (BM25)\n",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total questions | {total} |",
        f"| Answer found in top-{top_k} chunks | {retrieval_hits} |",
        f"| **Retrieval recall** | **{recall:.1%}** |",
        "",
        "---",
        "",
    ]

    if llm_rows:
        correct_n = sum(1 for r in llm_rows if r["correct"])
        avg_lat   = sum(r["lat_ms"] for r in llm_rows) / len(llm_rows)
        lines += [
            f"## LLM Accuracy ({model}, {len(llm_rows)}-question sample)\n",
            "| # | Question | Expected | Correct | Latency ms |",
            "|---|----------|----------|---------|------------|",
        ]
        for r in llm_rows:
            tick = "✓" if r["correct"] else "✗"
            lines.append(
                f"| {r['qi']} | {r['question'][:60]} | {r['expected'][:30]} "
                f"| {tick} | {r['lat_ms']:.0f} |"
            )
        lines += [
            "",
            f"**LLM accuracy: {correct_n}/{len(llm_rows)} = {llm_acc:.1%}**  "
            f"(avg latency {avg_lat:.0f} ms)",
            "",
            "---",
            "",
        ]

    lines += [
        "## KV-Cache Efficiency (TurboQuant, averaged over all questions)\n",
        "| Scheme | Avg Tokens | FP16 MB | Quant MB | Compression | Pack µs | KV MSE |",
        "|--------|-----------|---------|---------|-------------|---------|--------|",
    ]
    for scheme, lst in [("turbo_prod", prod_lst), ("turbo_mse", mse_lst)]:
        if not lst:
            continue
        lines.append(
            f"| {scheme} | {_avg(lst,'num_tokens'):.0f} "
            f"| {_avg(lst,'fp16_mb'):.3f} "
            f"| {_avg(lst,'quant_mb'):.3f} "
            f"| {_avg(lst,'compression'):.2f}× "
            f"| {_avg(lst,'pack_us'):.1f} "
            f"| {_avg(lst,'kv_mse'):.3e} |"
        )
    lines.append("| FP16 baseline | — | — | — | 1.00× | — | 0 |")
    lines += [
        "",
        "---",
        "",
        "## Summary\n",
        f"- BM25 retrieval recall   : **{recall:.1%}** ({retrieval_hits}/{total})",
    ]
    if llm_acc is not None:
        lines.append(
            f"- LLM answer accuracy    : **{llm_acc:.1%}** ({correct_n}/{len(llm_rows)} sample)"
        )
    for scheme, lst in [("turbo_prod", prod_lst), ("turbo_mse", mse_lst)]:
        c = _avg(lst, "compression")
        if not math.isnan(c):
            lines.append(f"- {scheme} compression  : **{c:.2f}×** VRAM vs FP16")

    md_path = out_dir / f"{stem}.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    json_path = out_dir / f"{stem}.json"
    json_path.write_text(json.dumps({
        "retrieval_recall": recall,
        "llm_accuracy":     llm_acc,
        "llm_rows":         llm_rows,
        "kv_summary": {
            s: {"avg_compression": _avg(l,"compression"),
                "avg_pack_us":     _avg(l,"pack_us"),
                "avg_kv_mse":      _avg(l,"kv_mse")}
            for s, l in [("turbo_prod", prod_lst), ("turbo_mse", mse_lst)]
        },
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nReport    : {md_path}")
    print(f"JSON      : {json_path}")
    print("\n" + "─" * 60)
    print("\n".join(lines[-10:]))


# ─────────────────────────────────────────────────────────────────── #
# Main                                                                 #
# ─────────────────────────────────────────────────────────────────── #

def run(cfg: dict, llm_sample_override: int | None) -> None:
    # ── Load corpus ─────────────────────────────────────────────── #
    df        = load_dataframe(cfg)
    text_cols = _cfg(cfg, "dataset", "text_columns", default=["description"])
    corpus    = build_corpus(df, text_cols)
    print(f"Corpus    : {len(df)} rows, {len(corpus)} documents")

    # ── QA pairs ────────────────────────────────────────────────── #
    qa_pairs  = load_or_generate_qa(df, cfg)
    llm_sample = llm_sample_override or _cfg(cfg, "llm", "sample_size", default=50)
    print(f"Questions : {len(qa_pairs)} total  (LLM sample: {llm_sample})\n")

    # ── BM25 ────────────────────────────────────────────────────── #
    print("Building BM25 index …")
    bm25  = BM25(corpus)
    top_k = _cfg(cfg, "retrieval", "top_k", default=5)

    # ── GPU + TurboQuant ────────────────────────────────────────── #
    if not torch.cuda.is_available():
        sys.exit("CUDA GPU required for TurboQuant simulation.")
    print(f"GPU       : {torch.cuda.get_device_name(0)}")
    tq = TurboQuantWrapper()

    # ── Ollama ──────────────────────────────────────────────────── #
    ollama_url = _cfg(cfg, "llm", "ollama_url",  default="http://localhost:11434")
    model      = _cfg(cfg, "llm", "model",       default="qwen2.5:7b")
    timeout    = _cfg(cfg, "llm", "timeout_sec", default=120)
    system     = _cfg(cfg, "llm", "system_prompt",
                      default="You are a helpful assistant. Answer using ONLY the context.")
    ollama_ok  = ollama_available(ollama_url, model)
    print(f"Ollama    : {'✓ ' + model if ollama_ok else '✗ not available'}\n")

    # ── Main loop ───────────────────────────────────────────────── #
    print(f"Running retrieval + KV simulation on {len(qa_pairs)} questions …")
    retrieval_hits = 0
    kv_all:   list[KVResult] = []
    llm_rows: list[dict]     = []

    for qi, qa in enumerate(qa_pairs):
        hits     = bm25.retrieve(qa["question"], k=top_k)
        recalled = qa["answer"].lower() in " ".join(d.lower() for _, _, d in hits)
        if recalled:
            retrieval_hits += 1

        context  = "\n\n".join(doc for _, _, doc in hits)
        n_tokens = max(64, int(len(context) / CHARS_PER_TOK))
        kv_all.extend(simulate_kv(tq, n_tokens))

        if ollama_ok and qi < llm_sample:
            answer, lat_ms = ollama_generate(
                build_prompt(system, qa["question"], context),
                ollama_url, model, timeout,
            )
            correct = qa["answer"].lower() in answer.lower()
            llm_rows.append({
                "qi": qi + 1, "question": qa["question"],
                "expected": qa["answer"],
                "answer": answer[:200].replace("\n", " "),
                "correct": correct, "lat_ms": lat_ms,
            })
            if (qi + 1) % 10 == 0:
                print(f"  LLM {qi+1}/{llm_sample} done …")

        if (qi + 1) % 500 == 0:
            pct = retrieval_hits / (qi + 1) * 100
            print(f"  [{qi+1:>5}/{len(qa_pairs)}] retrieval recall so far: {pct:.1f}%")

    write_report(cfg, qa_pairs, retrieval_hits, llm_rows, kv_all)


def main() -> None:
    p = argparse.ArgumentParser(description="TurboRAG benchmark — edit config.yaml to use your own data")
    p.add_argument("--config",     default=str(ROOT / "config.yaml"), help="Path to config.yaml")
    p.add_argument("--llm-sample", type=int, default=None, help="Override llm.sample_size")
    args = p.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        sys.exit(f"Config not found: {cfg_path}")
    cfg = load_config(cfg_path)
    run(cfg, args.llm_sample)


if __name__ == "__main__":
    main()
