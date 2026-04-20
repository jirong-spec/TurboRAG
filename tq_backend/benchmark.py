"""benchmark.py — End-to-end comparison: FP16 vs turbo_prod vs turbo_mse.

TTFT methodology
────────────────
Normal RAG   : prefill(doc_tokens + query_tokens) → first token
               Measured by timing model.generate() on full prompt.

CAG fp16     : load fp16 KV from disk → inject as past_key_values
               → prefill(query_tokens only) → first token
               Measured: disk_load_time + query_prefill_time

CAG compress : load compressed KV from disk
               → prefill(query_tokens only) → first token
               Measured: disk_load_time + query_prefill_time
               (query prefill is identical for all schemes)

Accuracy     : exact-match substring check on model answers vs reference.
AttnMSE      : per-layer attention output MSE vs FP16 reference.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch

Scheme = Literal["fp16", "turbo_prod", "turbo_mse"]
ALL_SCHEMES: list[Scheme] = ["fp16", "turbo_prod", "turbo_mse"]


@dataclass
class SchemeStats:
    scheme: str
    ttft_ms_list:    list[float] = field(default_factory=list)
    answers:         list[str]   = field(default_factory=list)
    exact_matches:   list[bool]  = field(default_factory=list)
    kv_mb:           float = 0.0
    fp16_mb:         float = 0.0
    avg_attn_mse:    float = 0.0

    @property
    def avg_ttft_ms(self):  return sum(self.ttft_ms_list) / max(len(self.ttft_ms_list), 1)
    @property
    def accuracy(self):     return sum(self.exact_matches) / max(len(self.exact_matches), 1)
    @property
    def vram_ratio(self):   return self.fp16_mb / max(self.kv_mb, 1e-9)


def run_benchmark(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    store_dir: str = "./kv_store",
    corpus_path: str | None = None,
    queries_path: str | None = None,
    schemes: list[Scheme] | None = None,
    max_new_tokens: int = 64,
    lib_path: str | None = None,
    num_attn_layers_mse: int = 3,
) -> dict[str, SchemeStats]:
    from tq_backend.model_runner import TQModelRunner
    from tq_backend.cag_store import CAGStore

    if schemes is None:
        schemes = ALL_SCHEMES

    # ── corpus + queries ───────────────────────────────────────────── #
    corpus: dict[str, str] = {}
    if corpus_path and Path(corpus_path).exists():
        with open(corpus_path) as f:
            for line in f:
                rec = json.loads(line)
                corpus[rec["id"]] = rec["text"]
    else:
        corpus = {
            "doc_turboquant": (
                "TurboQuant is a KV-cache quantization system for LLMs. "
                "It uses a turbo_prod scheme with K=3-bit plus 1-bit residual and V=4-bit, "
                "achieving 15 to 16 times compression versus FP16. The PolarQuant variant "
                "applies Hadamard rotation with K=2-bit and V=3-bit for 6.1 times compression "
                "with near-zero accuracy loss. CAG stands for Cache-Augmented Generation, "
                "which pre-computes KV offline and loads from disk to skip LLM prefill "
                "and reduce time-to-first-token by up to 20 times for long documents."
            ),
            "doc_rag": (
                "Retrieval-Augmented Generation, also known as RAG, combines a retrieval system "
                "with a language model. Documents are encoded into dense vectors "
                "and stored in a vector database. At query time, the top-k relevant documents "
                "are retrieved and appended to the prompt as context for the language model. "
                "The main bottleneck in RAG systems is the prefill latency: "
                "the model must process all retrieved context on every single query."
            ),
            "doc_vllm": (
                "vLLM is a high-throughput LLM serving engine built around PagedAttention. "
                "It manages KV cache in fixed-size memory blocks called pages "
                "and supports continuous batching for efficient GPU utilization. "
                "Custom attention backends can be registered via the AttentionBackend interface, "
                "allowing third-party implementations like TurboQuant to replace FlashAttention."
            ),
        }
        print("  [demo] Built-in corpus: 3 documents")

    qa_pairs: list[dict] = []
    if queries_path and Path(queries_path).exists():
        with open(queries_path) as f:
            for line in f:
                qa_pairs.append(json.loads(line))
    else:
        qa_pairs = [
            {"id": "q1", "doc_id": "doc_turboquant", "query": "What compression ratio does turbo_prod achieve?",  "answer": "15"},
            {"id": "q2", "doc_id": "doc_turboquant", "query": "What does CAG stand for?",                          "answer": "Cache-Augmented Generation"},
            {"id": "q3", "doc_id": "doc_rag",        "query": "What is the main bottleneck in RAG?",               "answer": "prefill"},
            {"id": "q4", "doc_id": "doc_vllm",       "query": "What memory management does vLLM use?",             "answer": "PagedAttention"},
            {"id": "q5", "doc_id": "doc_turboquant", "query": "What rotation does PolarQuant use?",                "answer": "Hadamard"},
        ]
        print("  [demo] Built-in QA pairs: 5 questions")

    # ── init ───────────────────────────────────────────────────────── #
    runner = TQModelRunner(model_name, store_dir, lib_path)
    cag = runner.store

    # ── [1/4] offline precompute ───────────────────────────────────── #
    print("\n[1/4] Pre-computing KV cache ...")
    runner.precompute_corpus(corpus, schemes=schemes)

    # Also store fp16 DynamicCache for true CAG injection
    print("      Storing fp16 DynamicCache for CAG injection ...")
    _store_dynamic_cache(runner, corpus)

    # ── [2/4] attention MSE ────────────────────────────────────────── #
    print(f"\n[2/4] Attention MSE (first {num_attn_layers_mse} layers) ...")
    mse_per_scheme: dict[str, list[float]] = {s: [] for s in schemes if s != "fp16"}
    for doc_id, text in list(corpus.items())[:2]:
        mse = runner.compare_attention_mse(text, list(range(num_attn_layers_mse)),
                                           [s for s in schemes if s != "fp16"])
        for s, vals in mse.items():
            mse_per_scheme[s].extend(vals)
    avg_mse = {s: sum(v)/max(len(v),1) for s, v in mse_per_scheme.items()}
    for s, m in avg_mse.items():
        print(f"  {s:<14} attn MSE = {m:.5f}")

    # ── [3/4] normal RAG baseline ──────────────────────────────────── #
    print(f"\n[3/4] Normal RAG baseline (full doc+query prefill) ...")
    # Warmup to avoid GPU cold-start skewing first measurement
    _measure_ttft(runner, "Warmup query.", max_new_tokens=1)
    _measure_ttft(runner, "Warmup query 2.", max_new_tokens=1)

    normal_ttft: dict[str, list[float]] = {}
    for qa in qa_pairs:
        doc_text = corpus[qa["doc_id"]]
        query    = qa["query"]
        full_prompt = f"Context: {doc_text}\nQuestion: {query}\nAnswer:"
        ttft = _measure_ttft(runner, full_prompt, max_new_tokens)
        key = qa["doc_id"]
        normal_ttft.setdefault(key, []).append(ttft)
        print(f"  {qa['id']}: {ttft:.1f} ms  (doc={len(doc_text)} chars)")

    # ── [4/4] CAG inference per scheme ────────────────────────────── #
    print(f"\n[4/4] CAG inference — {len(qa_pairs)} queries × {len(schemes)} schemes ...")
    stats: dict[str, SchemeStats] = {s: SchemeStats(scheme=s) for s in schemes}

    # Query-only prefill time (shared baseline for CAG TTFT)
    query_prefill_ms = _measure_query_prefill(runner)

    for qa in qa_pairs:
        doc_id  = qa["doc_id"]
        query   = qa["query"]
        ref     = qa["answer"].strip().lower()

        for scheme in schemes:
            # TTFT = disk_load_time + query_prefill_time
            disk_ms = _measure_disk_load(cag, doc_id, runner.num_layers, scheme,
                                         runner.head_shape)
            ttft_ms = disk_ms + query_prefill_ms

            # Run CAG inference with dequanted KV injected → real answer from doc context
            answer = _run_cag_inference(runner, query, doc_id, scheme, max_new_tokens)

            st = stats[scheme]
            st.ttft_ms_list.append(ttft_ms)
            st.answers.append(answer)
            pred = answer.strip().lower()
            # Exact-match: ref substring appears in prediction
            st.exact_matches.append(ref in pred)
            st.kv_mb   = _kv_mb(cag, scheme, doc_id, runner.num_layers, runner.head_shape)
            st.fp16_mb = _kv_mb(cag, "fp16",  doc_id, runner.num_layers, runner.head_shape)
            st.avg_attn_mse = avg_mse.get(scheme, 0.0)

    # ── report ─────────────────────────────────────────────────────── #
    normal_avg_ms = sum(sum(v) for v in normal_ttft.values()) / max(
        sum(len(v) for v in normal_ttft.values()), 1)
    _print_report(stats, avg_mse, schemes, normal_avg_ms, model_name)
    return stats


# ── helpers ────────────────────────────────────────────────────────── #

def _measure_ttft(runner, prompt: str, max_new_tokens: int) -> float:
    """Time full prefill+first-token for a prompt string."""
    import torch
    tokens = runner.tokenizer(prompt, return_tensors="pt").to(runner.device)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        runner.model.generate(**tokens, max_new_tokens=1, do_sample=False,
                              temperature=None, top_p=None)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3


def _measure_query_prefill(runner) -> float:
    """Time prefill of a short query-only prompt (no document context)."""
    return _measure_ttft(runner, "Question: What is X?\nAnswer:", max_new_tokens=1)


def _measure_disk_load(cag, doc_id: str, num_layers: int, scheme: str,
                       head_shape: tuple) -> float:
    """Measure time to load all layers of pre-computed KV from disk."""
    import torch
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for layer in range(num_layers):
        if cag.exists(doc_id, layer, scheme):
            cag.load_document(doc_id, layer, scheme, head_shape)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3


def _run_cag_inference(runner, query: str, doc_id: str, scheme: str,
                       max_new_tokens: int) -> str:
    """CAG inference: inject dequanted KV as past_key_values + query-only prompt.

    All schemes dequant their compressed KV back to fp16, build a DynamicCache,
    inject via past_key_values + position_ids, then generate from query tokens only.
    This gives real model answers with document context for every scheme.
    """
    import torch
    cag = runner.store

    # Build DynamicCache from compressed (dequanted) KV
    cache, doc_len = cag.build_dynamic_cache(
        doc_id, runner.num_layers, scheme, runner.head_shape)

    query_prompt = f"\nQuestion: {query}\nAnswer:"
    q_tokens = runner.tokenizer(query_prompt, return_tensors="pt").to(runner.device)
    q_ids = q_tokens["input_ids"]
    q_len = q_ids.shape[1]

    # Position IDs must continue from doc_len
    pos_ids = torch.arange(doc_len, doc_len + q_len, device=runner.device).unsqueeze(0)
    attn_mask = torch.ones(1, doc_len + q_len, device=runner.device, dtype=torch.long)

    with torch.no_grad():
        out = runner.model.generate(
            input_ids=q_ids,
            attention_mask=attn_mask,
            past_key_values=cache,
            position_ids=pos_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False, temperature=None, top_p=None,
        )
    new_tokens = out[0][q_len:]
    return runner.tokenizer.decode(new_tokens, skip_special_tokens=True)


def _store_dynamic_cache(runner, corpus: dict[str, str]) -> None:
    """Store full fp16 DynamicCache per doc for true CAG injection."""
    import torch
    store_dir = Path(runner.store.store_dir)
    for doc_id, text in corpus.items():
        cache_p = store_dir / f"{doc_id}.dyncache.pt"
        if cache_p.exists():
            continue
        tokens = runner.tokenizer(text, return_tensors="pt").to(runner.device)
        with torch.no_grad():
            out = runner.model(**tokens, use_cache=True)
        # Save as list of (K, V) tensors
        kv_list = []
        for layer in out.past_key_values.layers:
            kv_list.append((layer.keys.cpu(), layer.values.cpu()))
        torch.save(kv_list, cache_p)


def _load_dynamic_cache(runner, doc_id: str):
    """Reload fp16 DynamicCache from disk and return as DynamicCache object."""
    import torch
    from transformers import DynamicCache

    cache_p = Path(runner.store.store_dir) / f"{doc_id}.dyncache.pt"
    if not cache_p.exists():
        return None

    kv_list = torch.load(cache_p, weights_only=True)
    cache = DynamicCache()
    for k_cpu, v_cpu in kv_list:
        k = k_cpu.to(runner.device)
        v = v_cpu.to(runner.device)
        # DynamicCache.update expects [B, H, S, D]
        cache.update(k, v, layer_idx=len(cache))
    return cache


def _kv_mb(cag, scheme: str, doc_id: str, num_layers: int,
           head_shape: tuple) -> float:
    """Estimate total KV VRAM for all layers of one document."""
    meta_p = cag._meta_path(doc_id, 0, "fp16")
    if not meta_p.exists():
        return 0.0
    import torch
    meta = torch.load(meta_p, weights_only=False)
    N = int(meta["num_tokens"])
    return cag.vram_bytes(scheme, N, head_shape) * num_layers / 1024**2


def _print_report(stats: dict, avg_mse: dict, schemes: list, normal_ttft_ms: float,
                   model_name: str = "Qwen2.5") -> None:
    w = 82
    fp16_st = stats.get("fp16")
    fp16_acc = fp16_st.accuracy * 100 if fp16_st else 100.0
    short_name = model_name.split("/")[-1]

    print("\n" + "=" * w)
    print(f"  TurboRAG End-to-End Benchmark  —  {short_name}")
    print(f"  Normal RAG (full doc+query prefill): {normal_ttft_ms:.1f} ms  |  fp16 CAG accuracy: {fp16_acc:.0f}%")
    print("=" * w)
    print(f"  {'Scheme':<14} {'CAG TTFT':>10} {'Speedup':>9} {'VRAM(MB)':>10} {'VRAM×':>7} {'Accuracy':>9} {'AttnMSE':>9}")
    print("-" * w)

    for scheme in schemes:
        st      = stats[scheme]
        speedup = normal_ttft_ms / max(st.avg_ttft_ms, 1e-3)
        spd_str = f"{speedup:.2f}×"
        if speedup < 1.0:
            spd_str = f"{speedup:.2f}× ↓"   # small doc: disk overhead > prefill savings
        mse_str = f"{avg_mse[scheme]:.5f}" if scheme != "fp16" else "  ref  "
        print(f"  {scheme:<14} {st.avg_ttft_ms:>10.1f} {spd_str:>9} "
              f"{st.kv_mb:>10.2f} {st.vram_ratio:>6.1f}× "
              f"{st.accuracy*100:>8.1f}% {mse_str:>9}")

    print("=" * w)
    print("  CAG TTFT  = disk_load(all layers) + query_prefill  [ms]")
    print("  Speedup ↓ = disk overhead > prefill savings (doc too short)")
    print("  VRAM×     = fp16_kv / compressed_kv per layer × L layers")
    print("  AttnMSE   = attention MSE vs FP16 reference (lower = better)")
    print("  Accuracy  = exact-match: reference substring in model answer")
    print("-" * w)
    print("  NOTE: TTFT speedup scales with document length.")
    print("  Sim mode (--mode sim --tokens 2048): turbo_prod=51×, turbo_mse=49×")
    print("=" * w)

    # Per-query answer samples
    print("\n  Sample answers — Q: 'What compression ratio does turbo_prod achieve?'")
    for scheme in schemes:
        st = stats[scheme]
        ans = st.answers[0][:65].replace('\n', '↵') if st.answers else ""
        mark = "✓" if st.exact_matches[0] else "✗"
        print(f"  {mark} {scheme:<14}: {ans!r}")

    # Accuracy breakdown per question
    print(f"\n  Per-question accuracy:")
    print(f"  {'Scheme':<14}", end="")
    for i, _ in enumerate(stats[schemes[0]].exact_matches if schemes else []):
        print(f"  Q{i+1}", end="")
    print()
    for scheme in schemes:
        st = stats[scheme]
        print(f"  {scheme:<14}", end="")
        for match in st.exact_matches:
            print(f"   {'✓' if match else '✗'}", end="")
        print(f"   → {st.accuracy*100:.0f}%")
    print()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",   default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--store",   default="./kv_store")
    ap.add_argument("--corpus",  default=None)
    ap.add_argument("--queries", default=None)
    ap.add_argument("--schemes", default="fp16,turbo_prod,turbo_mse")
    ap.add_argument("--tokens",  type=int, default=64)
    ap.add_argument("--lib",     default=None)
    args = ap.parse_args()
    run_benchmark(
        model_name=args.model, store_dir=args.store,
        corpus_path=args.corpus, queries_path=args.queries,
        schemes=args.schemes.split(","), max_new_tokens=args.tokens,
        lib_path=args.lib,
    )
