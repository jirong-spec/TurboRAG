"""longbench_eval.py — LongBench TTFT / VRAM / F1 evaluation for TurboCAG.

Methodology
───────────
  Offline  : precompute_corpus(context_field, schemes)
             context is truncated to max_length tokens before the forward pass
  Online   : TTFT = disk_load(all L layers) + query_prefill(question only)
             VRAM  = theoretical compressed bytes (consistent across runs)
             F1    = token-level F1 vs LongBench "answers" field

OOM handling
────────────
  Both torch.cuda.OutOfMemoryError and RuntimeError("CUDA out of memory")
  are caught per-sample-per-scheme and counted in SchemeStats.oom.

Supported datasets
──────────────────
  qasper     single-doc QA on scientific papers  (~12K tokens avg)
  2wikimqa   multi-hop QA                        (~5K tokens avg)
  gov_report long-document summarization         (~8K tokens avg)
"""
from __future__ import annotations

import re
import string
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import torch

# ── OOM exception compat (PyTorch < 2.0 has no OutOfMemoryError) ──── #
try:
    _OOM_ERRORS = (RuntimeError, torch.cuda.OutOfMemoryError)
except AttributeError:
    _OOM_ERRORS = (RuntimeError,)


# ── F1 evaluation ─────────────────────────────────────────────────── #

def _normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = re.sub(f'[{re.escape(string.punctuation)}]', ' ', s)
    return ' '.join(s.split())


def _token_f1(pred: str, ref: str) -> float:
    p = _normalize(pred).split()
    r = _normalize(ref).split()
    if not p or not r:
        return float(p == r)
    n_common = sum((Counter(p) & Counter(r)).values())
    if n_common == 0:
        return 0.0
    prec = n_common / len(p)
    rec  = n_common / len(r)
    return 2 * prec * rec / (prec + rec)


def compute_f1(prediction: str, ground_truths: list[str]) -> float:
    """Max token-level F1 over all reference strings."""
    if not ground_truths:
        return 0.0
    return max(_token_f1(prediction, gt) for gt in ground_truths)


# ── dataset helpers ────────────────────────────────────────────────── #

_QA_DATASETS  = {
    "qasper", "2wikimqa", "multifieldqa_en",
    "hotpotqa", "triviaqa", "narrativeqa", "musique",
}
_SUM_DATASETS = {"gov_report", "qmsum", "multi_news", "vcsum"}


def _extract_query(item: dict, dataset_name: str) -> str:
    """Return the query/instruction suffix that follows the context."""
    inp = item.get("input", "")

    if dataset_name in _QA_DATASETS:
        # LongBench QA: question is at the end after the last "Question:"
        if "\nQuestion:" in inp:
            tail = inp.split("\nQuestion:")[-1].strip()
            tail = tail.split("\nAnswer:")[0].strip()
            return f"Question: {tail}\nAnswer:"
        for ln in reversed(inp.strip().splitlines()):
            if ln.strip():
                return f"{ln.strip()}\nAnswer:"

    if dataset_name in _SUM_DATASETS:
        return "Please provide a concise summary of the document above.\nSummary:"

    # Generic fallback: last paragraph of input
    paras = [p.strip() for p in inp.split("\n\n") if p.strip()]
    if paras:
        return f"{paras[-1]}\n"
    return "Respond based on the context above.\n"


def load_longbench_samples(
    dataset_name: str,
    tokenizer,
    max_samples: int,
    max_length: int,
) -> list[dict]:
    """Load LongBench test split; truncate context to max_length tokens.

    Returns list of dicts:
        doc_id, context_text, query_prompt, answers, context_tokens
    """
    from datasets import load_dataset

    print(f"Loading THUDM/LongBench '{dataset_name}' (test split) ...")
    ds = load_dataset("THUDM/LongBench", dataset_name, split="test",
                      trust_remote_code=True)
    samples: list[dict] = []

    for i, item in enumerate(ds):
        if i >= max_samples:
            break

        context = item.get("context", "") or item.get("input", "")

        # Truncate context to max_length tokens
        ctx_ids = tokenizer.encode(context, add_special_tokens=False)
        if len(ctx_ids) > max_length:
            ctx_ids = ctx_ids[:max_length]
            context  = tokenizer.decode(ctx_ids, skip_special_tokens=True)

        samples.append({
            "doc_id":         f"{dataset_name}_{i:04d}",
            "context_text":   context,
            "query_prompt":   _extract_query(item, dataset_name),
            "answers":        item.get("answers", []),
            "context_tokens": len(ctx_ids),
        })

    avg_tok = sum(s["context_tokens"] for s in samples) // max(len(samples), 1)
    print(f"  {len(samples)} samples loaded, avg {avg_tok} context tokens "
          f"(max_length={max_length})")
    return samples


# ── per-scheme statistics ──────────────────────────────────────────── #

@dataclass
class LBSchemeStats:
    scheme: str
    ttft_ms:  list[float] = field(default_factory=list)
    kv_mb:    list[float] = field(default_factory=list)
    f1:       list[float] = field(default_factory=list)
    oom: int = 0

    @property
    def avg_ttft(self):  return sum(self.ttft_ms) / max(len(self.ttft_ms), 1)
    @property
    def avg_kv_mb(self): return sum(self.kv_mb)   / max(len(self.kv_mb),   1)
    @property
    def avg_f1(self):    return sum(self.f1)       / max(len(self.f1),      1)


# ── timing / inference helpers ─────────────────────────────────────── #

def _sync_time(fn) -> float:
    """Run fn(), return wall time in ms with CUDA sync."""
    torch.cuda.synchronize()
    t = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) * 1e3


def _measure_prefill_ms(runner, prompt: str) -> float:
    toks = runner.tokenizer(prompt, return_tensors="pt").to(runner.device)
    return _sync_time(lambda: runner.model.generate(
        **toks, max_new_tokens=1, do_sample=False,
        temperature=None, top_p=None))


def _measure_disk_load_ms(cag, doc_id: str, num_layers: int,
                           scheme: str, head_shape: tuple) -> float:
    """Time to load all layers from disk (no tensors retained)."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for layer in range(num_layers):
        if cag.exists(doc_id, layer, scheme):
            cag.load_document(doc_id, layer, scheme, head_shape)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3


def _run_cag_inference(runner, query_prompt: str, doc_id: str,
                       scheme: str, max_new_tokens: int) -> str:
    """Build DynamicCache from disk KV, inject into model, generate."""
    cache, doc_len = runner.store.build_dynamic_cache(
        doc_id, runner.num_layers, scheme, runner.head_shape)

    q_toks = runner.tokenizer(
        "\n" + query_prompt, return_tensors="pt").to(runner.device)
    q_ids = q_toks["input_ids"]
    q_len = q_ids.shape[1]

    pos_ids   = torch.arange(doc_len, doc_len + q_len,
                              device=runner.device).unsqueeze(0)
    attn_mask = torch.ones(1, doc_len + q_len,
                           device=runner.device, dtype=torch.long)

    with torch.no_grad():
        out = runner.model.generate(
            input_ids=q_ids,
            attention_mask=attn_mask,
            past_key_values=cache,
            position_ids=pos_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False, temperature=None, top_p=None,
        )
    return runner.tokenizer.decode(out[0][q_len:], skip_special_tokens=True)


# ── main benchmark ─────────────────────────────────────────────────── #

def run_longbench_benchmark(
    dataset_name:   str = "qasper",
    model_name:     str = "Qwen/Qwen2.5-3B-Instruct",
    store_dir:      str = "./kv_store_longbench",
    schemes: list[str] | None = None,
    max_samples:    int  = 20,
    max_length:     int  = 32768,
    max_new_tokens: int  = 64,
    ttft_only:      bool = False,
    lib_path: str | None = None,
) -> dict[str, LBSchemeStats]:
    """Run the full LongBench pipeline and return per-scheme statistics.

    Args:
        dataset_name:   LongBench subset (qasper / 2wikimqa / gov_report)
        model_name:     HuggingFace model ID; Qwen2.5-3B-Instruct recommended
        store_dir:      Directory for pre-computed KV files
        schemes:        Compression schemes to compare
        max_samples:    Number of LongBench test items to evaluate
        max_length:     Context truncation in tokens (Qwen2.5-3B limit: 32768)
        max_new_tokens: Max generated tokens per answer
        ttft_only:      Skip inference; measure TTFT + VRAM only (no F1)
        lib_path:       Path to libturboquant.so (auto-detected if None)
    """
    from tq_backend.model_runner import TQModelRunner

    if schemes is None:
        schemes = ["fp16", "turbo_prod", "turbo_mse", "polar"]

    # ── init ──────────────────────────────────────────────────────── #
    runner = TQModelRunner(model_name, store_dir, lib_path)
    cag    = runner.store

    # ── load dataset ─────────────────────────────────────────────── #
    samples = load_longbench_samples(
        dataset_name, runner.tokenizer, max_samples, max_length)

    # ── [1] offline precompute ────────────────────────────────────── #
    print(f"\n[1/3] Pre-computing KV caches ({len(schemes)} schemes × "
          f"{len(samples)} docs) ...")
    corpus = {s["doc_id"]: s["context_text"] for s in samples}
    runner.precompute_corpus(corpus, schemes=schemes, max_length=max_length)

    # ── [2] normal-RAG baseline ───────────────────────────────────── #
    print("\n[2/3] Normal RAG baseline (full doc + query prefill) ...")
    normal_ms: list[float] = []
    if not ttft_only:
        _measure_prefill_ms(runner, "Warmup.")
        _measure_prefill_ms(runner, "Warmup 2.")
        for samp in samples[:min(5, len(samples))]:
            full = samp["context_text"] + "\n\n" + samp["query_prompt"]
            ttft = _measure_prefill_ms(runner, full)
            normal_ms.append(ttft)
            print(f"  {samp['doc_id']}  {ttft:.1f} ms "
                  f"({samp['context_tokens']} tok)")
    normal_avg_ms = sum(normal_ms) / max(len(normal_ms), 1)

    # Query-only prefill is shared across all schemes
    query_prefill_ms = _measure_prefill_ms(runner, samples[0]["query_prompt"]) \
        if samples else 5.0
    print(f"  query-only prefill: {query_prefill_ms:.1f} ms")

    # ── [3] CAG per sample per scheme ────────────────────────────── #
    print(f"\n[3/3] CAG — {len(samples)} samples × {len(schemes)} schemes ...")
    stats: dict[str, LBSchemeStats] = {s: LBSchemeStats(scheme=s) for s in schemes}
    fp16_mb_per_doc: dict[str, float] = {}  # for compression ratio

    for i, samp in enumerate(samples):
        doc_id = samp["doc_id"]
        N      = samp["context_tokens"]
        print(f"\n  [{i+1:02d}/{len(samples):02d}] {doc_id}  ({N} ctx tokens)")

        for scheme in schemes:
            st = stats[scheme]
            try:
                # ---- TTFT ------------------------------------------------
                disk_ms = _measure_disk_load_ms(
                    cag, doc_id, runner.num_layers, scheme, runner.head_shape)
                ttft_ms = disk_ms + query_prefill_ms

                # ---- VRAM (theoretical — consistent across machines) ------
                kv_mb = (cag.vram_bytes(scheme, N, runner.head_shape)
                         * runner.num_layers / 1024 ** 2)
                if scheme == "fp16":
                    fp16_mb_per_doc[doc_id] = kv_mb

                st.ttft_ms.append(ttft_ms)
                st.kv_mb.append(kv_mb)

                fp16_ref = fp16_mb_per_doc.get(doc_id, kv_mb)
                ratio    = fp16_ref / max(kv_mb, 1e-6)

                # ---- answer + F1 -----------------------------------------
                if not ttft_only and samp["answers"]:
                    try:
                        answer = _run_cag_inference(
                            runner, samp["query_prompt"], doc_id,
                            scheme, max_new_tokens)
                        f1 = compute_f1(answer, samp["answers"])
                        st.f1.append(f1)
                        print(f"    {scheme:<12} TTFT={ttft_ms:6.1f}ms  "
                              f"KV={kv_mb:5.1f}MB ({ratio:.1f}×)  "
                              f"F1={f1:.3f}  {answer[:45]!r}")
                    except _OOM_ERRORS as exc:
                        _check_oom(exc)
                        print(f"    {scheme:<12} inference OOM: {exc}")
                        st.oom += 1
                        torch.cuda.empty_cache()
                else:
                    print(f"    {scheme:<12} TTFT={ttft_ms:6.1f}ms  "
                          f"KV={kv_mb:5.1f}MB ({ratio:.1f}×)")

                torch.cuda.empty_cache()

            except _OOM_ERRORS as exc:
                _check_oom(exc)
                print(f"    {scheme:<12} OOM: {exc}")
                st.oom += 1
                torch.cuda.empty_cache()

    _print_report(stats, schemes, normal_avg_ms, fp16_mb_per_doc,
                  model_name, dataset_name)
    return stats


def _check_oom(exc: Exception) -> None:
    """Re-raise non-OOM RuntimeErrors so programming bugs surface."""
    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        if "out of memory" not in msg and "cuda" not in msg:
            raise exc


# ── report ─────────────────────────────────────────────────────────── #

def _print_report(
    stats: dict[str, LBSchemeStats],
    schemes: list[str],
    normal_avg_ms: float,
    fp16_mb_per_doc: dict[str, float],
    model_name: str,
    dataset_name: str,
) -> None:
    fp16_avg = sum(fp16_mb_per_doc.values()) / max(len(fp16_mb_per_doc), 1)
    short    = model_name.split("/")[-1]
    has_f1   = any(stats[s].f1 for s in schemes)
    w = 88

    print("\n" + "=" * w)
    print(f"  TurboCAG × LongBench/{dataset_name}  —  {short}")
    if normal_avg_ms:
        print(f"  Normal RAG (full doc+query prefill, 5-sample avg): "
              f"{normal_avg_ms:.1f} ms")
    print("=" * w)

    hdr = (f"  {'Scheme':<14} {'TTFT(ms)':>10} {'Speedup':>9} "
           f"{'KV(MB)':>8} {'VRAM×':>7}")
    hdr += f" {'F1':>7}"  if has_f1 else ""
    hdr += f" {'OOM':>4}"
    print(hdr)
    print("-" * w)

    for scheme in schemes:
        st = stats[scheme]
        if not st.ttft_ms:
            row = f"  {scheme:<14} {'—':>10} {'—':>9} {'—':>8} {'—':>7}"
            row += f" {'—':>7}" if has_f1 else ""
            row += f" {st.oom:>4}"
            print(row)
            continue

        speedup = normal_avg_ms / max(st.avg_ttft, 1e-3) if normal_avg_ms else 0.0
        vram_x  = fp16_avg / max(st.avg_kv_mb, 1e-6)
        row = (f"  {scheme:<14} {st.avg_ttft:>10.1f} {speedup:>8.1f}× "
               f"{st.avg_kv_mb:>8.1f} {vram_x:>6.1f}×")
        row += f" {st.avg_f1:>7.3f}" if has_f1 else ""
        row += f" {st.oom:>4}"
        print(row)

    print("=" * w)
    print("  TTFT    = disk_load(all L layers) + query_prefill  [ms]")
    print("  VRAM×   = fp16_kv / compressed_kv  (theoretical, avg over samples)")
    if has_f1:
        print("  F1      = token-level F1 vs LongBench ground truth (avg)")
    print("  OOM     = out-of-memory or CUDA error count")
    print("=" * w)
