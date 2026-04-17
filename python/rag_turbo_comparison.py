"""TurboQuant RAG Workflow — Quantized vs FP16 Comparison.

Simulates a two-phase RAG inference loop:

  Phase 1  Prefill / KV population
           Multiple retrieved documents are encoded and their KV tensors
           are stored.  TurboQuant packs them into a quantized paged pool;
           FP16 copies them into a contiguous buffer.

  Phase 2  Single-token decode (attention)
           TurboQuant uses the fused online-softmax kernel that reads
           directly from the compressed pool with no intermediate KV
           materialisation.  FP16 runs standard scaled dot-product attention
           via an explicit batched-matmul + softmax path.

Logit convention used by both paths: <q, k>  (no 1/sqrt(D) scale factor),
which matches the TurboQuant kernel.

Usage:
    python rag_turbo_comparison.py
    python rag_turbo_comparison.py --num-docs 8 --doc-tokens 1024
"""
from __future__ import annotations

import argparse
import statistics as stat
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from turboquant_wrapper import TurboQuantWrapper

DEVICE = "cuda"


# --------------------------------------------------------------------------- #
# Timing primitive                                                             #
# --------------------------------------------------------------------------- #

def _sync() -> None:
    torch.cuda.synchronize()


def _bench(fn, warmup: int, iters: int) -> float:
    """Return mean wall-clock time in µs.  Each iteration individually fenced."""
    for _ in range(warmup):
        fn()
    _sync()
    times: list[float] = []
    for _ in range(iters):
        _sync()
        t0 = time.perf_counter()
        fn()
        _sync()
        times.append((time.perf_counter() - t0) * 1e6)
    return stat.mean(times)


# --------------------------------------------------------------------------- #
# FP16 attention (matches TurboQuant logit convention: no 1/sqrt(D) scale)    #
# --------------------------------------------------------------------------- #

def fp16_attention(
    query: torch.Tensor,   # [num_q, H, D]  fp16
    key:   torch.Tensor,   # [num_kv, H, D] fp16
    value: torch.Tensor,   # [num_kv, H, D] fp16
) -> torch.Tensor:         # [num_q, H, D]  fp16
    """Batched multi-head attention with logit = <q, k> (unscaled).

    Uses float32 arithmetic to avoid fp16 overflow in the softmax.
    """
    # [H, num_q, D] and [H, num_kv, D]
    q = query.permute(1, 0, 2).float()
    k = key.permute(1, 0, 2).float()
    v = value.permute(1, 0, 2).float()

    logits = torch.bmm(q, k.transpose(-2, -1))     # [H, num_q, num_kv]
    probs  = torch.softmax(logits, dim=-1)
    out    = torch.bmm(probs, v)                    # [H, num_q, D]

    return out.permute(1, 0, 2).to(torch.float16).contiguous()


# --------------------------------------------------------------------------- #
# Main comparison                                                              #
# --------------------------------------------------------------------------- #

def run_comparison(
    num_docs:   int,
    doc_tokens: int,
    warmup:     int,
    iters:      int,
) -> None:
    if not torch.cuda.is_available():
        sys.exit("CUDA device required")

    tq         = TurboQuantWrapper()
    cfg        = tq.default_config()
    layout     = tq.make_layout_for(cfg)
    H, D       = cfg.num_kv_heads, cfg.head_dim
    kv_tokens  = num_docs * doc_tokens
    num_q      = 1        # single decode step

    gpu = torch.cuda.get_device_name(0)
    print(f"GPU       : {gpu}")
    print(f"Context   : {num_docs} docs × {doc_tokens} tokens = {kv_tokens} KV tokens")
    print(f"Query     : {num_q} decode token(s)")
    print(f"Heads/Dim : {H} × {D}\n")

    # ------------------------------------------------------------------ #
    # Allocate tensors                                                     #
    # ------------------------------------------------------------------ #

    # Simulated prefill output: KV tensors from encoding retrieved docs
    key          = torch.randn(kv_tokens, H, D, device=DEVICE, dtype=torch.float16).contiguous()
    value        = torch.randn(kv_tokens, H, D, device=DEVICE, dtype=torch.float16).contiguous()
    query        = torch.randn(num_q,     H, D, device=DEVICE, dtype=torch.float16).contiguous()
    slot_mapping = torch.arange(kv_tokens, dtype=torch.int32, device=DEVICE)

    # TurboQuant: quantized paged pool
    page_pool = tq.alloc_page_pool(kv_tokens, layout, cfg)

    # FP16: contiguous KV store (K and V interleaved in a single buffer)
    fp16_kv = torch.empty(2 * kv_tokens, H, D, device=DEVICE, dtype=torch.float16)
    fp16_k  = fp16_kv[:kv_tokens]
    fp16_v  = fp16_kv[kv_tokens:]

    # TurboQuant attention output buffer
    tq_out  = torch.empty(num_q, H, D, device=DEVICE, dtype=torch.float16).contiguous()

    # ------------------------------------------------------------------ #
    # Benchmark helpers                                                    #
    # ------------------------------------------------------------------ #

    def tq_prefill():
        tq.pack(key, value, slot_mapping, page_pool, layout, cfg)

    def tq_decode():
        tq.fused_attn_output(
            query, page_pool, slot_mapping, tq_out,
            layout, cfg, num_q, kv_tokens,
        )

    def fp16_prefill():
        fp16_k.copy_(key)
        fp16_v.copy_(value)

    def fp16_decode():
        fp16_attention(query, fp16_k, fp16_v)

    # ------------------------------------------------------------------ #
    # Phase 1: Prefill                                                     #
    # ------------------------------------------------------------------ #

    tq_pack_us   = _bench(tq_prefill,   warmup, iters)
    fp16_store_us = _bench(fp16_prefill, warmup, iters)

    # ------------------------------------------------------------------ #
    # Phase 2: Decode                                                      #
    # ------------------------------------------------------------------ #

    # Ensure page pool is populated before timing decode
    tq_prefill();  fp16_prefill()
    _sync()

    tq_decode_us   = _bench(tq_decode,   warmup, iters)
    fp16_decode_us = _bench(fp16_decode, warmup, iters)

    # ------------------------------------------------------------------ #
    # Accuracy — attention output MSE                                      #
    # ------------------------------------------------------------------ #

    tq_decode();  _sync()
    fp16_out = fp16_attention(query, fp16_k, fp16_v);  _sync()

    attn_mse = torch.mean((tq_out.float() - fp16_out.float()) ** 2).item()

    # ------------------------------------------------------------------ #
    # Memory                                                               #
    # ------------------------------------------------------------------ #

    tq_mem_kb   = tq.quant_bytes(kv_tokens, layout, cfg) / 1024
    fp16_mem_kb = tq.fp16_bytes(kv_tokens, cfg) / 1024
    compression = fp16_mem_kb / tq_mem_kb

    # ------------------------------------------------------------------ #
    # Report                                                               #
    # ------------------------------------------------------------------ #

    print("## Phase 1 — Prefill / KV Population\n")
    print(f"| {'Metric':<30} | {'TurboQuant':>22} | {'FP16':>14} |")
    print(f"|{'-'*32}|{'-'*24}|{'-'*16}|")
    print(f"| {'Pack / copy latency (µs)':<30} | {tq_pack_us:>22.2f} | {fp16_store_us:>14.2f} |")
    print(f"| {'KV memory (KB)':<30} | {tq_mem_kb:>22.1f} | {fp16_mem_kb:>14.1f} |")
    print(f"| {'Compression vs FP16':<30} | {compression:>21.2f}x | {'1.00x':>14} |")
    print(f"| {'Representation':<30} | {'INT3 (K) + INT4 (V)':>22} | {'FP16':>14} |")

    print("\n## Phase 2 — Single-Token Decode (Attention)\n")
    print(f"| {'Metric':<30} | {'TurboQuant':>22} | {'FP16':>14} |")
    print(f"|{'-'*32}|{'-'*24}|{'-'*16}|")
    print(f"| {'Attention latency (µs)':<30} | {tq_decode_us:>22.2f} | {fp16_decode_us:>14.2f} |")
    print(f"| {'Output MSE vs FP16':<30} | {attn_mse:>22.6f} | {'0.000000':>14} |")
    print(f"| {'No KV materialisation':<30} | {'Yes (fused)':>22} | {'N/A':>14} |")

    print("\n## End-to-End Totals\n")
    tq_e2e   = tq_pack_us   + tq_decode_us
    fp16_e2e = fp16_store_us + fp16_decode_us
    print(f"| {'Metric':<30} | {'TurboQuant':>22} | {'FP16':>14} |")
    print(f"|{'-'*32}|{'-'*24}|{'-'*16}|")
    print(f"| {'Total latency (µs)':<30} | {tq_e2e:>22.2f} | {fp16_e2e:>14.2f} |")
    print(f"| {'Peak VRAM (KB)':<30} | {tq_mem_kb:>22.1f} | {fp16_mem_kb:>14.1f} |")
    print(f"| {'Attention output MSE':<30} | {attn_mse:>22.6f} | {'0.000000':>14} |")


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description="TurboQuant RAG comparison")
    parser.add_argument("--num-docs",   type=int, default=4,   help="Number of retrieved documents")
    parser.add_argument("--doc-tokens", type=int, default=512, help="Tokens per document")
    parser.add_argument("--warmup",     type=int, default=10,  help="Warmup iterations")
    parser.add_argument("--iters",      type=int, default=50,  help="Benchmark iterations")
    args = parser.parse_args()

    run_comparison(
        num_docs   = args.num_docs,
        doc_tokens = args.doc_tokens,
        warmup     = args.warmup,
        iters      = args.iters,
    )


if __name__ == "__main__":
    main()
