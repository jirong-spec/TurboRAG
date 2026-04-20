"""ttft_sim.py — Pure-GPU TTFT benchmark (no LLM download required).

Simulates end-to-end TTFT on real GPU hardware:
  FP16       : L × (QKV proj + QK^T attn + FFN) matmuls
  turbo_prod : L × disk_load_time (pre-packed turbo_prod KV)
  turbo_mse  : L × disk_load_time (pre-packed turbo_mse KV)

Uses actual CUDA timing, not estimates.  No model weights needed.
"""
from __future__ import annotations

import time
from pathlib import Path

import torch

from .turboquant_wrapper import TurboQuantWrapper, TQConfig


# ── Qwen2.5-0.5B scale defaults ────────────────────────────────────── #
QWEN_05B = dict(
    d_model=896,
    num_layers=24,
    num_kv_heads=2,
    head_dim=64,
    num_heads=14,
)


def _make_cfg(num_kv_heads: int, head_dim: int) -> TQConfig:
    tq   = TurboQuantWrapper()
    cfg  = tq.default_config()
    cfg.num_kv_heads = num_kv_heads
    cfg.head_dim     = head_dim
    return cfg


def simulate_prefill_us(
    num_tokens: int,
    d_model: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    num_layers: int,
    warmup: int = 3,
    iters: int  = 20,
) -> float:
    """Simulate L-layer transformer prefill via real CUDA matmuls.

    Includes per-layer: QKV projections, scaled dot-product attention, FFN (4×).
    """
    d_ffn  = d_model * 4
    D_full = num_heads * head_dim

    x  = torch.randn(num_tokens, d_model, device="cuda", dtype=torch.float16)
    wq = torch.randn(d_model, D_full,    device="cuda", dtype=torch.float16)
    wk = torch.randn(d_model, num_kv_heads * head_dim, device="cuda", dtype=torch.float16)
    wv = torch.randn(d_model, num_kv_heads * head_dim, device="cuda", dtype=torch.float16)
    wo = torch.randn(D_full,  d_model,   device="cuda", dtype=torch.float16)
    w1 = torch.randn(d_model, d_ffn,     device="cuda", dtype=torch.float16)
    w2 = torch.randn(d_ffn,   d_model,   device="cuda", dtype=torch.float16)
    scale = head_dim ** -0.5

    def _one_pass():
        _x = x
        for _ in range(num_layers):
            q  = (_x @ wq).view(num_tokens, num_heads, head_dim).permute(1, 0, 2)
            k  = (_x @ wk).view(num_tokens, num_kv_heads, head_dim)
            v  = (_x @ wv).view(num_tokens, num_kv_heads, head_dim)
            # GQA: expand KV to match Q heads
            groups = num_heads // num_kv_heads
            k = k.unsqueeze(2).expand(-1, -1, groups, -1).reshape(num_tokens, num_heads, head_dim).permute(1, 0, 2)
            v = v.unsqueeze(2).expand(-1, -1, groups, -1).reshape(num_tokens, num_heads, head_dim).permute(1, 0, 2)
            scores = torch.bmm(q, k.transpose(1, 2)) * scale
            attn   = torch.softmax(scores, dim=-1)
            _x     = torch.bmm(attn, v).permute(1, 0, 2).reshape(num_tokens, D_full) @ wo
            _x     = (_x @ w1).relu() @ w2

    for _ in range(warmup):
        _one_pass(); torch.cuda.synchronize()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        _one_pass()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


def simulate_disk_load_us(
    tq: TurboQuantWrapper,
    scheme: str,
    num_tokens: int,
    num_kv_heads: int,
    head_dim: int,
    store_dir: Path,
    warmup: int = 3,
    iters: int  = 20,
) -> float:
    """Measure disk-load latency for one pre-packed KV layer."""
    cfg = _make_cfg(num_kv_heads, head_dim)

    if scheme == "turbo_prod":
        layout = tq.make_layout_for(cfg)
    elif scheme == "turbo_mse":
        layout = tq.make_mse_layout_for(cfg)
    else:
        return 0.0

    doc_id = f"_ttft_sim_{scheme}_{num_tokens}"
    pool_p = store_dir / f"{doc_id}.bin"

    # Write once
    if not pool_p.exists():
        key   = torch.randn(num_tokens, num_kv_heads, head_dim, device="cuda", dtype=torch.float16).contiguous()
        value = torch.randn(num_tokens, num_kv_heads, head_dim, device="cuda", dtype=torch.float16).contiguous()
        slots = torch.arange(num_tokens, dtype=torch.int32, device="cuda")

        if scheme == "turbo_prod":
            pool = tq.alloc_page_pool(num_tokens, layout, cfg)
            tq.pack(key, value, slots, pool, layout, cfg)
        elif scheme == "turbo_mse":
            pool = tq.alloc_mse_pool(num_tokens, layout, cfg)
            tq.mse_pack(key, value, slots, pool, layout, cfg)
        torch.cuda.synchronize()
        with pool_p.open("wb") as f:
            f.write(pool.cpu().numpy().tobytes())

    if scheme == "turbo_prod":
        pool_bytes = tq.quant_bytes(num_tokens, layout, cfg)
    else:
        pool_bytes = tq.mse_bytes(num_tokens, layout, cfg)

    def _load():
        raw = pool_p.read_bytes()
        pool_cpu = torch.frombuffer(bytearray(raw[:pool_bytes]), dtype=torch.uint8).clone()
        pool_cpu.to("cuda", non_blocking=True)
        torch.cuda.synchronize()

    for _ in range(warmup):
        _load()

    t0 = time.perf_counter()
    for _ in range(iters):
        _load()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


def run_ttft_sim(
    num_tokens: int   = 512,
    num_layers: int   = 24,
    warmup: int       = 3,
    iters: int        = 20,
    store_dir: Path   = Path("/tmp/ttft_sim_store"),
    model: dict | None = None,
) -> dict:
    if model is None:
        model = QWEN_05B
    model = {**QWEN_05B, **model}
    model["num_layers"] = num_layers

    store_dir.mkdir(parents=True, exist_ok=True)
    tq = TurboQuantWrapper()

    cfg = _make_cfg(model["num_kv_heads"], model["head_dim"])
    prod_layout = tq.make_layout_for(cfg)
    mse_layout  = tq.make_mse_layout_for(cfg)

    fp16_b = tq.fp16_bytes(num_tokens, cfg)
    prod_b = tq.quant_bytes(num_tokens, prod_layout, cfg)
    mse_b  = tq.mse_bytes(num_tokens, mse_layout, cfg)

    print(f"Simulating TTFT for N={num_tokens} tokens, L={num_layers} layers ...")

    prefill_us = simulate_prefill_us(
        num_tokens, model["d_model"], model["num_heads"],
        model["num_kv_heads"], model["head_dim"], num_layers,
        warmup, iters,
    )
    print(f"  FP16 prefill (L={num_layers}):  {prefill_us/1e3:.1f} ms")

    disk_us: dict[str, float] = {}
    for scheme in ["turbo_prod", "turbo_mse"]:
        us = simulate_disk_load_us(tq, scheme, num_tokens, model["num_kv_heads"], model["head_dim"], store_dir, warmup, iters)
        disk_us[scheme] = us * num_layers
        print(f"  {scheme:<12} disk load (L×):  {disk_us[scheme]/1e3:.1f} ms  (single={us:.1f} µs)")

    results = {
        "num_tokens": num_tokens,
        "num_layers": num_layers,
        "fp16": {
            "ttft_us":    prefill_us,
            "kv_mb":      fp16_b / 1024**2 * num_layers,
            "vram_ratio": 1.0,
            "speedup":    1.0,
        },
    }
    for scheme, b in [("turbo_prod", prod_b), ("turbo_mse", mse_b)]:
        ttft = disk_us[scheme]
        results[scheme] = {
            "ttft_us":    ttft,
            "kv_mb":      b / 1024**2 * num_layers,
            "vram_ratio": fp16_b / max(b, 1),
            "speedup":    prefill_us / max(ttft, 1e-3),
        }

    _print_ttft_table(results)
    return results


def _print_ttft_table(results: dict) -> None:
    print("\n" + "=" * 60)
    print("  TurboRAG TTFT Simulation  —  Qwen2.5-0.5B scale")
    print("=" * 60)
    print(f"  {'Scheme':<14} {'TTFT(ms)':>10} {'Speedup':>9} {'VRAM(MB)':>10} {'VRAM×':>7}")
    print("-" * 60)
    for scheme, r in results.items():
        if scheme in ("num_tokens", "num_layers"):
            continue
        ttft_ms  = r["ttft_us"] / 1e3
        speedup  = r["speedup"]
        vram_mb  = r["kv_mb"]
        vram_x   = r["vram_ratio"]
        print(f"  {scheme:<14} {ttft_ms:>10.1f} {speedup:>8.2f}× {vram_mb:>10.2f} {vram_x:>6.1f}×")
    print("=" * 60)
    print("  TTFT = time-to-first-token  |  Speedup vs FP16 prefill")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens",  type=int, default=512)
    ap.add_argument("--layers",  type=int, default=24)
    ap.add_argument("--warmup",  type=int, default=3)
    ap.add_argument("--iters",   type=int, default=20)
    ap.add_argument("--store",   default="/tmp/ttft_sim_store")
    args = ap.parse_args()

    run_ttft_sim(
        num_tokens=args.tokens,
        num_layers=args.layers,
        warmup=args.warmup,
        iters=args.iters,
        store_dir=Path(args.store),
    )
