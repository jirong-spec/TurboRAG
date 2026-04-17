"""TurboQuant vs FP16 — Paper-Grade Benchmark Suite.

Measures two axes for every token count in TOKEN_COUNTS:

  Efficiency  — Write latency (pack / FP16 copy), Read latency (dequant),
                and tokens-per-microsecond throughput.
  Fidelity    — Reconstruction MSE (K and V) and VRAM compression ratio.

Usage:
    python benchmark.py
    python benchmark.py --output-dir /path/to/save
"""
from __future__ import annotations

import argparse
import statistics as stat
import sys
import time
from pathlib import Path

import torch

# Allow running from any CWD
sys.path.insert(0, str(Path(__file__).resolve().parent))
from turboquant_wrapper import TurboQuantWrapper, TQConfig, TQTurboProdPageLayout

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# Configuration                                                                #
# --------------------------------------------------------------------------- #

TOKEN_COUNTS = [128, 512, 1024, 2048, 4096, 8192]
WARMUP_ITERS = 20
BENCH_ITERS  = 100


# --------------------------------------------------------------------------- #
# Timing primitive                                                             #
# --------------------------------------------------------------------------- #

def _sync() -> None:
    torch.cuda.synchronize()


def _bench(fn, iters: int) -> float:
    """Return mean wall-clock time in microseconds over `iters` iterations.

    Each iteration is individually fenced with torch.cuda.synchronize().
    """
    times: list[float] = []
    for _ in range(iters):
        _sync()
        t0 = time.perf_counter()
        fn()
        _sync()
        times.append((time.perf_counter() - t0) * 1e6)
    return stat.mean(times)


# --------------------------------------------------------------------------- #
# Per-token-count benchmarks                                                   #
# --------------------------------------------------------------------------- #

def bench_turboquant(tq: TurboQuantWrapper, tokens: int) -> dict:
    cfg    = tq.default_config()
    layout = tq.make_layout_for(cfg)
    H, D   = cfg.num_kv_heads, cfg.head_dim

    key          = torch.randn(tokens, H, D, device="cuda", dtype=torch.float16).contiguous()
    value        = torch.randn(tokens, H, D, device="cuda", dtype=torch.float16).contiguous()
    out_key      = torch.empty_like(key)
    out_value    = torch.empty_like(value)
    slot_mapping = torch.arange(tokens, dtype=torch.int32, device="cuda")
    page_pool    = tq.alloc_page_pool(tokens, layout, cfg)

    # Warmup — fills CUDA caches and avoids first-launch overhead
    for _ in range(WARMUP_ITERS):
        tq.pack(key, value, slot_mapping, page_pool, layout, cfg)
        tq.dequant(page_pool, slot_mapping, out_key, out_value, layout, cfg)
    _sync()

    pack_us   = _bench(lambda: tq.pack(key, value, slot_mapping, page_pool, layout, cfg), BENCH_ITERS)
    dequant_us = _bench(lambda: tq.dequant(page_pool, slot_mapping, out_key, out_value, layout, cfg), BENCH_ITERS)

    # Reconstruction fidelity — run one final pack+dequant, then compare
    tq.pack(key, value, slot_mapping, page_pool, layout, cfg)
    tq.dequant(page_pool, slot_mapping, out_key, out_value, layout, cfg)
    _sync()
    k_mse = torch.mean((key.float() - out_key.float())   ** 2).item()
    v_mse = torch.mean((value.float() - out_value.float()) ** 2).item()

    return {
        "tokens":    tokens,
        "pack_us":   pack_us,
        "dequant_us": dequant_us,
        "k_mse":     k_mse,
        "v_mse":     v_mse,
        "quant_kb":  tq.quant_bytes(tokens, layout, cfg) / 1024,
        "fp16_kb":   tq.fp16_bytes(tokens, cfg) / 1024,
    }


def bench_fp16(tokens: int, cfg: TQConfig) -> dict:
    """FP16 baseline: contiguous copy for 'write', identity view for 'read'."""
    H, D  = cfg.num_kv_heads, cfg.head_dim
    key   = torch.randn(tokens, H, D, device="cuda", dtype=torch.float16).contiguous()
    value = torch.randn(tokens, H, D, device="cuda", dtype=torch.float16).contiguous()
    # Pre-allocate destination buffer (simulates a paged FP16 KV store)
    kv_buf = torch.empty(2 * tokens, H, D, device="cuda", dtype=torch.float16)

    for _ in range(WARMUP_ITERS):
        kv_buf[:tokens].copy_(key)
        kv_buf[tokens:].copy_(value)
    _sync()

    # Time a full write: copy both K and V into the contiguous buffer
    def _write():
        kv_buf[:tokens].copy_(key)
        kv_buf[tokens:].copy_(value)

    copy_us = _bench(_write, BENCH_ITERS)
    fp16_kb = tokens * H * D * 2 * 2 / 1024   # K + V, 2 bytes/element

    return {"tokens": tokens, "copy_us": copy_us, "fp16_kb": fp16_kb}


# --------------------------------------------------------------------------- #
# Reporting                                                                    #
# --------------------------------------------------------------------------- #

def print_markdown_table(tq_results: list[dict], fp16_results: list[dict]) -> None:
    print("\n## TurboQuant vs FP16 — Benchmark Results\n")
    print(
        f"| {'Tokens':>6} "
        f"| {'TQ Pack (µs)':>13} "
        f"| {'TQ Dequant (µs)':>16} "
        f"| {'FP16 Copy (µs)':>15} "
        f"| {'K MSE':>10} "
        f"| {'V MSE':>10} "
        f"| {'TQ Mem (KB)':>12} "
        f"| {'FP16 Mem (KB)':>14} "
        f"| {'Compression':>12} |"
    )
    print(
        f"|{'-':->8}|{'-':->15}|{'-':->18}|{'-':->17}"
        f"|{'-':->12}|{'-':->12}|{'-':->14}|{'-':->16}|{'-':->14}|"
    )
    for tq, fp16 in zip(tq_results, fp16_results):
        ratio = fp16["fp16_kb"] / tq["quant_kb"]
        print(
            f"| {tq['tokens']:>6} "
            f"| {tq['pack_us']:>13.2f} "
            f"| {tq['dequant_us']:>16.2f} "
            f"| {fp16['copy_us']:>15.2f} "
            f"| {tq['k_mse']:>10.5f} "
            f"| {tq['v_mse']:>10.5f} "
            f"| {tq['quant_kb']:>12.1f} "
            f"| {fp16['fp16_kb']:>14.1f} "
            f"| {ratio:>11.2f}x |"
        )


def save_markdown_table(tq_results: list[dict], fp16_results: list[dict], path: Path) -> None:
    lines = [
        "## TurboQuant vs FP16 — Benchmark Results\n",
        f"| {'Tokens':>6} "
        f"| {'TQ Pack (µs)':>13} "
        f"| {'TQ Dequant (µs)':>16} "
        f"| {'FP16 Copy (µs)':>15} "
        f"| {'K MSE':>10} "
        f"| {'V MSE':>10} "
        f"| {'TQ Mem (KB)':>12} "
        f"| {'FP16 Mem (KB)':>14} "
        f"| {'Compression':>12} |\n",
        f"|{'-':->8}|{'-':->15}|{'-':->18}|{'-':->17}"
        f"|{'-':->12}|{'-':->12}|{'-':->14}|{'-':->16}|{'-':->14}|\n",
    ]
    for tq, fp16 in zip(tq_results, fp16_results):
        ratio = fp16["fp16_kb"] / tq["quant_kb"]
        lines.append(
            f"| {tq['tokens']:>6} "
            f"| {tq['pack_us']:>13.2f} "
            f"| {tq['dequant_us']:>16.2f} "
            f"| {fp16['copy_us']:>15.2f} "
            f"| {tq['k_mse']:>10.5f} "
            f"| {tq['v_mse']:>10.5f} "
            f"| {tq['quant_kb']:>12.1f} "
            f"| {fp16['fp16_kb']:>14.1f} "
            f"| {ratio:>11.2f}x |\n"
        )
    path.write_text("".join(lines))
    print(f"Markdown table  → {path}")


def plot_results(
    tq_results: list[dict],
    fp16_results: list[dict],
    out_path: Path,
) -> None:
    tokens     = [r["tokens"]     for r in tq_results]
    pack_us    = [r["pack_us"]    for r in tq_results]
    dequant_us = [r["dequant_us"] for r in tq_results]
    copy_us    = [r["copy_us"]    for r in fp16_results]
    k_mse      = [r["k_mse"]     for r in tq_results]
    v_mse      = [r["v_mse"]     for r in tq_results]
    tq_mem     = [r["quant_kb"]  for r in tq_results]
    fp16_mem   = [r["fp16_kb"]   for r in fp16_results]
    ratios     = [f / q for f, q in zip(fp16_mem, tq_mem)]

    xi = list(range(len(tokens)))
    xlabels = [str(t) for t in tokens]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("TurboQuant (INT3+INT4) vs FP16 — Performance & Accuracy", fontsize=13, fontweight="bold")

    # Panel 1: Write latency
    ax = axes[0, 0]
    ax.plot(tokens, pack_us,  "o-",  color="tab:blue",   label="TQ Pack (INT3 K + INT4 V)")
    ax.plot(tokens, copy_us,  "s--", color="tab:gray",   label="FP16 Contiguous Copy")
    ax.set_title("Write Latency")
    ax.set_xlabel("Tokens")
    ax.set_ylabel("Latency (µs)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: Read / dequant latency
    ax = axes[0, 1]
    ax.plot(tokens, dequant_us, "o-", color="tab:orange", label="TQ Dequant")
    ax.set_title("Read (Dequant) Latency")
    ax.set_xlabel("Tokens")
    ax.set_ylabel("Latency (µs)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 3: Reconstruction MSE
    ax = axes[1, 0]
    ax.plot(tokens, k_mse, "D-", color="tab:red",   label="Key MSE")
    ax.plot(tokens, v_mse, "x-", color="tab:green", label="Value MSE")
    ax.set_title("Reconstruction MSE (↓ Better)")
    ax.set_xlabel("Tokens")
    ax.set_ylabel("Mean Squared Error")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 4: Memory footprint (grouped bar)
    w = 0.35
    ax = axes[1, 1]
    ax.bar([x - w / 2 for x in xi], fp16_mem, width=w, label="FP16",        color="tab:blue",   alpha=0.8)
    ax.bar([x + w / 2 for x in xi], tq_mem,   width=w, label="TurboQuant",  color="tab:orange", alpha=0.8)
    for x, r in zip(xi, ratios):
        ax.text(x, fp16_mem[x] * 1.02, f"{r:.1f}×", ha="center", fontsize=8, color="black")
    ax.set_xticks(xi)
    ax.set_xticklabels(xlabels)
    ax.set_title("VRAM Footprint (↓ Better)")
    ax.set_xlabel("Tokens")
    ax.set_ylabel("Memory (KB)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Chart           → {out_path}")


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description="TurboQuant benchmark")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory for PNG chart and Markdown table (default: script directory)",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA device required")

    tq  = TurboQuantWrapper()
    cfg = tq.default_config()

    gpu = torch.cuda.get_device_name(0)
    print(f"GPU       : {gpu}")
    print(f"Config    : block_size={cfg.block_size}, heads={cfg.num_kv_heads}, head_dim={cfg.head_dim}")
    print(f"Tokens    : {TOKEN_COUNTS}")
    print(f"Warmup/iter: {WARMUP_ITERS}/{BENCH_ITERS}\n")

    tq_results   = [bench_turboquant(tq, t) for t in TOKEN_COUNTS]
    fp16_results = [bench_fp16(t, cfg)       for t in TOKEN_COUNTS]

    print_markdown_table(tq_results, fp16_results)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    save_markdown_table(tq_results, fp16_results, out_dir / "benchmark_results.md")
    plot_results(tq_results, fp16_results,         out_dir / "turboquant_benchmark_report.png")


if __name__ == "__main__":
    main()
