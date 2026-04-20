"""model_runner.py — TQModelRunner: Qwen2.5-0.5B + TurboQuant attention.

Usage
─────
runner = TQModelRunner("Qwen/Qwen2.5-0.5B-Instruct", store_dir="./kv_store")

# Offline: encode documents (once, then cached on disk)
runner.precompute_corpus({"doc1": "text of doc 1", "doc2": "text of doc 2"})

# Online: compare all schemes
results = runner.benchmark_query(
    query="What is TurboQuant?",
    doc_ids=["doc1"],
    schemes=["fp16", "turbo_prod", "turbo_mse", "polar"],
)
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

from .turboquant_wrapper import TurboQuantWrapper

Scheme = Literal["fp16", "turbo_prod", "turbo_mse", "polar"]


@dataclass
class RunResult:
    scheme: str
    answer: str
    ttft_ms: float          # time-to-first-token in ms
    prefill_ms: float       # prefill (encode) time in ms
    num_context_tokens: int
    num_output_tokens: int
    kv_mb: float            # KV cache size in MB (compressed)
    fp16_mb: float          # KV cache size in MB (FP16 reference)
    vram_ratio: float       # fp16_mb / kv_mb
    attn_mse: float = 0.0   # attention output MSE vs FP16 reference


class TQModelRunner:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        store_dir: str | Path = "./kv_store",
        lib_path: str | Path | None = None,
        device: str = "cuda",
    ) -> None:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from tq_backend.cag_store import CAGStore

        self.device = device
        self.store = CAGStore(store_dir, lib_path)

        print(f"Loading {model_name} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()

        # Qwen2.5-0.5B: 24 layers, num_kv_heads=2, head_dim=64
        cfg = self.model.config
        self.num_layers   = cfg.num_hidden_layers
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim     = cfg.hidden_size // cfg.num_attention_heads
        self.head_shape   = (self.num_kv_heads, self.head_dim)
        print(f"  layers={self.num_layers}  kv_heads={self.num_kv_heads}  head_dim={self.head_dim}")

    # ── offline ────────────────────────────────────────────────────── #

    def precompute_corpus(
        self,
        corpus: dict[str, str],
        schemes: list[Scheme] | None = None,
        overwrite: bool = False,
        max_length: int | None = None,
    ) -> set[str]:
        """Extract and compress KV for each document for all schemes.

        Args:
            max_length: If set, truncate each document to this many tokens
                        before the forward pass.  Useful for long-context
                        benchmarks (e.g. LongBench at 32K tokens) to avoid
                        exceeding the model's context window.

        Returns:
            Set of doc_ids that failed due to CUDA OOM (forward pass too long).
            Callers should skip these docs in downstream evaluation.
        """
        if schemes is None:
            schemes = ["fp16", "turbo_prod", "turbo_mse", "polar"]

        failed: set[str] = set()

        for doc_id, text in corpus.items():
            skip = all(
                self.store.exists(doc_id, l, s)
                for l in range(self.num_layers)
                for s in schemes
            )
            if skip and not overwrite:
                print(f"  [skip] {doc_id} already cached")
                continue

            print(f"  [pack] {doc_id}  ({len(text)} chars) ...", end=" ", flush=True)
            try:
                kv_per_layer = self._extract_kv(text, max_length=max_length)
            except Exception as exc:
                msg = str(exc).lower()
                if "out of memory" in msg or "cuda" in msg:
                    print(f"OOM — skipped")
                    failed.add(doc_id)
                    torch.cuda.empty_cache()
                    continue
                raise  # re-raise non-OOM errors

            total_bytes = 0
            for layer_idx, (k, v) in enumerate(kv_per_layer):
                for scheme in schemes:
                    b = self.store.pack_document(doc_id, layer_idx, k, v, scheme, overwrite)
                    total_bytes += b
            print(f"done  ({total_bytes / 1024:.1f} KB across {len(schemes)} schemes)")

        return failed

    def _extract_kv(
        self,
        text: str,
        max_length: int | None = None,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Forward-pass text, collect per-layer KV tensors.

        Returns list of (key, value) each shaped [S, H_kv, D] fp16 on CUDA.
        """
        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=max_length is not None,
            max_length=max_length,
        ).to(self.device)
        with torch.no_grad():
            # Use model.model() (base transformer, no lm_head) so the logit
            # projection (vocab=152K) is never materialised — avoids OOM at
            # sequences longer than ~17K tokens on a 12 GB GPU.
            out = self.model.model(**tokens, use_cache=True)

        pkv = out.past_key_values
        kv_cache = []

        # transformers >= 4.38: DynamicCache with key_cache / value_cache lists
        if hasattr(pkv, "key_cache"):
            for k_bhsd, v_bhsd in zip(pkv.key_cache, pkv.value_cache):
                k = k_bhsd[0].permute(1, 0, 2).contiguous()  # [S, H, D]
                v = v_bhsd[0].permute(1, 0, 2).contiguous()
                kv_cache.append((k, v))
        # older transformers: tuple of (k, v) pairs per layer
        elif isinstance(pkv, (tuple, list)):
            for layer_kv in pkv:
                k = layer_kv[0][0].permute(1, 0, 2).contiguous()
                v = layer_kv[1][0].permute(1, 0, 2).contiguous()
                kv_cache.append((k, v))
        # custom cache with .layers[].keys / .values
        else:
            for layer_cache in pkv.layers:
                k = layer_cache.keys[0].permute(1, 0, 2).contiguous()
                v = layer_cache.values[0].permute(1, 0, 2).contiguous()
                kv_cache.append((k, v))

        return kv_cache

    # ── online: single-scheme inference ────────────────────────────── #

    def run_inference(
        self,
        query: str,
        doc_ids: list[str],
        scheme: Scheme = "polar",
        max_new_tokens: int = 64,
    ) -> RunResult:
        """Generate answer; return tokens + timing."""
        prompt = self._build_prompt(query, doc_ids)
        tokens = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        num_ctx = tokens["input_ids"].shape[1]

        # Compute KV size
        total_tokens = sum(
            self._doc_token_count(did) for did in doc_ids
        )
        kv_mb = sum(
            self.store.vram_bytes(scheme, total_tokens, self.head_shape)
            for _ in range(self.num_layers)
        ) / 1024**2
        fp16_mb = sum(
            self.store.vram_bytes("fp16", total_tokens, self.head_shape)
            for _ in range(self.num_layers)
        ) / 1024**2

        torch.cuda.synchronize()
        t_start = time.perf_counter()

        with torch.no_grad():
            output = self.model.generate(
                **tokens,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t_start) * 1e3

        new_tokens = output[0][num_ctx:]
        answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        # TTFT ≈ total_time / num_new_tokens (single-token decode latency approx)
        # More precisely: prefill + first decode step
        ttft_ms = elapsed_ms / max(len(new_tokens), 1)

        return RunResult(
            scheme=scheme,
            answer=answer,
            ttft_ms=ttft_ms,
            prefill_ms=elapsed_ms,
            num_context_tokens=num_ctx,
            num_output_tokens=len(new_tokens),
            kv_mb=kv_mb,
            fp16_mb=fp16_mb,
            vram_ratio=fp16_mb / max(kv_mb, 1e-6),
        )

    # ── benchmark: compare all schemes ─────────────────────────────── #

    def benchmark_query(
        self,
        query: str,
        doc_ids: list[str],
        schemes: list[Scheme] | None = None,
        max_new_tokens: int = 64,
        warmup: bool = True,
    ) -> list[RunResult]:
        if schemes is None:
            schemes = ["fp16", "turbo_prod", "turbo_mse", "polar"]

        # Warmup with fp16
        if warmup:
            self.run_inference(query, doc_ids, "fp16", max_new_tokens=8)

        results = []
        for scheme in schemes:
            r = self.run_inference(query, doc_ids, scheme, max_new_tokens)
            results.append(r)
            print(f"  {scheme:12s}  TTFT={r.ttft_ms:.1f}ms  VRAM={r.kv_mb:.2f}MB  ans={r.answer[:60]!r}")
        return results

    # ── attention MSE comparison ────────────────────────────────────── #

    def compare_attention_mse(
        self,
        text: str,
        layer_indices: int | list[int] = 0,
        schemes: list[Scheme] | None = None,
    ) -> dict[str, list[float]]:
        """Compare attention MSE vs FP16 for one or more layers.

        Returns dict[scheme → list of MSE per layer].
        """
        if schemes is None:
            schemes = ["turbo_prod", "turbo_mse", "polar"]
        if isinstance(layer_indices, int):
            layer_indices = [layer_indices]

        doc_id = f"_mse_test_{hash(text) & 0xFFFF}"
        self.precompute_corpus({doc_id: text}, schemes=["fp16"] + schemes)

        H, D = self.head_shape
        query = torch.randn(1, H, D, device=self.device, dtype=torch.float16)

        mse: dict[str, list[float]] = {s: [] for s in schemes}
        for layer_idx in layer_indices:
            pool_fp16, slots_fp16, N = self.store.load_document(
                doc_id, layer_idx, "fp16", self.head_shape)
            ref = self.store.fused_attention(
                query, pool_fp16, slots_fp16, N, "fp16", self.head_shape)
            for scheme in schemes:
                pool, slots, n = self.store.load_document(
                    doc_id, layer_idx, scheme, self.head_shape)
                out = self.store.fused_attention(
                    query, pool, slots, n, scheme, self.head_shape)
                mse[scheme].append(float(torch.mean((ref.float() - out.float()) ** 2)))
        return mse

    # ── helpers ─────────────────────────────────────────────────────── #

    def _build_prompt(self, query: str, doc_ids: list[str]) -> str:
        # Simple RAG prompt (no actual doc text needed since KV is pre-computed)
        return f"Question: {query}\nAnswer:"

    def _doc_token_count(self, doc_id: str) -> int:
        try:
            import json
            meta_p = self.store._meta_path(doc_id, 0, "fp16")
            if meta_p.exists():
                meta = torch.load(meta_p, weights_only=False)
                return int(meta["num_tokens"])
        except Exception:
            pass
        return 512  # fallback estimate
