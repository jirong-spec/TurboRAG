"""cag_store.py — Per-layer KV disk store for CAG.

Offline: CAGStore.pack_document(doc_id, layer_idx, key, value, scheme)
Online:  CAGStore.load_document(doc_id, layer_idx, scheme) → (pool, slots, N)
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Literal

import torch

from .turboquant_wrapper import TurboQuantWrapper, TQConfig

_MANIFEST_FILE = "manifest.json"

Scheme = Literal["fp16", "turbo_prod", "turbo_mse"]


class CAGStore:
    def __init__(self, store_dir: str | Path, lib_path: str | Path | None = None) -> None:
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.tq = TurboQuantWrapper(lib_path)
        self._manifest_path = self.store_dir / _MANIFEST_FILE

    # ── manifest helpers ──────────────────────────────────────────── #

    def _load_manifest(self) -> dict:
        if self._manifest_path.exists():
            try:
                return json.loads(self._manifest_path.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        return {"v": 1, "entries": {}}

    def _save_manifest(self, manifest: dict) -> None:
        # Atomic write: write to .tmp then rename so a partial write never
        # leaves the manifest in a corrupted state.
        tmp = self._manifest_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(manifest, indent=2))
        tmp.replace(self._manifest_path)

    def _doc_hash(self, doc_id: str) -> str:
        """Full 64-char SHA-256 hex digest of the doc_id string."""
        return hashlib.sha256(doc_id.encode()).hexdigest()

    def _update_manifest(
        self, doc_id: str, layer: int, scheme: Scheme, num_tokens: int
    ) -> None:
        manifest = self._load_manifest()  # reload to pick up concurrent writes
        doc_hash = self._doc_hash(doc_id)
        layer_key = f"{layer}::{scheme}"
        entries = manifest.setdefault("entries", {})
        entry = entries.setdefault(doc_id, {"sha256": doc_hash, "layers": {}})
        entry["sha256"] = doc_hash
        entry["layers"][layer_key] = {
            "key": self._key(doc_id, layer, scheme),
            "num_tokens": num_tokens,
            "stored_at": time.time(),
        }
        self._save_manifest(manifest)

    def _verify_manifest(self, doc_id: str, layer: int, scheme: Scheme) -> None:
        """Raise RuntimeError if the manifest entry for doc+layer+scheme is inconsistent.

        Missing manifest entry is allowed (backward compat with pre-manifest stores).
        """
        manifest = self._load_manifest()
        entries = manifest.get("entries", {})
        if doc_id not in entries:
            return  # pre-manifest store — skip integrity check
        entry = entries[doc_id]
        stored_hash = entry.get("sha256", "")
        computed_hash = self._doc_hash(doc_id)
        if stored_hash and stored_hash != computed_hash:
            raise RuntimeError(
                f"Manifest integrity error for doc_id={doc_id!r}: "
                f"stored SHA-256 {stored_hash[:8]}… != computed {computed_hash[:8]}…"
            )
        layer_key = f"{layer}::{scheme}"
        layer_entry = entry.get("layers", {}).get(layer_key)
        if layer_entry:
            expected_key = self._key(doc_id, layer, scheme)
            if layer_entry.get("key") != expected_key:
                raise RuntimeError(
                    f"Manifest key mismatch for doc_id={doc_id!r} "
                    f"layer={layer} scheme={scheme}: "
                    f"stored {layer_entry.get('key')!r} != computed {expected_key!r}"
                )

    # ── path helpers ──────────────────────────────────────────────── #

    def _key(self, doc_id: str, layer: int, scheme: Scheme) -> str:
        # 16-char SHA-256 prefix for filenames (64 bits, vs old 48-bit MD5[:12]).
        # Full hash lives in manifest.json for collision-safe verification.
        return f"{self._doc_hash(doc_id)[:16]}_L{layer:02d}_{scheme}"

    def _pool_path(self, doc_id: str, layer: int, scheme: Scheme) -> Path:
        return self.store_dir / f"{self._key(doc_id, layer, scheme)}.bin"

    def _meta_path(self, doc_id: str, layer: int, scheme: Scheme) -> Path:
        return self.store_dir / f"{self._key(doc_id, layer, scheme)}.meta"

    def exists(self, doc_id: str, layer: int, scheme: Scheme) -> bool:
        return self._pool_path(doc_id, layer, scheme).exists()

    # ── offline: pack ─────────────────────────────────────────────── #

    def pack_document(
        self,
        doc_id: str,
        layer: int,
        key: torch.Tensor,    # [N, H, D] fp16 CUDA
        value: torch.Tensor,  # [N, H, D] fp16 CUDA
        scheme: Scheme = "turbo_prod",
        overwrite: bool = False,
    ) -> int:
        """Compress and save one layer's KV. Returns bytes written."""
        pool_p = self._pool_path(doc_id, layer, scheme)
        meta_p = self._meta_path(doc_id, layer, scheme)
        if pool_p.exists() and not overwrite:
            return pool_p.stat().st_size

        N = key.shape[0]
        slots = torch.arange(N, dtype=torch.int32, device=key.device)

        if scheme == "fp16":
            # Store raw fp16 K and V concatenated
            kv = torch.cat([key.reshape(-1), value.reshape(-1)]).cpu()
            torch.save({"num_tokens": N, "kv": kv, "shape": list(key.shape)}, pool_p)
            torch.save({"num_tokens": N, "slots": slots.cpu()}, meta_p)
            self._update_manifest(doc_id, layer, scheme, N)
            return pool_p.stat().st_size

        cfg = self._make_cfg(key)
        if scheme == "turbo_prod":
            layout = self.tq.make_layout_for(cfg)
            pool = self.tq.alloc_page_pool(N, layout, cfg)
            self.tq.pack(key, value, slots, pool, layout, cfg)
        elif scheme == "turbo_mse":
            layout = self.tq.make_mse_layout_for(cfg)
            pool = self.tq.alloc_mse_pool(N, layout, cfg)
            self.tq.mse_pack(key, value, slots, pool, layout, cfg)
        else:
            raise ValueError(f"Unknown scheme: {scheme}")

        torch.cuda.synchronize()
        pool_cpu = pool.cpu()
        with pool_p.open("wb") as f:
            f.write(pool_cpu.numpy().tobytes())
        torch.save({"num_tokens": N, "slots": slots.cpu()}, meta_p)
        self._update_manifest(doc_id, layer, scheme, N)
        return pool_p.stat().st_size

    # ── online: load ──────────────────────────────────────────────── #

    def load_document(
        self,
        doc_id: str,
        layer: int,
        scheme: Scheme = "turbo_prod",
        head_shape: tuple[int, int] = (2, 64),  # (num_kv_heads, head_dim)
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Load compressed KV to CUDA. Returns (pool_or_kv, slots, num_tokens)."""
        pool_p = self._pool_path(doc_id, layer, scheme)
        meta_p = self._meta_path(doc_id, layer, scheme)
        if not pool_p.exists():
            raise FileNotFoundError(f"No pre-computed KV: doc={doc_id} layer={layer} scheme={scheme}")

        self._verify_manifest(doc_id, layer, scheme)

        meta = torch.load(meta_p, weights_only=False)
        N = int(meta["num_tokens"])
        slots = meta["slots"]

        if scheme == "fp16":
            data = torch.load(pool_p, weights_only=False)
            kv = data["kv"].to("cuda", non_blocking=True)
            H, D = head_shape
            key   = kv[:N * H * D].reshape(N, H, D).contiguous()
            value = kv[N * H * D:].reshape(N, H, D).contiguous()
            return (key, value), slots.to("cuda"), N

        cfg = self._make_cfg_from_shape(head_shape)
        if scheme == "turbo_prod":
            layout = self.tq.make_layout_for(cfg)
            pool_bytes = self.tq.quant_bytes(N, layout, cfg)
        elif scheme == "turbo_mse":
            layout = self.tq.make_mse_layout_for(cfg)
            pool_bytes = self.tq.mse_bytes(N, layout, cfg)
        else:
            raise ValueError(f"Unknown scheme: {scheme}")

        raw = pool_p.read_bytes()
        pool_cpu = torch.frombuffer(bytearray(raw[:pool_bytes]), dtype=torch.uint8).clone()
        pool = pool_cpu.to("cuda", non_blocking=True)
        return pool, slots.to("cuda"), N

    def load_as_kv_fp16(
        self,
        doc_id: str,
        layer: int,
        scheme: Scheme,
        head_shape: tuple[int, int] = (2, 64),
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Load and dequant one layer → (key, value) each [N, H, D] fp16 CUDA."""
        pool, slots, N = self.load_document(doc_id, layer, scheme, head_shape)

        if scheme == "fp16":
            key, value = pool  # already (key, value) tensors
            return key, value

        cfg = self._make_cfg_from_shape(head_shape)
        H, D = head_shape
        out_key   = torch.zeros(N, H, D, device="cuda", dtype=torch.float16)
        out_value = torch.zeros(N, H, D, device="cuda", dtype=torch.float16)

        if scheme == "turbo_prod":
            layout = self.tq.make_layout_for(cfg)
            self.tq.dequant(pool, slots, out_key, out_value, layout, cfg)
        elif scheme == "turbo_mse":
            layout = self.tq.make_mse_layout_for(cfg)
            self.tq.mse_dequant(pool, slots, out_key, out_value, layout, cfg)
        return out_key, out_value

    def build_dynamic_cache(
        self,
        doc_id: str,
        num_layers: int,
        scheme: Scheme,
        head_shape: tuple[int, int] = (2, 64),
    ):
        """Build a HF DynamicCache from disk-loaded (dequanted) KV for all layers.

        Returns (cache, doc_num_tokens) where cache can be passed to model.generate().
        Key/Value tensors are reshaped to [B=1, H_kv, S, D] as required by transformers.
        """
        from transformers import DynamicCache
        cache = DynamicCache()
        doc_len = None
        for layer_idx in range(num_layers):
            if not self.exists(doc_id, layer_idx, scheme):
                continue
            k, v = self.load_as_kv_fp16(doc_id, layer_idx, scheme, head_shape)
            # k: [N, H, D]  →  [1, H, N, D]
            k = k.permute(1, 0, 2).unsqueeze(0).contiguous()
            v = v.permute(1, 0, 2).unsqueeze(0).contiguous()
            cache.update(k, v, layer_idx)
            if doc_len is None:
                doc_len = k.shape[2]
        return cache, (doc_len or 0)

    def fused_attention(
        self,
        query: torch.Tensor,   # [1, H, D] fp16 CUDA
        pool: torch.Tensor,
        slots: torch.Tensor,
        N: int,
        scheme: Scheme,
        head_shape: tuple[int, int] = (2, 64),
    ) -> torch.Tensor:
        """Run compressed-KV attention; returns [1, H, D] fp16."""
        cfg = self._make_cfg_from_shape(head_shape)
        output = torch.zeros_like(query)

        if scheme == "fp16":
            key, value = pool  # (key [N,H,D], value [N,H,D])
            H, D = head_shape
            q = query.permute(1, 0, 2)   # [H, 1, D]
            k = key.permute(1, 0, 2)     # [H, N, D]
            v = value.permute(1, 0, 2)   # [H, N, D]
            scale = D ** -0.5
            scores = torch.bmm(q, k.transpose(1, 2)) * scale  # [H, 1, N]
            attn = torch.softmax(scores, dim=-1)
            out = torch.bmm(attn, v)                          # [H, 1, D]
            return out.permute(1, 0, 2)

        if scheme == "turbo_prod":
            layout = self.tq.make_layout_for(cfg)
            self.tq.fused_attn_output(query, pool, slots, output, layout, cfg, 1, N)
        elif scheme == "turbo_mse":
            layout = self.tq.make_mse_layout_for(cfg)
            self.tq.mse_fused_attn_output(query, pool, slots, output, layout, cfg, 1, N)
        return output

    # ── cfg helpers ───────────────────────────────────────────────── #

    def _make_cfg(self, key: torch.Tensor) -> TQConfig:
        H, D = key.shape[1], key.shape[2]
        return self._make_cfg_from_shape((H, D))

    def _make_cfg_from_shape(self, head_shape: tuple[int, int]) -> TQConfig:
        H, D = head_shape
        cfg = self.tq.default_config()
        cfg.num_kv_heads = H
        cfg.head_dim = D
        return cfg

    # ── VRAM estimate ─────────────────────────────────────────────── #

    def vram_bytes(self, scheme: Scheme, num_tokens: int, head_shape: tuple[int, int]) -> int:
        cfg = self._make_cfg_from_shape(head_shape)
        if scheme == "fp16":
            return self.tq.fp16_bytes(num_tokens, cfg)
        elif scheme == "turbo_prod":
            return self.tq.quant_bytes(num_tokens, self.tq.make_layout_for(cfg), cfg)
        elif scheme == "turbo_mse":
            return self.tq.mse_bytes(num_tokens, self.tq.make_mse_layout_for(cfg), cfg)
        return 0
