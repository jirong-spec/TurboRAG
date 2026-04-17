"""TurboQuant Python wrapper — turbo_prod and turbo_mse kernels.

Exposes pack / dequant / fused-attention for both quantisation schemes
via a tensor-friendly API.  All ctypes casting is handled internally;
callers work with torch.Tensor and plain Python ints.
"""
from __future__ import annotations

import ctypes
from ctypes import c_float, c_int, c_int32, c_size_t, c_uint8, c_void_p
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LIB_PATH = ROOT / "build" / "libturboquant.so"


# ─────────────────────────────────────────────────────────────────── #
# C-compatible structs (must match headers exactly)                   #
# ─────────────────────────────────────────────────────────────────── #

class TQConfig(ctypes.Structure):
    _fields_ = [
        ("block_size",     c_int),
        ("num_kv_heads",   c_int),
        ("head_dim",       c_int),
        ("nbits",          c_int),
        ("group_size",     c_int),
        ("page_alignment", c_int),
        ("scale_type",     c_int),
        ("quant_mode",     c_int),
    ]


class TQTurboProdPageLayout(ctypes.Structure):
    """Matches tq_turbo_prod.cuh :: TQTurboProdPageLayout."""
    _fields_ = [
        ("page_size_bytes",            c_size_t),
        ("k3_codes_offset",            c_size_t),
        ("k_residual_offset",          c_size_t),
        ("k_residual_scales_offset",   c_size_t),
        ("k_scales_offset",            c_size_t),
        ("v4_codes_offset",            c_size_t),
        ("v_scales_offset",            c_size_t),
        ("k3_bytes_per_token_head",    c_int),
        ("kres_bytes_per_token_head",  c_int),
        ("v4_bytes_per_token_head",    c_int),
        ("scale_bytes_per_token_head", c_int),
    ]


class TQTurbomsePageLayout(ctypes.Structure):
    """Matches tq_turbo_mse_layout.h :: TQTurbomsePageLayout."""
    _fields_ = [
        ("code_bytes_per_token_head", c_size_t),
        ("norm_bytes_per_token_head", c_size_t),
        ("k_codes_offset",           c_size_t),
        ("v_codes_offset",           c_size_t),
        ("k_norms_offset",           c_size_t),
        ("v_norms_offset",           c_size_t),
        ("page_size_bytes",          c_size_t),
    ]


# ─────────────────────────────────────────────────────────────────── #
# Wrapper                                                             #
# ─────────────────────────────────────────────────────────────────── #

class TurboQuantWrapper:
    """Thin wrapper around libturboquant.so exposing turbo_prod and turbo_mse.

    turbo_prod  — K=3-bit Lloyd-Max + 1-bit QJL residual, V=4-bit.
                  Optimised for maximum throughput and compression (~15–16×).
    turbo_mse   — INT4, loss function minimises mean-squared error.
                  Optimised for reconstruction fidelity (~8×).
    """

    def __init__(self, lib_path: str | Path | None = None) -> None:
        path = Path(lib_path) if lib_path else DEFAULT_LIB_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"libturboquant.so not found at {path}. "
                "Run:  cmake --build build  first."
            )
        self.lib_path = path
        self._lib = ctypes.CDLL(str(path))
        self._bind_symbols()

    # ── Symbol binding ────────────────────────────────────────────── #

    def _bind_symbols(self) -> None:
        L = self._lib

        # config / layout factories
        L.tq_make_default_config.argtypes  = [ctypes.POINTER(TQConfig)]
        L.tq_make_default_config.restype   = c_int

        L.tq_make_turbo_prod_layout.argtypes = [
            ctypes.POINTER(TQConfig),
            ctypes.POINTER(TQTurboProdPageLayout),
        ]
        L.tq_make_turbo_prod_layout.restype = c_int

        L.tq_make_turbo_mse_layout.argtypes = [
            ctypes.POINTER(TQConfig),
            ctypes.POINTER(TQTurbomsePageLayout),
        ]
        L.tq_make_turbo_mse_layout.restype = c_int

        # ── turbo_prod kernels ───────────────────────────────────── #
        L.tq_launch_turbo_prod_pack_kv.argtypes = [
            c_void_p, c_void_p,
            ctypes.POINTER(c_int32), ctypes.POINTER(c_uint8),
            ctypes.POINTER(TQTurboProdPageLayout), ctypes.POINTER(TQConfig),
            c_int, c_void_p,
        ]
        L.tq_launch_turbo_prod_pack_kv.restype = c_int

        L.tq_launch_turbo_prod_dequant_kv.argtypes = [
            ctypes.POINTER(c_uint8), ctypes.POINTER(c_int32),
            c_void_p, c_void_p,
            ctypes.POINTER(TQTurboProdPageLayout), ctypes.POINTER(TQConfig),
            c_int, c_void_p,
        ]
        L.tq_launch_turbo_prod_dequant_kv.restype = c_int

        L.tq_launch_turbo_prod_fused_attention_logits.argtypes = [
            c_void_p,
            ctypes.POINTER(c_uint8), ctypes.POINTER(c_int32),
            ctypes.POINTER(c_float),
            ctypes.POINTER(TQTurboProdPageLayout), ctypes.POINTER(TQConfig),
            c_int, c_int, c_void_p,
        ]
        L.tq_launch_turbo_prod_fused_attention_logits.restype = c_int

        L.tq_launch_turbo_prod_fused_attention_output.argtypes = [
            c_void_p,
            ctypes.POINTER(c_uint8), ctypes.POINTER(c_int32),
            c_void_p,
            ctypes.POINTER(TQTurboProdPageLayout), ctypes.POINTER(TQConfig),
            c_int, c_int, c_void_p,
        ]
        L.tq_launch_turbo_prod_fused_attention_output.restype = c_int

        # ── turbo_mse kernels ────────────────────────────────────── #
        L.tq_launch_turbo_mse_pack_kv.argtypes = [
            c_void_p, c_void_p,
            ctypes.POINTER(c_int32), ctypes.POINTER(c_uint8),
            ctypes.POINTER(TQTurbomsePageLayout), ctypes.POINTER(TQConfig),
            c_int, c_void_p,
        ]
        L.tq_launch_turbo_mse_pack_kv.restype = c_int

        L.tq_launch_turbo_mse_dequant_kv.argtypes = [
            ctypes.POINTER(c_uint8), ctypes.POINTER(c_int32),
            c_void_p, c_void_p,
            ctypes.POINTER(TQTurbomsePageLayout), ctypes.POINTER(TQConfig),
            c_int, c_void_p,
        ]
        L.tq_launch_turbo_mse_dequant_kv.restype = c_int

        L.tq_launch_turbo_mse_fused_attention_output.argtypes = [
            c_void_p,
            ctypes.POINTER(c_uint8), ctypes.POINTER(c_int32),
            c_void_p,
            ctypes.POINTER(TQTurbomsePageLayout), ctypes.POINTER(TQConfig),
            c_int, c_int, c_void_p,
        ]
        L.tq_launch_turbo_mse_fused_attention_output.restype = c_int

    # ── Config / layout ───────────────────────────────────────────── #

    def default_config(self) -> TQConfig:
        cfg = TQConfig()
        if self._lib.tq_make_default_config(ctypes.byref(cfg)) != 0:
            raise RuntimeError("tq_make_default_config failed")
        return cfg

    def make_layout_for(self, cfg: TQConfig) -> TQTurboProdPageLayout:
        layout = TQTurboProdPageLayout()
        if self._lib.tq_make_turbo_prod_layout(ctypes.byref(cfg), ctypes.byref(layout)) != 0:
            raise RuntimeError("tq_make_turbo_prod_layout failed")
        return layout

    def make_mse_layout_for(self, cfg: TQConfig) -> TQTurbomsePageLayout:
        layout = TQTurbomsePageLayout()
        if self._lib.tq_make_turbo_mse_layout(ctypes.byref(cfg), ctypes.byref(layout)) != 0:
            raise RuntimeError("tq_make_turbo_mse_layout failed")
        return layout

    # ── Pointer helpers ───────────────────────────────────────────── #

    @staticmethod
    def _vp(t: torch.Tensor) -> c_void_p:
        return c_void_p(t.data_ptr())

    @staticmethod
    def _i32p(t: torch.Tensor) -> ctypes.POINTER(c_int32):
        return ctypes.cast(c_void_p(t.data_ptr()), ctypes.POINTER(c_int32))

    @staticmethod
    def _u8p(t: torch.Tensor) -> ctypes.POINTER(c_uint8):
        return ctypes.cast(c_void_p(t.data_ptr()), ctypes.POINTER(c_uint8))

    @staticmethod
    def _f32p(t: torch.Tensor) -> ctypes.POINTER(c_float):
        return ctypes.cast(c_void_p(t.data_ptr()), ctypes.POINTER(c_float))

    # ─────────────────────────────────────────────────────────────── #
    # turbo_prod — K=3b+1b residual, V=4b  (~15–16× vs FP16)        #
    # ─────────────────────────────────────────────────────────────── #

    def pack(
        self,
        key:          torch.Tensor,   # [tokens, H, D]  fp16  contiguous
        value:        torch.Tensor,   # [tokens, H, D]  fp16  contiguous
        slot_mapping: torch.Tensor,   # [tokens]        int32
        page_pool:    torch.Tensor,   # [pool_bytes]    uint8
        layout:       TQTurboProdPageLayout,
        cfg:          TQConfig,
        stream:       int = 0,
    ) -> None:
        rc = self._lib.tq_launch_turbo_prod_pack_kv(
            self._vp(key), self._vp(value),
            self._i32p(slot_mapping), self._u8p(page_pool),
            ctypes.byref(layout), ctypes.byref(cfg),
            key.shape[0], c_void_p(stream),
        )
        if rc != 0:
            raise RuntimeError(f"turbo_prod pack_kv failed (rc={rc})")

    def dequant(
        self,
        page_pool:    torch.Tensor,   # [pool_bytes]    uint8
        slot_mapping: torch.Tensor,   # [tokens]        int32
        out_key:      torch.Tensor,   # [tokens, H, D]  fp16  contiguous
        out_value:    torch.Tensor,   # [tokens, H, D]  fp16  contiguous
        layout:       TQTurboProdPageLayout,
        cfg:          TQConfig,
        stream:       int = 0,
    ) -> None:
        rc = self._lib.tq_launch_turbo_prod_dequant_kv(
            self._u8p(page_pool), self._i32p(slot_mapping),
            self._vp(out_key), self._vp(out_value),
            ctypes.byref(layout), ctypes.byref(cfg),
            out_key.shape[0], c_void_p(stream),
        )
        if rc != 0:
            raise RuntimeError(f"turbo_prod dequant_kv failed (rc={rc})")

    def fused_attn_logits(
        self,
        query:         torch.Tensor,  # [num_q, H, D]               fp16
        page_pool:     torch.Tensor,  # [pool_bytes]                uint8
        slot_mapping:  torch.Tensor,  # [num_kv_tokens]             int32
        logits:        torch.Tensor,  # [num_q * H * num_kv_tokens] float32
        layout:        TQTurboProdPageLayout,
        cfg:           TQConfig,
        num_queries:   int,
        num_kv_tokens: int,
        stream:        int = 0,
    ) -> None:
        """Logit convention: <q, k>  (no 1/sqrt(D) — matches kernel)."""
        rc = self._lib.tq_launch_turbo_prod_fused_attention_logits(
            self._vp(query),
            self._u8p(page_pool), self._i32p(slot_mapping),
            self._f32p(logits),
            ctypes.byref(layout), ctypes.byref(cfg),
            num_queries, num_kv_tokens, c_void_p(stream),
        )
        if rc != 0:
            raise RuntimeError(f"turbo_prod fused_attn_logits failed (rc={rc})")

    def fused_attn_output(
        self,
        query:         torch.Tensor,  # [num_q, H, D]  fp16
        page_pool:     torch.Tensor,  # [pool_bytes]   uint8
        slot_mapping:  torch.Tensor,  # [num_kv_tokens] int32
        output:        torch.Tensor,  # [num_q, H, D]  fp16  contiguous
        layout:        TQTurboProdPageLayout,
        cfg:           TQConfig,
        num_queries:   int,
        num_kv_tokens: int,
        stream:        int = 0,
    ) -> None:
        """Online-softmax FlashAttention over compressed pool; no KV rematerialisation."""
        rc = self._lib.tq_launch_turbo_prod_fused_attention_output(
            self._vp(query),
            self._u8p(page_pool), self._i32p(slot_mapping),
            self._vp(output),
            ctypes.byref(layout), ctypes.byref(cfg),
            num_queries, num_kv_tokens, c_void_p(stream),
        )
        if rc != 0:
            raise RuntimeError(f"turbo_prod fused_attn_output failed (rc={rc})")

    # ─────────────────────────────────────────────────────────────── #
    # turbo_mse — INT4 MSE-optimised  (~8× vs FP16)                  #
    # ─────────────────────────────────────────────────────────────── #

    def mse_pack(
        self,
        key:          torch.Tensor,   # [tokens, H, D]  fp16  contiguous
        value:        torch.Tensor,   # [tokens, H, D]  fp16  contiguous
        slot_mapping: torch.Tensor,   # [tokens]        int32
        page_pool:    torch.Tensor,   # [pool_bytes]    uint8
        layout:       TQTurbomsePageLayout,
        cfg:          TQConfig,
        stream:       int = 0,
    ) -> None:
        rc = self._lib.tq_launch_turbo_mse_pack_kv(
            self._vp(key), self._vp(value),
            self._i32p(slot_mapping), self._u8p(page_pool),
            ctypes.byref(layout), ctypes.byref(cfg),
            key.shape[0], c_void_p(stream),
        )
        if rc != 0:
            raise RuntimeError(f"turbo_mse pack_kv failed (rc={rc})")

    def mse_dequant(
        self,
        page_pool:    torch.Tensor,   # [pool_bytes]    uint8
        slot_mapping: torch.Tensor,   # [tokens]        int32
        out_key:      torch.Tensor,   # [tokens, H, D]  fp16  contiguous
        out_value:    torch.Tensor,   # [tokens, H, D]  fp16  contiguous
        layout:       TQTurbomsePageLayout,
        cfg:          TQConfig,
        stream:       int = 0,
    ) -> None:
        rc = self._lib.tq_launch_turbo_mse_dequant_kv(
            self._u8p(page_pool), self._i32p(slot_mapping),
            self._vp(out_key), self._vp(out_value),
            ctypes.byref(layout), ctypes.byref(cfg),
            out_key.shape[0], c_void_p(stream),
        )
        if rc != 0:
            raise RuntimeError(f"turbo_mse dequant_kv failed (rc={rc})")

    def mse_fused_attn_output(
        self,
        query:         torch.Tensor,  # [num_q, H, D]   fp16
        page_pool:     torch.Tensor,  # [pool_bytes]    uint8
        slot_mapping:  torch.Tensor,  # [num_kv_tokens] int32
        output:        torch.Tensor,  # [num_q, H, D]   fp16  contiguous
        layout:        TQTurbomsePageLayout,
        cfg:           TQConfig,
        num_queries:   int,
        num_kv_tokens: int,
        stream:        int = 0,
    ) -> None:
        rc = self._lib.tq_launch_turbo_mse_fused_attention_output(
            self._vp(query),
            self._u8p(page_pool), self._i32p(slot_mapping),
            self._vp(output),
            ctypes.byref(layout), ctypes.byref(cfg),
            num_queries, num_kv_tokens, c_void_p(stream),
        )
        if rc != 0:
            raise RuntimeError(f"turbo_mse fused_attn_output failed (rc={rc})")

    # ─────────────────────────────────────────────────────────────── #
    # Allocation helpers                                              #
    # ─────────────────────────────────────────────────────────────── #

    def alloc_page_pool(
        self,
        num_tokens: int,
        layout: TQTurboProdPageLayout,
        cfg:    TQConfig,
    ) -> torch.Tensor:
        num_blocks = (num_tokens + cfg.block_size - 1) // cfg.block_size
        return torch.zeros(num_blocks * layout.page_size_bytes,
                           device="cuda", dtype=torch.uint8)

    def alloc_mse_pool(
        self,
        num_tokens: int,
        layout: TQTurbomsePageLayout,
        cfg:    TQConfig,
    ) -> torch.Tensor:
        num_blocks = (num_tokens + cfg.block_size - 1) // cfg.block_size
        return torch.zeros(num_blocks * layout.page_size_bytes,
                           device="cuda", dtype=torch.uint8)

    def fp16_bytes(self, num_tokens: int, cfg: TQConfig) -> int:
        """Bytes for both K and V in FP16."""
        return num_tokens * cfg.num_kv_heads * cfg.head_dim * 2 * 2

    def quant_bytes(self, num_tokens: int, layout: TQTurboProdPageLayout, cfg: TQConfig) -> int:
        num_blocks = (num_tokens + cfg.block_size - 1) // cfg.block_size
        return num_blocks * layout.page_size_bytes

    def mse_bytes(self, num_tokens: int, layout: TQTurbomsePageLayout, cfg: TQConfig) -> int:
        num_blocks = (num_tokens + cfg.block_size - 1) // cfg.block_size
        return num_blocks * layout.page_size_bytes

    # ─────────────────────────────────────────────────────────────── #
    # Utility                                                         #
    # ─────────────────────────────────────────────────────────────── #

    @staticmethod
    def compute_mse(a: torch.Tensor, b: torch.Tensor) -> float:
        """Element-wise MSE between two tensors (promotes to float32)."""
        return torch.mean((a.float() - b.float()) ** 2).item()

    def summary(self) -> dict:
        cfg    = self.default_config()
        prod   = self.make_layout_for(cfg)
        fp16_b = cfg.block_size * cfg.num_kv_heads * cfg.head_dim * 2 * 2
        try:
            mse_l  = self.make_mse_layout_for(cfg)
            mse_ratio = round(fp16_b / mse_l.page_size_bytes, 2)
        except Exception:
            mse_ratio = "N/A"
        return {
            "lib_path": str(self.lib_path),
            "config": {
                "block_size":   cfg.block_size,
                "num_kv_heads": cfg.num_kv_heads,
                "head_dim":     cfg.head_dim,
                "nbits":        cfg.nbits,
            },
            "compression_ratio_vs_fp16":      round(fp16_b / prod.page_size_bytes, 2),
            "mse_compression_ratio_vs_fp16":  mse_ratio,
        }
