"""tq_backend — TurboQuant attention backend for Qwen2.5.

Cache-Augmented Generation with compressed KV caches.
Supports two compression schemes:
  fp16       — reference, no compression
  turbo_prod — K=3b+1b residual / V=4b  (~3.8× VRAM reduction)
  turbo_mse  — INT4 MSE-optimised         (~3.6× VRAM reduction)

CAG:
  Offline: precompute_corpus() → per-layer compressed KV stored to disk
  Online:  model.generate() skips prefill; loads KV from disk per layer
"""
from tq_backend.model_runner import TQModelRunner
from tq_backend.cag_store import CAGStore

__all__ = ["TQModelRunner", "CAGStore"]
