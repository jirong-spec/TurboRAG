"""Pytest suite for tq_backend — no GPU required for most tests.

GPU-dependent tests are marked with @pytest.mark.skipif and skipped
when torch.cuda.is_available() is False or libturboquant.so is absent.

Run from repo root:
    pytest tests/test_cag.py -v
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch


# ── helpers ───────────────────────────────────────────────────────────────── #

_LIB = Path(__file__).resolve().parent.parent / "build" / "libturboquant.so"
_HAS_LIB = _LIB.exists()
_HAS_CUDA = torch.cuda.is_available()

_needs_lib  = pytest.mark.skipif(not _HAS_LIB,  reason="libturboquant.so not built")
_needs_cuda = pytest.mark.skipif(not _HAS_CUDA, reason="CUDA not available")


def _make_store(tmp_path: Path):
    """Return a CAGStore with a fully mocked TurboQuantWrapper."""
    from tq_backend.cag_store import CAGStore

    with patch("tq_backend.cag_store.TurboQuantWrapper"):
        store = CAGStore(tmp_path, lib_path="__mock__")
    store.tq = MagicMock()
    return store


def _fp16_kv(N=4, H=2, D=8) -> tuple[torch.Tensor, torch.Tensor]:
    key = torch.zeros(N, H, D, dtype=torch.float16)
    val = torch.zeros(N, H, D, dtype=torch.float16)
    return key, val


# ── 1. Package import smoke test ──────────────────────────────────────────── #

def test_cag_store_importable():
    """CAGStore is importable as a package member — no sys.path magic needed."""
    from tq_backend.cag_store import CAGStore  # noqa: F401
    from tq_backend import CAGStore as CAGStore2  # noqa: F401


def test_no_sys_path_insertion():
    """cag_store.py must not insert anything into sys.path after our fix."""
    import sys
    tq_backend_dir = str(Path(__file__).resolve().parent.parent / "tq_backend")
    # Fresh import (may already be cached; that's fine — just verify path not polluted)
    import tq_backend.cag_store  # noqa: F401
    assert tq_backend_dir not in sys.path, (
        f"cag_store still inserts {tq_backend_dir!r} into sys.path"
    )


# ── 2. SHA-256 hash helpers ───────────────────────────────────────────────── #

def test_doc_hash_is_full_sha256(tmp_path):
    store = _make_store(tmp_path)
    doc_id = "my_document_v1"
    expected = hashlib.sha256(doc_id.encode()).hexdigest()
    assert store._doc_hash(doc_id) == expected
    assert len(store._doc_hash(doc_id)) == 64


def test_filename_key_uses_sha256_prefix(tmp_path):
    store = _make_store(tmp_path)
    doc_id = "corpus/chunk_007"
    key = store._key(doc_id, layer=3, scheme="turbo_mse")
    prefix = hashlib.sha256(doc_id.encode()).hexdigest()[:16]
    assert key.startswith(prefix), f"key={key!r} should start with sha256[:16]={prefix!r}"
    assert "L03" in key
    assert "turbo_mse" in key


def test_filename_key_longer_than_old_md5(tmp_path):
    """New key prefix (16 chars) is longer than old MD5[:12] prefix."""
    store = _make_store(tmp_path)
    key = store._key("any_doc", layer=0, scheme="fp16")
    prefix = key.split("_")[0]
    assert len(prefix) == 16, f"Expected 16-char prefix, got {len(prefix)}: {prefix!r}"


# ── 3. Manifest creation and structure ───────────────────────────────────── #

def test_manifest_created_on_pack(tmp_path):
    store = _make_store(tmp_path)
    key, val = _fp16_kv()
    store.pack_document("doc_abc", layer=0, key=key, value=val, scheme="fp16", overwrite=True)

    manifest_path = tmp_path / "manifest.json"
    assert manifest_path.exists(), "manifest.json not created"
    data = json.loads(manifest_path.read_text())
    assert "doc_abc" in data["entries"]
    entry = data["entries"]["doc_abc"]
    assert len(entry["sha256"]) == 64, "manifest must store full 64-char SHA-256"
    assert "0::fp16" in entry["layers"]
    layer_entry = entry["layers"]["0::fp16"]
    assert layer_entry["num_tokens"] == 4
    assert "stored_at" in layer_entry
    assert "key" in layer_entry


def test_manifest_updated_on_second_pack(tmp_path):
    store = _make_store(tmp_path)
    key, val = _fp16_kv()
    store.pack_document("doc_x", layer=0, key=key, value=val, scheme="fp16", overwrite=True)
    store.pack_document("doc_x", layer=1, key=key, value=val, scheme="fp16", overwrite=True)

    data = json.loads((tmp_path / "manifest.json").read_text())
    layers = data["entries"]["doc_x"]["layers"]
    assert "0::fp16" in layers
    assert "1::fp16" in layers


# ── 4. Manifest integrity verification ───────────────────────────────────── #

def test_verify_passes_for_correct_manifest(tmp_path):
    store = _make_store(tmp_path)
    key, val = _fp16_kv()
    store.pack_document("real_doc", layer=0, key=key, value=val, scheme="fp16", overwrite=True)
    # Should not raise
    store._verify_manifest("real_doc", layer=0, scheme="fp16")


def test_verify_raises_on_tampered_sha256(tmp_path):
    """load_document must raise RuntimeError if manifest SHA-256 has been tampered."""
    store = _make_store(tmp_path)
    key, val = _fp16_kv()
    store.pack_document("real_doc", layer=0, key=key, value=val, scheme="fp16", overwrite=True)

    # Tamper: overwrite the stored SHA-256 with garbage
    manifest_path = tmp_path / "manifest.json"
    data = json.loads(manifest_path.read_text())
    data["entries"]["real_doc"]["sha256"] = "a" * 64
    manifest_path.write_text(json.dumps(data))

    with pytest.raises(RuntimeError, match="[Mm]anifest"):
        store.load_document("real_doc", layer=0, scheme="fp16", head_shape=(2, 8))


def test_verify_skips_for_unknown_doc(tmp_path):
    """Missing manifest entry is allowed (backward compat). _verify_manifest must not raise."""
    store = _make_store(tmp_path)
    # No pack — manifest is empty; verify should silently pass
    store._verify_manifest("unknown_doc", layer=0, scheme="fp16")


# ── 5. Persistence: .bin and .meta survive store re-open ─────────────────── #

def test_bin_and_meta_files_created(tmp_path):
    store = _make_store(tmp_path)
    key, val = _fp16_kv(N=8, H=2, D=8)
    store.pack_document("persist_doc", layer=1, key=key, value=val, scheme="fp16", overwrite=True)

    bin_files = list(tmp_path.glob("*.bin"))
    meta_files = list(tmp_path.glob("*.meta"))
    assert len(bin_files) >= 1, f"No .bin file in {tmp_path}"
    assert len(meta_files) >= 1, f"No .meta file in {tmp_path}"


def test_exists_returns_true_after_pack(tmp_path):
    store = _make_store(tmp_path)
    key, val = _fp16_kv()
    store.pack_document("check_doc", layer=0, key=key, value=val, scheme="fp16", overwrite=True)
    assert store.exists("check_doc", layer=0, scheme="fp16")
    assert not store.exists("check_doc", layer=1, scheme="fp16")


# ── 6. HF DynamicCache stub ───────────────────────────────────────────────── #

def test_build_dynamic_cache_calls_update(tmp_path):
    """build_dynamic_cache must call cache.update for each packed layer."""
    store = _make_store(tmp_path)
    key, val = _fp16_kv(N=4, H=2, D=8)
    for layer in [0, 1]:
        store.pack_document("cache_doc", layer=layer, key=key, value=val,
                            scheme="fp16", overwrite=True)

    mock_cache = MagicMock()
    # Patch the DynamicCache import inside build_dynamic_cache's local scope
    with patch("transformers.DynamicCache", return_value=mock_cache):
        import importlib
        import transformers
        orig = getattr(transformers, "DynamicCache", None)
        transformers.DynamicCache = lambda: mock_cache
        try:
            cache, doc_len = store.build_dynamic_cache(
                "cache_doc", num_layers=2, scheme="fp16", head_shape=(2, 8)
            )
        finally:
            if orig is not None:
                transformers.DynamicCache = orig

    assert mock_cache.update.call_count == 2, (
        f"Expected update() called twice, got {mock_cache.update.call_count}"
    )
    assert doc_len == 4


# ── 7. GPU pack/dequant roundtrip ─────────────────────────────────────────── #

_ROUNDTRIP_MSE_TOL = {
    "turbo_prod": 0.35,
    "turbo_mse":  0.25,
}


@_needs_lib
@_needs_cuda
@pytest.mark.parametrize("scheme", ["turbo_prod", "turbo_mse"])
def test_roundtrip_smoke(tmp_path, scheme):
    """Pack→dequant roundtrip must preserve shape/dtype; relative MSE within tolerance."""
    from tq_backend.cag_store import CAGStore

    store = CAGStore(tmp_path, lib_path=_LIB)
    N, H, D = 16, 2, 128  # all kernels require head_dim == 128
    key = torch.randn(N, H, D, device="cuda", dtype=torch.float16)
    val = torch.randn(N, H, D, device="cuda", dtype=torch.float16)

    store.pack_document("rt_doc", layer=0, key=key, value=val, scheme=scheme, overwrite=True)
    assert store.exists("rt_doc", layer=0, scheme=scheme)

    k_out, v_out = store.load_as_kv_fp16("rt_doc", layer=0, scheme=scheme,
                                          head_shape=(H, D))
    assert k_out.shape == (N, H, D), f"K shape mismatch: {k_out.shape}"
    assert v_out.shape == (N, H, D), f"V shape mismatch: {v_out.shape}"
    assert k_out.dtype == torch.float16
    assert v_out.dtype == torch.float16

    tol = _ROUNDTRIP_MSE_TOL[scheme]
    k_mse = torch.mean((key.float() - k_out.float()) ** 2).item()
    v_mse = torch.mean((val.float() - v_out.float()) ** 2).item()
    k_var = torch.var(key.float()).item() + 1e-8
    v_var = torch.var(val.float()).item() + 1e-8
    assert k_mse / k_var < tol, f"[{scheme}] K MSE {k_mse:.4f}/var {k_var:.4f} > tol {tol}"
    assert v_mse / v_var < tol, f"[{scheme}] V MSE {v_mse:.4f}/var {v_var:.4f} > tol {tol}"

    # Manifest must record the roundtrip layer.
    import json
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert "rt_doc" in manifest["entries"]
    assert f"0::{scheme}" in manifest["entries"]["rt_doc"]["layers"]


# ── 8. Migration script smoke test ────────────────────────────────────────── #

def test_migrate_store_dry_run(tmp_path):
    """migrate_store dry-run must detect old-format files without modifying anything."""
    import hashlib, json
    from scripts.migrate_store import migrate_store  # noqa: E402

    doc_id = "legacy_doc"
    old_pfx = hashlib.md5(doc_id.encode()).hexdigest()[:12]
    new_pfx = hashlib.sha256(doc_id.encode()).hexdigest()[:16]

    # Create fake old-format files
    (tmp_path / f"{old_pfx}_L00_turbo_mse.bin").write_bytes(b"\x00" * 16)
    (tmp_path / f"{old_pfx}_L00_turbo_mse.meta").write_bytes(b"")

    migrate_store(tmp_path, [doc_id], dry_run=True)

    # Dry run: old files must still exist, new files must not
    assert (tmp_path / f"{old_pfx}_L00_turbo_mse.bin").exists(), "old .bin removed in dry run"
    assert not (tmp_path / f"{new_pfx}_L00_turbo_mse.bin").exists(), "new .bin created in dry run"
    assert not (tmp_path / "manifest.json").exists(), "manifest created in dry run"


def test_migrate_store_live(tmp_path):
    """migrate_store live run renames files and creates a valid manifest."""
    import hashlib, json
    from scripts.migrate_store import migrate_store

    doc_id = "legacy_doc"
    old_pfx = hashlib.md5(doc_id.encode()).hexdigest()[:12]
    new_pfx = hashlib.sha256(doc_id.encode()).hexdigest()[:16]
    full_hash = hashlib.sha256(doc_id.encode()).hexdigest()

    (tmp_path / f"{old_pfx}_L00_turbo_mse.bin").write_bytes(b"\x00" * 16)
    (tmp_path / f"{old_pfx}_L00_turbo_mse.meta").write_bytes(b"")

    migrate_store(tmp_path, [doc_id], dry_run=False)

    assert not (tmp_path / f"{old_pfx}_L00_turbo_mse.bin").exists(), "old .bin still exists"
    assert (tmp_path / f"{new_pfx}_L00_turbo_mse.bin").exists(), "new .bin not created"

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert doc_id in manifest["entries"]
    entry = manifest["entries"][doc_id]
    assert entry["sha256"] == full_hash
    assert "0::turbo_mse" in entry["layers"]
