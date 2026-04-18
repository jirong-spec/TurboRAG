# TurboQuant — Fused CUDA Kernels for Paged KV-Cache Quantization

TurboQuant is a research library for **RAG and long-context LLM inference** on NVIDIA GPUs.  
It fuses sub-4-bit quantization, paged KV-cache management, and FlashAttention-style online softmax into a single CUDA kernel pipeline — eliminating intermediate FP16 materialisation of the KV cache during decode.

---

## Quantization Schemes

| Scheme | K precision | V precision | Effective bits | Compression vs FP16 |
|---|---|---|---|---|
| **turbo_prod** | INT3 (Lloyd-Max) + INT1 QJL residual | INT4 (Lloyd-Max) | ~3.5 bits | **3.82×** |
| **turbo_mse** | INT4 MSE-optimal | INT4 MSE-optimal | 4 bits | **3.88×** |
| FP16 baseline | — | — | 16 bits | 1× |

### turbo_prod
Production-grade scheme optimised for throughput.  
K is compressed with a 3-bit Lloyd-Max codebook and a 1-bit QJL signed-residual correction.  
The fused attention kernel decodes K and V **on-the-fly inside shared memory** (FlashAttention algorithm), computing the full softmax-weighted output without writing any FP16 KV tensor to global memory.

### turbo_mse
Validation scheme optimised for reconstruction fidelity.  
Both K and V use INT4 quantization with a loss function that minimises mean-squared error.  
Pack latency is **~40% lower** than turbo_prod, making it preferable when throughput matters more than compression depth.

---

## Project Layout

```
tuboRAG/
├── CMakeLists.txt
├── include/                      # C++ / CUDA headers
│   ├── tq_config.h               #   TQConfig (block_size=16, heads=8, dim=128)
│   ├── tq_types.h                #   block_id_t, TQScaleType, TQQuantMode
│   ├── tq_turbo_prod.cuh         #   turbo_prod layout + kernel declarations
│   ├── tq_turbo_mse_layout.h     #   turbo_mse layout struct
│   ├── tq_turbo_mse_kernels.cuh  #   turbo_mse kernel declarations
│   ├── tq_allocator.h            #   Paged block allocator
│   ├── tq_block_table.h          #   Sequence-state + slot-map builder
│   └── tq_attention_ref.h        #   CPU/GPU reference SDPA
├── src/
│   ├── tq_capi.cpp               #   extern "C" API surface (turbo_prod + turbo_mse)
│   ├── tq_turbo_prod.cpp         #   Page-layout arithmetic
│   ├── tq_turbo_mse_layout.cpp   #   MSE layout arithmetic
│   └── cuda/
│       ├── tq_turbo_prod_kernels.cu      # Pack / dequant / fused-attention
│       ├── tq_turbo_mse_kernels.cu       # MSE pack / dequant / fused-attention
│       └── tq_attention_ref_kernel.cu    # Reference SDPA kernel
├── build/
│   └── libturboquant.so          # Compiled shared library
├── config.yaml                   # ← Edit this to use your own dataset
├── python/
│   ├── turboquant_wrapper.py     # ctypes bindings — turbo_prod + turbo_mse
│   ├── rag_turbo_demo.py         # Config-driven RAG benchmark (bring your own data)
│   ├── rag_demo.py               # GYG demo (KaggleHub, fixed dataset)
│   ├── benchmark.py              # Latency / fidelity / memory sweep
│   └── rag_turbo_comparison.py   # Synthetic RAG prefill + decode comparison
├── data/
│   └── qa_pairs.jsonl            # Auto-generated Q&A pairs (cached)
├── output/
│   ├── rag_results.md            # Benchmark report (Markdown)
│   └── rag_results.json          # Benchmark report (JSON)
├── start_ollama_gpu.sh           # Ollama GPU startup helper
└── README.md
```

---

## Requirements

| Dependency | Minimum version |
|---|---|
| CUDA Toolkit | 11.7 |
| CMake | 3.20 |
| C++ | 17 |
| Python | 3.10 |
| PyTorch | 2.0 (CUDA build) |
| requests | any |
| kagglehub | 1.0+ |
| pandas | 2.0+ |
| Ollama | 0.21+ (for RAG demo) |

Tested on **NVIDIA GeForce RTX 3060 (12 GB)**, CUDA 12.4, driver 550.163.

---

## Build

```bash
# configure (Ampere sm_80; edit CMakeLists.txt for other arches)
cmake -B build -S .

# compile — produces build/libturboquant.so
cmake --build build --parallel $(nproc)
```

---

## Python API

```python
from turboquant_wrapper import TurboQuantWrapper
import torch

tq     = TurboQuantWrapper()          # loads build/libturboquant.so
cfg    = tq.default_config()
layout = tq.make_layout_for(cfg)      # turbo_prod layout
T, H, D = 2048, cfg.num_kv_heads, cfg.head_dim

key   = torch.randn(T, H, D, device="cuda", dtype=torch.float16).contiguous()
value = torch.randn(T, H, D, device="cuda", dtype=torch.float16).contiguous()
slots = torch.arange(T, dtype=torch.int32, device="cuda")
pool  = tq.alloc_page_pool(T, layout, cfg)

# ── turbo_prod ──────────────────────────────────────────────────────
tq.pack(key, value, slots, pool, layout, cfg)          # compress KV

out_k = torch.empty_like(key)
out_v = torch.empty_like(value)
tq.dequant(pool, slots, out_k, out_v, layout, cfg)     # decompress
torch.cuda.synchronize()

kv_mse = TurboQuantWrapper.compute_mse(
    torch.cat([out_k, out_v]), torch.cat([key, value]))

query  = torch.randn(1, H, D, device="cuda", dtype=torch.float16).contiguous()
output = torch.empty(1, H, D, device="cuda", dtype=torch.float16).contiguous()
tq.fused_attn_output(                                  # fused decode, no KV spill
    query, pool, slots, output, layout, cfg,
    num_queries=1, num_kv_tokens=T)

# ── turbo_mse ───────────────────────────────────────────────────────
mse_layout = tq.make_mse_layout_for(cfg)
mse_pool   = tq.alloc_mse_pool(T, mse_layout, cfg)

tq.mse_pack(key, value, slots, mse_pool, mse_layout, cfg)

out_mk = torch.empty_like(key)
out_mv = torch.empty_like(value)
tq.mse_dequant(mse_pool, slots, out_mk, out_mv, mse_layout, cfg)

mse_out = torch.empty(1, H, D, device="cuda", dtype=torch.float16).contiguous()
tq.mse_fused_attn_output(
    query, mse_pool, slots, mse_out, mse_layout, cfg,
    num_queries=1, num_kv_tokens=T)
```

### Memory helpers

```python
fp16_mb  = tq.fp16_bytes(T, cfg)          / 1024**2   # FP16 KV bytes
prod_mb  = tq.quant_bytes(T, layout, cfg) / 1024**2   # turbo_prod bytes
mse_mb   = tq.mse_bytes(T, mse_layout, cfg) / 1024**2 # turbo_mse bytes
print(tq.summary())                                    # config + compression ratios
```

---

## Bring Your Own Data — `rag_turbo_demo.py`

`rag_turbo_demo.py` is a **config-driven** RAG benchmark. Swap the dataset by editing `config.yaml` — no code changes needed.

### Quick start

```bash
# 1. Edit config.yaml (see options below)
# 2. Start Ollama on GPU
bash start_ollama_gpu.sh &
ollama pull qwen2.5:7b

# 3. Set Kaggle credentials (if using a Kaggle dataset)
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key

# 4. Run
cd python
python3 rag_turbo_demo.py
```

### config.yaml — key options

```yaml
dataset:
  source: kaggle          # "kaggle" | "local"
  kaggle_handle: "owner/dataset-name"
  kaggle_file:   "file.csv"
  # local_path: "data/my_corpus.csv"   # alternative to Kaggle
  nrows: 5000             # rows to load (keeps memory manageable)
  text_columns:           # columns joined to build the BM25 corpus
    - name
    - description

qa:
  path: "data/qa_pairs.jsonl"   # cached Q&A file
  auto_generate: true           # generate from structured columns
  name_column: "name"           # subject of generated questions
  templates:                    # one entry per answer column
    city:     "In which city can you find '{name}'?"
    country:  "What country is '{name}' located in?"

retrieval:
  top_k: 5

llm:
  model: "qwen2.5:7b"
  sample_size: 50         # questions sent to the LLM (rest use BM25 only)
  system_prompt: "You are a helpful assistant. Answer using ONLY the context."

output:
  dir: "output"
  report_stem: "rag_results"   # → output/rag_results.md + .json
```

### Using a local CSV

```yaml
dataset:
  source: local
  local_path: "data/my_corpus.csv"   # relative to repo root, or absolute
  nrows: 10000
  text_columns:
    - title
    - body
```

### Using your own Q&A file

Set `qa.auto_generate: false` and point `qa.path` at a JSONL file where each line is:

```json
{"question": "What is X?", "answer": "Y"}
```

### Output files

| File | Contents |
|---|---|
| `output/rag_results.md` | Retrieval recall, LLM accuracy, KV efficiency table |
| `output/rag_results.json` | Same data in JSON for downstream processing |
| `data/qa_pairs.jsonl` | Auto-generated Q&A pairs (cached on first run) |

### Results (RTX 3060, GYG dataset, qwen2.5:7b)

| Metric | Value |
|---|---|
| BM25 retrieval recall (5 000 Q) | **48.1%** |
| LLM answer accuracy (50-Q sample) | **22–26%** |
| turbo_prod VRAM compression | **3.81×** |
| turbo_mse VRAM compression | **3.86×** |
| turbo_mse pack latency vs turbo_prod | **~40% faster** |

---

## RAG Demo

`rag_demo.py` runs an end-to-end comparison of **TurboQuant + Ollama** vs **standard FP16 Ollama** on the [GYG travel activities dataset](https://www.kaggle.com/datasets/sanjarbek1/rag-dataset-with-gyg) fetched live from KaggleHub.

### Pipeline

```
KaggleHub: sanjarbek1/rag-dataset-with-gyg (gyg_data_full.csv, 5 000 rows)
      │
      ▼ table-aware chunking (~15 000 chunks)
      │
      ▼ BM25 retrieval (top-5 chunks per question)
      │
      ├─► Standard Ollama ──► answer + latency       (Metric A)
      │
      └─► TurboQuant KV simulation
              ├─ turbo_prod: pack → dequant → fused-attn
              └─ turbo_mse:  pack → dequant → fused-attn
                                              ▼
                               VRAM, Pack µs, KV MSE, Attn MSE  (Metric B)
```

### Start Ollama on GPU

```bash
# First-time setup: ensures libggml-base.so.0 and libnvidia-ml are found
bash start_ollama_gpu.sh &
ollama pull qwen2.5:7b
```

### Kaggle credentials

```bash
export KAGGLE_USERNAME=<your_username>
export KAGGLE_KEY=<your_api_key>
```

Get a key at **kaggle.com → Settings → API → Create New Token**.

### Run

```bash
cd python
python3 rag_demo.py
python3 rag_demo.py --model qwen2.5:7b --top-k 5 --iters 30
```

### Results (RTX 3060, qwen2.5:7b)

#### Metric A — Answer Accuracy

| Q | Question | Score | Latency |
|---|---|---|---|
| Q1 | What thermal spa or hot spring experiences are available? | **60%** | 9.1 s |
| Q2 | What guided city tours or walking tours are available? | **100%** | 3.7 s |
| Q3 | What outdoor adventure activities are offered? | **60%** | 3.4 s |
| Q4 | What food or culinary experiences are available? | **60%** | 3.6 s |
| Q5 | What museum or cultural attraction entrance tickets are available? | **40%** | 4.5 s |
| **avg** | | **64%** | **4.9 s** |

#### Metric B — KV-Cache Efficiency

| Scheme | Tokens | FP16 MB | Quant MB | Compression | Pack µs | KV MSE | Attn MSE |
|---|---|---|---|---|---|---|---|
| turbo_prod | ~689 | 2.69 | 0.70 | **3.8×** | 150 | 1.07e-02 | 1.54e-01 |
| turbo_mse | ~689 | 2.69 | 0.69 | **3.8×** | **91** | **9.3e-03** | 8.3e-02 |
| FP16 | ~689 | 2.69 | 2.69 | 1× | — | 0 | 0 |

> turbo_mse packs **40% faster** than turbo_prod and achieves lower KV reconstruction error.  
> turbo_prod reaches **3.8× VRAM reduction** with fused no-spill attention.

---

## Synthetic RAG Benchmark

```bash
cd python
python rag_turbo_comparison.py
python rag_turbo_comparison.py --num-docs 8 --doc-tokens 1024 --iters 100
```

Simulates prefill (pack) + single-token decode (fused attention) for a configurable number of retrieved documents. Prints a three-section Markdown report: Phase 1 prefill, Phase 2 decode, end-to-end totals.

---

## Kernel Benchmarks

```bash
cd python
python benchmark.py
```

Sweeps token counts `[128, 512, 1024, 2048, 4096, 8192]` with 20 warmup + 100 timed iterations each. Outputs latency, throughput (tokens/µs), KV MSE, and compression to stdout and saves `turboquant_benchmark_report.png`.

**Sample output (RTX 3060)**

| Tokens | Pack µs | Dequant µs | KV MSE |
|---|---|---|---|
| 512 | ~45 | ~23 | 0.0122 |
| 2048 | ~108 | ~55 | 0.0122 |
| 8192 | ~128 | ~69 | 0.0122 |

MSE is constant across all sequence lengths — quantization error does not accumulate with context depth.

---

## Design Notes

**Logit convention.** All kernels use `logit = ⟨q, k⟩ / √D` (standard scaled dot-product attention). The reference FP16 attention in `rag_turbo_comparison.py` and `rag_demo.py` matches this convention so MSE comparisons are apples-to-apples.

**Paged allocation.** `TQAllocator` manages a GPU page pool of `block_size=16` token slots. `TQBlockTable` maps sequence IDs to slot lists, enabling dynamic KV eviction and multi-sequence batching.

**No CUDA graphs needed.** The pack and fused-attention kernels are launch-overhead-minimal (single dispatch per sequence) and work correctly with arbitrary slot permutations.

**GPU startup.** On systems where Ollama's GPU runner cannot find `libnvidia-ml.so.1`, use `start_ollama_gpu.sh` which sets the required `LD_LIBRARY_PATH` before launching the server.
