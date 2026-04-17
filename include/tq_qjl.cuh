#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "tq_config.h"

// ---------------------------------------------------------------------------
// QJL-1b page layout
//
// Applies the same randomised Hadamard rotation as turbo_prod, then quantises:
//   K : 1-bit sign per dimension  +  FP16 per-token-head RMS scale
//   V : 4-bit Gaussian codebook   +  FP16 per-token-head RMS scale
//
// Memory per token-head at head_dim=128:
//   K: 16 B (codes) + 2 B (scale) = 18 B
//   V: 64 B (codes) + 2 B (scale) = 66 B
//   Total: 84 B  vs  512 B dense  →  ~6.1× compression
// ---------------------------------------------------------------------------
struct TQQJLPageLayout {
    size_t page_size_bytes;

    size_t k1_codes_offset;   // 1-bit K sign codes
    size_t k_scales_offset;   // FP16 K RMS scales

    size_t v4_codes_offset;   // 4-bit V codes (Gaussian Lloyd-Max codebook)
    size_t v_scales_offset;   // FP16 V RMS scales

    int k1_bytes_per_token_head;
    int v4_bytes_per_token_head;
    int scale_bytes_per_token_head;
};

TQQJLPageLayout make_tq_qjl_page_layout(const TQConfig& cfg);

// Byte offset of a (token_in_block, head_idx) cell inside one region.
__host__ __device__ inline size_t qjl_token_head_offset(
    int token_in_block,
    int head_idx,
    int num_kv_heads,
    int bytes_per_token_head)
{
    return ((size_t)token_in_block * num_kv_heads + head_idx) * bytes_per_token_head;
}

// Compress KV tokens into the page pool (1-bit K + 4-bit V in rotated domain).
void launch_tq_qjl_pack_kv(
    const half*            key,
    const half*            value,
    const int32_t*         slot_mapping,
    uint8_t*               page_pool,
    const TQQJLPageLayout& layout,
    const TQConfig&        cfg,
    int                    num_tokens,
    cudaStream_t           stream);

// Decompress KV tokens back to FP16.
void launch_tq_qjl_dequant_kv(
    const uint8_t*         page_pool,
    const int32_t*         slot_mapping,
    half*                  out_key,
    half*                  out_value,
    const TQQJLPageLayout& layout,
    const TQConfig&        cfg,
    int                    num_tokens,
    cudaStream_t           stream);

// Full fused attention: online softmax → weighted-V sum → inverse rotation.
// Single pass over KV tokens; no logit_scratch global memory required.
// Logit convention: <q, k>  (no 1/sqrt(D)), matching turbo_prod fused kernel.
void launch_tq_qjl_fused_attention_output(
    const half*            query,
    const uint8_t*         page_pool,
    const int32_t*         slot_mapping,
    half*                  output,
    const TQQJLPageLayout& layout,
    const TQConfig&        cfg,
    int                    num_queries,
    int                    num_kv_tokens,
    cudaStream_t           stream);
