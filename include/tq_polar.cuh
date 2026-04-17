#pragma once
// ---------------------------------------------------------------------------
// PolarQuant KV cache kernels (K=2-bit + V=3-bit, Hadamard-rotated domain)
//
// Same WHT preprocessing as turbo_prod and QJL; inner-product identity
// <qrot, krot> = <q, k> means attention logits computed in rotated domain.
// ---------------------------------------------------------------------------

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "tq_config.h"
#include "tq_polar_layout.h"

// Compress KV tokens into the page pool.
void launch_tq_polar_pack_kv(
    const half*              key,
    const half*              value,
    const int32_t*           slot_mapping,
    uint8_t*                 page_pool,
    const TQPolarPageLayout& layout,
    const TQConfig&          cfg,
    int                      num_tokens,
    cudaStream_t             stream);

// Decompress KV tokens back to FP16.
void launch_tq_polar_dequant_kv(
    const uint8_t*           page_pool,
    const int32_t*           slot_mapping,
    half*                    out_key,
    half*                    out_value,
    const TQPolarPageLayout& layout,
    const TQConfig&          cfg,
    int                      num_tokens,
    cudaStream_t             stream);

// Fused attention: online softmax → weighted-V sum → inverse rotation.
// Single pass over KV tokens; no logit_scratch global memory required.
// Logit convention: <q, k>  (no 1/sqrt(D)), matching V6 / QJL convention.
void launch_tq_polar_fused_attention_output(
    const half*              query,
    const uint8_t*           page_pool,
    const int32_t*           slot_mapping,
    half*                    output,
    const TQPolarPageLayout& layout,
    const TQConfig&          cfg,
    int                      num_queries,
    int                      num_kv_tokens,
    cudaStream_t             stream);
