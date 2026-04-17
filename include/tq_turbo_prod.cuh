#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "tq_config.h"

struct TQTurboProdPageLayout {
    size_t page_size_bytes;

    size_t k3_codes_offset;
    size_t k_residual_offset;
    size_t k_residual_scales_offset;
    size_t k_scales_offset;

    size_t v4_codes_offset;
    size_t v_scales_offset;

    int k3_bytes_per_token_head;
    int kres_bytes_per_token_head;
    int v4_bytes_per_token_head;
    int scale_bytes_per_token_head;
};

__host__ __device__ inline int ceil_div_int(int a, int b) {
    return (a + b - 1) / b;
}

__host__ __device__ inline int packed_3bit_bytes(int n) {
    return ceil_div_int(n * 3, 8);
}

__host__ __device__ inline int packed_1bit_bytes(int n) {
    return ceil_div_int(n, 8);
}

__host__ __device__ inline int packed_4bit_bytes(int n) {
    return ceil_div_int(n, 2);
}

__host__ __device__ inline size_t turbo_prod_token_head_offset(
    int token_in_block,
    int head_idx,
    int num_kv_heads,
    int bytes_per_token_head) {
    return ((size_t)token_in_block * num_kv_heads + head_idx) * bytes_per_token_head;
}

#ifdef __cplusplus
extern "C" {
#endif

TQTurboProdPageLayout make_tq_turbo_prod_page_layout(const TQConfig& cfg);

void launch_tq_turbo_prod_pack_kv(
    const half* key,
    const half* value,
    const int32_t* slot_mapping,
    uint8_t* page_pool,
    const TQTurboProdPageLayout& layout,
    const TQConfig& cfg,
    int num_tokens,
    cudaStream_t stream,
    float* debug_k_rot = nullptr,
    float* debug_v_rot = nullptr,
    float* debug_kn = nullptr,
    float* debug_vn = nullptr,
    int* debug_kidx = nullptr,
    int* debug_vidx = nullptr);

void launch_tq_turbo_prod_dequant_kv(
    const uint8_t* page_pool,
    const int32_t* slot_mapping,
    half* out_key,
    half* out_value,
    const TQTurboProdPageLayout& layout,
    const TQConfig& cfg,
    int num_tokens,
    cudaStream_t stream);

void launch_tq_turbo_prod_fused_attention_logits(
    const half* query,
    const uint8_t* page_pool,
    const int32_t* slot_mapping,
    float* logits,
    const TQTurboProdPageLayout& layout,
    const TQConfig& cfg,
    int num_queries,
    int num_kv_tokens,
    cudaStream_t stream);

void launch_tq_turbo_prod_fused_attention_output(
    const half* query,
    const uint8_t* page_pool,
    const int32_t* slot_mapping,
    half* output,
    const TQTurboProdPageLayout& layout,
    const TQConfig& cfg,
    int num_queries,
    int num_kv_tokens,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif