#pragma once
#include <vector>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "tq_config.h"

// CPU scaled dot-product attention (single head, row-major KV).
std::vector<float> attention_ref_single_head(
    const std::vector<half>& q,
    const std::vector<half>& k_all,
    const std::vector<half>& v_all,
    int num_tokens,
    int head_dim);

double mse_vec(const std::vector<float>& a, const std::vector<float>& b);
double max_abs_vec(const std::vector<float>& a, const std::vector<float>& b);

// GPU scaled dot-product attention (all heads, contiguous KV layout):
//   query  : [num_queries, num_kv_heads, head_dim] FP16
//   key    : [num_kv_tokens, num_kv_heads, head_dim] FP16
//   value  : [num_kv_tokens, num_kv_heads, head_dim] FP16
//   output : [num_queries, num_kv_heads, head_dim] FP16
//   logit_scratch : caller-allocated FP32 scratch,
//                   size = num_queries * num_kv_heads * num_kv_tokens
//
// Logit scaling: 1/sqrt(head_dim)  (standard SDPA, matches attention_ref_single_head).
// head_dim must be a power-of-two and <= 128.
void launch_attention_ref_gpu(
    const half*    query,
    const half*    key,
    const half*    value,
    half*          output,
    float*         logit_scratch,
    const TQConfig& cfg,
    int            num_queries,
    int            num_kv_tokens,
    cudaStream_t   stream = 0);
