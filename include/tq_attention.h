#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "tq_turbo_prod.cuh"

// Full fused attention output for turbo_prod quantization scheme.
//
// Computes attention in the Hadamard-rotated domain:
//   logit[q, h, t] = <Rotate(q[q,h,:]), k_rotated[t,h,:]>
//   weight[q, h, t] = softmax_t(logit[q, h, :])
//   output[q, h, :] = InverseRotate( sum_t weight[q,h,t] * v_rotated[t,h,:] )
//
// This avoids materialising dequantised K/V in global memory.
//
// Parameters
// ----------
// query        : [num_queries, num_kv_heads, head_dim] in FP16
// page_pool    : compressed KV cache (turbo_prod paged layout)
// slot_mapping : [num_kv_tokens] physical slot index for each KV token
// output       : [num_queries, num_kv_heads, head_dim] in FP16, written by this call
// logit_scratch: caller-allocated FP32 buffer of size
//                num_queries * num_kv_heads * num_kv_tokens (floats)
// layout, cfg  : turbo_prod page layout and model configuration
// num_queries  : number of query tokens (Q length)
// num_kv_tokens: number of KV tokens already packed in the cache (K/V length)
// stream       : CUDA stream (0 = default)
void launch_tq_turbo_prod_fused_attention_output(
    const half*                   query,
    const uint8_t*                page_pool,
    const int32_t*                slot_mapping,
    half*                         output,
    float*                        logit_scratch,
    const TQTurboProdPageLayout&    layout,
    const TQConfig&               cfg,
    int                           num_queries,
    int                           num_kv_tokens,
    cudaStream_t                  stream);
