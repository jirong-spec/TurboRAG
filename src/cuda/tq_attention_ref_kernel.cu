#include "tq_attention_ref.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdexcept>

// ---------------------------------------------------------------------------
// GPU scaled dot-product attention reference kernel.
//
// One block per (query token, KV head).
// One thread per head-dim element (tid ∈ [0, head_dim)).
//
// Shared memory layout: q[MAX_D] | vaccum[MAX_D] | red[MAX_D]
//
// Algorithm (two passes over KV tokens):
//   Pass 1 – compute scaled dot-products → logit_scratch, track running max
//   Softmax – numerically stable exp/sum using the tracked max
//   Pass 2 – accumulate weighted V into vaccum, write output
// ---------------------------------------------------------------------------

namespace {

template<int MAX_D>
__global__ void attention_ref_kernel(
    const half* __restrict__ query,
    const half* __restrict__ key,
    const half* __restrict__ value,
    half* __restrict__ output,
    float* __restrict__ logit_scratch,
    TQConfig cfg,
    int num_queries,
    int num_kv_tokens)
{
    int q_idx    = blockIdx.x;
    int head_idx = blockIdx.y;
    int tid      = threadIdx.x;

    if (q_idx >= num_queries || head_idx >= cfg.num_kv_heads) return;
    if (cfg.head_dim > MAX_D) return;

    int D = cfg.head_dim;
    if (tid >= D) return;

    const float inv_sqrt_d = rsqrtf((float)D);

    extern __shared__ float smem[];
    float* q_smem  = smem;           // [MAX_D]  – query elements
    float* vaccum  = smem + MAX_D;   // [MAX_D]  – weighted-V accumulator
    float* red     = smem + 2*MAX_D; // [MAX_D]  – reduction scratch

    // Load query into shared memory
    const int qbase = (q_idx * cfg.num_kv_heads + head_idx) * D;
    q_smem[tid] = __half2float(query[qbase + tid]);
    __syncthreads();

    float* my_logits = logit_scratch +
        (size_t)(q_idx * cfg.num_kv_heads + head_idx) * num_kv_tokens;

    // -------------------------------------------------------------------
    // Pass 1: compute scaled dot-products and track the running maximum
    // -------------------------------------------------------------------
    float max_logit = -1e30f;

    for (int t = 0; t < num_kv_tokens; ++t) {
        const int kbase = (t * cfg.num_kv_heads + head_idx) * D;
        red[tid] = q_smem[tid] * __half2float(key[kbase + tid]);
        __syncthreads();

        // Parallel reduce → dot product in red[0]
        for (int stride = D >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) red[tid] += red[tid + stride];
            __syncthreads();
        }

        if (tid == 0) {
            float logit  = red[0] * inv_sqrt_d;
            my_logits[t] = logit;
            if (logit > max_logit) max_logit = logit;
        }
        __syncthreads();
    }

    // Broadcast max_logit to all threads via shared memory
    if (tid == 0) red[0] = max_logit;
    __syncthreads();
    max_logit = red[0];

    // -------------------------------------------------------------------
    // Compute softmax denominator (done by thread 0, result broadcast)
    // -------------------------------------------------------------------
    if (tid == 0) {
        float s = 0.0f;
        for (int t = 0; t < num_kv_tokens; ++t)
            s += expf(my_logits[t] - max_logit);
        red[0] = (s > 0.0f) ? (1.0f / s) : 1.0f;
    }
    __syncthreads();
    const float inv_sum_exp = red[0];

    // -------------------------------------------------------------------
    // Pass 2: softmax-weighted V accumulation
    // -------------------------------------------------------------------
    vaccum[tid] = 0.0f;

    for (int t = 0; t < num_kv_tokens; ++t) {
        const int vbase = (t * cfg.num_kv_heads + head_idx) * D;
        float weight = expf(my_logits[t] - max_logit) * inv_sum_exp;
        vaccum[tid] += weight * __half2float(value[vbase + tid]);
    }

    // Write output (no __syncthreads needed: vaccum[tid] is private to each thread here)
    const int obase = (q_idx * cfg.num_kv_heads + head_idx) * D;
    output[obase + tid] = __float2half(vaccum[tid]);
}

} // namespace

void launch_attention_ref_gpu(
    const half*     query,
    const half*     key,
    const half*     value,
    half*           output,
    float*          logit_scratch,
    const TQConfig& cfg,
    int             num_queries,
    int             num_kv_tokens,
    cudaStream_t    stream)
{
    if (cfg.head_dim > 128)
        throw std::runtime_error("attention_ref_gpu: head_dim > 128 is not supported");

    dim3 grid(num_queries, cfg.num_kv_heads);
    int  threads = cfg.head_dim;
    // smem: q[128] + vaccum[128] + red[128]
    size_t shmem = sizeof(float) * (3 * 128);

    attention_ref_kernel<128><<<grid, threads, shmem, stream>>>(
        query, key, value, output, logit_scratch,
        cfg, num_queries, num_kv_tokens);
}
