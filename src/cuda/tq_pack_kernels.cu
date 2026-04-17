#include "tq_pack_kernels.cuh"
#include "tq_quant.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>

__inline__ __device__ float warp_max(float x) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        x = fmaxf(x, __shfl_down_sync(0xffffffff, x, offset));
    }
    return x;
}

__inline__ __device__ float block_max(float x) {
    __shared__ float shared[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;

    x = warp_max(x);
    if (lane == 0) shared[wid] = x;
    __syncthreads();

    x = (threadIdx.x < (blockDim.x + 31) / 32) ? shared[lane] : 0.0f;
    if (wid == 0) x = warp_max(x);
    return x;
}

__device__ inline int quantize_to_int4(float x, float scale) {
    if (scale <= 1e-12f) return 0;
    int q = __float2int_rn(x / scale);
    return clamp_int(q, -8, 7);
}

__global__ void tq_pack_kv_kernel(
    const half* __restrict__ key,
    const half* __restrict__ value,
    const int32_t* __restrict__ slot_mapping,
    uint8_t* __restrict__ page_pool,
    TQPageLayout layout,
    TQConfig cfg,
    int num_tokens) {

    int token_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int lane = threadIdx.x;
    int D = cfg.head_dim;

    if (token_idx >= num_tokens || head_idx >= cfg.num_kv_heads) return;

    int slot = slot_mapping[token_idx];
    int physical_block = slot / cfg.block_size;
    int token_in_block = slot % cfg.block_size;
    uint8_t* page_base = page_pool + static_cast<size_t>(physical_block) * layout.page_size_bytes;

    uint8_t* k_codes_ptr = page_base + layout.k_codes_offset +
        token_head_code_offset(layout, token_in_block, head_idx, cfg.num_kv_heads);
    uint8_t* v_codes_ptr = page_base + layout.v_codes_offset +
        token_head_code_offset(layout, token_in_block, head_idx, cfg.num_kv_heads);

    half* k_scale_ptr = reinterpret_cast<half*>(
        page_base + layout.k_scales_offset +
        token_head_scale_offset(layout, token_in_block, head_idx, cfg.num_kv_heads));
    half* v_scale_ptr = reinterpret_cast<half*>(
        page_base + layout.v_scales_offset +
        token_head_scale_offset(layout, token_in_block, head_idx, cfg.num_kv_heads));

    extern __shared__ float smem[];
    float* k_tmp = smem;
    float* v_tmp = smem + D;

    int base = (token_idx * cfg.num_kv_heads + head_idx) * D;
    float local_k_absmax = 0.f;
    float local_v_absmax = 0.f;

    for (int i = lane; i < D; i += blockDim.x) {
        float k = __half2float(key[base + i]);
        float v = __half2float(value[base + i]);
        k_tmp[i] = k;
        v_tmp[i] = v;
        local_k_absmax = fmaxf(local_k_absmax, fabsf(k));
        local_v_absmax = fmaxf(local_v_absmax, fabsf(v));
    }
    __syncthreads();

    float k_absmax = block_max(local_k_absmax);
    float v_absmax = block_max(local_v_absmax);

    __shared__ float k_scale;
    __shared__ float v_scale;
    if (threadIdx.x == 0) {
        k_scale = fmaxf(k_absmax / 7.0f, 1e-8f);
        v_scale = fmaxf(v_absmax / 7.0f, 1e-8f);
        *k_scale_ptr = __float2half(k_scale);
        *v_scale_ptr = __float2half(v_scale);
    }
    __syncthreads();

    for (int i = lane * 2; i < D; i += blockDim.x * 2) {
        int qk0 = quantize_to_int4(k_tmp[i], k_scale);
        int qv0 = quantize_to_int4(v_tmp[i], v_scale);
        int qk1 = 0;
        int qv1 = 0;

        if (i + 1 < D) {
            qk1 = quantize_to_int4(k_tmp[i + 1], k_scale);
            qv1 = quantize_to_int4(v_tmp[i + 1], v_scale);
        }

        k_codes_ptr[i / 2] = pack_int4(qk0, qk1);
        v_codes_ptr[i / 2] = pack_int4(qv0, qv1);
    }
}

void launch_tq_pack_kv(
    const half* key,
    const half* value,
    const int32_t* slot_mapping,
    uint8_t* page_pool,
    const TQPageLayout& layout,
    const TQConfig& cfg,
    int num_tokens,
    cudaStream_t stream) {
    dim3 grid(num_tokens, cfg.num_kv_heads);
    dim3 block(128);
    size_t smem = sizeof(float) * cfg.head_dim * 2;
    tq_pack_kv_kernel<<<grid, block, smem, stream>>>(
        key, value, slot_mapping, page_pool, layout, cfg, num_tokens);
}
