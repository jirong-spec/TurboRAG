#include "tq_dequant_kernels.cuh"
#include "tq_quant.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

__global__ void tq_dequant_kv_kernel(
    const uint8_t* __restrict__ page_pool,
    const int32_t* __restrict__ slot_mapping,
    half* __restrict__ out_key,
    half* __restrict__ out_value,
    TQPageLayout layout,
    TQConfig cfg,
    int num_tokens) {

    int token_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int lane = threadIdx.x;

    if (token_idx >= num_tokens || head_idx >= cfg.num_kv_heads) return;

    int slot = slot_mapping[token_idx];
    int physical_block = slot / cfg.block_size;
    int token_in_block = slot % cfg.block_size;
    const uint8_t* page_base = page_pool + static_cast<size_t>(physical_block) * layout.page_size_bytes;

    const uint8_t* k_codes_ptr = page_base + layout.k_codes_offset +
        token_head_code_offset(layout, token_in_block, head_idx, cfg.num_kv_heads);
    const uint8_t* v_codes_ptr = page_base + layout.v_codes_offset +
        token_head_code_offset(layout, token_in_block, head_idx, cfg.num_kv_heads);

    const half* k_scale_ptr = reinterpret_cast<const half*>(
        page_base + layout.k_scales_offset +
        token_head_scale_offset(layout, token_in_block, head_idx, cfg.num_kv_heads));
    const half* v_scale_ptr = reinterpret_cast<const half*>(
        page_base + layout.v_scales_offset +
        token_head_scale_offset(layout, token_in_block, head_idx, cfg.num_kv_heads));

    float k_scale = __half2float(*k_scale_ptr);
    float v_scale = __half2float(*v_scale_ptr);
    int D = cfg.head_dim;
    int base = (token_idx * cfg.num_kv_heads + head_idx) * D;

    for (int i = lane * 2; i < D; i += blockDim.x * 2) {
        uint8_t kb = k_codes_ptr[i / 2];
        uint8_t vb = v_codes_ptr[i / 2];

        out_key[base + i] = __float2half(unpack_int4_low(kb) * k_scale);
        out_value[base + i] = __float2half(unpack_int4_low(vb) * v_scale);

        if (i + 1 < D) {
            out_key[base + i + 1] = __float2half(unpack_int4_high(kb) * k_scale);
            out_value[base + i + 1] = __float2half(unpack_int4_high(vb) * v_scale);
        }
    }
}

void launch_tq_dequant_kv(
    const uint8_t* page_pool,
    const int32_t* slot_mapping,
    half* out_key,
    half* out_value,
    const TQPageLayout& layout,
    const TQConfig& cfg,
    int num_tokens,
    cudaStream_t stream) {
    dim3 grid(num_tokens, cfg.num_kv_heads);
    dim3 block(128);
    tq_dequant_kv_kernel<<<grid, block, 0, stream>>>(
        page_pool, slot_mapping, out_key, out_value, layout, cfg, num_tokens);
}
