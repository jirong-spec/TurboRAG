#include "tq_dense_kernels.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

__global__ void dense_store_kv_kernel(
    const half* __restrict__ key,
    const half* __restrict__ value,
    const int32_t* __restrict__ slot_mapping,
    uint8_t* __restrict__ page_pool,
    TQDensePageLayout layout,
    TQConfig cfg,
    int num_tokens) {

    int token_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int lane = threadIdx.x;

    if (token_idx >= num_tokens || head_idx >= cfg.num_kv_heads) return;

    int slot = slot_mapping[token_idx];
    int physical_block = slot / cfg.block_size;
    int token_in_block = slot % cfg.block_size;
    uint8_t* page_base = page_pool + static_cast<size_t>(physical_block) * layout.page_size_bytes;

    half* k_ptr = reinterpret_cast<half*>(
        page_base + layout.k_offset + dense_token_head_offset(layout, cfg, token_in_block, head_idx));
    half* v_ptr = reinterpret_cast<half*>(
        page_base + layout.v_offset + dense_token_head_offset(layout, cfg, token_in_block, head_idx));

    int D = cfg.head_dim;
    int base = (token_idx * cfg.num_kv_heads + head_idx) * D;
    for (int i = lane; i < D; i += blockDim.x) {
        k_ptr[i] = key[base + i];
        v_ptr[i] = value[base + i];
    }
}

__global__ void dense_load_kv_kernel(
    const uint8_t* __restrict__ page_pool,
    const int32_t* __restrict__ slot_mapping,
    half* __restrict__ out_key,
    half* __restrict__ out_value,
    TQDensePageLayout layout,
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

    const half* k_ptr = reinterpret_cast<const half*>(
        page_base + layout.k_offset + dense_token_head_offset(layout, cfg, token_in_block, head_idx));
    const half* v_ptr = reinterpret_cast<const half*>(
        page_base + layout.v_offset + dense_token_head_offset(layout, cfg, token_in_block, head_idx));

    int D = cfg.head_dim;
    int base = (token_idx * cfg.num_kv_heads + head_idx) * D;
    for (int i = lane; i < D; i += blockDim.x) {
        out_key[base + i] = k_ptr[i];
        out_value[base + i] = v_ptr[i];
    }
}

void launch_dense_store_kv(
    const half* key,
    const half* value,
    const int32_t* slot_mapping,
    uint8_t* page_pool,
    const TQDensePageLayout& layout,
    const TQConfig& cfg,
    int num_tokens,
    cudaStream_t stream) {
    dim3 grid(num_tokens, cfg.num_kv_heads);
    dim3 block(128);
    dense_store_kv_kernel<<<grid, block, 0, stream>>>(
        key, value, slot_mapping, page_pool, layout, cfg, num_tokens);
}

void launch_dense_load_kv(
    const uint8_t* page_pool,
    const int32_t* slot_mapping,
    half* out_key,
    half* out_value,
    const TQDensePageLayout& layout,
    const TQConfig& cfg,
    int num_tokens,
    cudaStream_t stream) {
    dim3 grid(num_tokens, cfg.num_kv_heads);
    dim3 block(128);
    dense_load_kv_kernel<<<grid, block, 0, stream>>>(
        page_pool, slot_mapping, out_key, out_value, layout, cfg, num_tokens);
}
