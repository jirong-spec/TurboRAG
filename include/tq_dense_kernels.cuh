#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "tq_dense_layout.h"

void launch_dense_store_kv(
    const half* key,
    const half* value,
    const int32_t* slot_mapping,
    uint8_t* page_pool,
    const TQDensePageLayout& layout,
    const TQConfig& cfg,
    int num_tokens,
    cudaStream_t stream);

void launch_dense_load_kv(
    const uint8_t* page_pool,
    const int32_t* slot_mapping,
    half* out_key,
    half* out_value,
    const TQDensePageLayout& layout,
    const TQConfig& cfg,
    int num_tokens,
    cudaStream_t stream);
