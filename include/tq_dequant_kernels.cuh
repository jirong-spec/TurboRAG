#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "tq_layout.h"

void launch_tq_dequant_kv(
    const uint8_t* page_pool,
    const int32_t* slot_mapping,
    half* out_key,
    half* out_value,
    const TQPageLayout& layout,
    const TQConfig& cfg,
    int num_tokens,
    cudaStream_t stream);
