#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "tq_layout.h"

void launch_tq_pack_kv(
    const half* key,
    const half* value,
    const int32_t* slot_mapping,
    uint8_t* page_pool,
    const TQPageLayout& layout,
    const TQConfig& cfg,
    int num_tokens,
    cudaStream_t stream);
