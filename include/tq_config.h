#pragma once
#include "tq_types.h"

struct TQConfig {
    int block_size = 16;
    int num_kv_heads = 8;
    int head_dim = 128;
    int nbits = 4;
    int group_size = 128;
    int page_alignment = 256;
    TQScaleType scale_type = TQScaleType::FP16;
    TQQuantMode quant_mode = TQQuantMode::MSE_INT4;
};
