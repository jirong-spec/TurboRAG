#pragma once
#include <cstddef>
#include "tq_config.h"

struct TQDensePageLayout {
    size_t elem_bytes = 2;
    size_t k_offset = 0;
    size_t v_offset = 0;
    size_t k_bytes = 0;
    size_t v_bytes = 0;
    size_t page_size_bytes = 0;
};

TQDensePageLayout make_tq_dense_page_layout(const TQConfig& cfg);

#ifdef __CUDACC__
#define TQ_HD __host__ __device__
#else
#define TQ_HD
#endif

inline TQ_HD size_t dense_token_head_offset(
    const TQDensePageLayout& layout,
    const TQConfig& cfg,
    int token_in_block,
    int head_idx) {
    return static_cast<size_t>(token_in_block * cfg.num_kv_heads + head_idx) *
           cfg.head_dim * layout.elem_bytes;
}

#undef TQ_HD
