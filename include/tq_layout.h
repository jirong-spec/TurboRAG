#pragma once
#include <cstddef>
#include "tq_config.h"

#ifdef __CUDACC__
#define TQ_HD __host__ __device__
#else
#define TQ_HD
#endif

struct TQPageLayout {
    size_t code_bytes_per_token_head = 0;
    size_t scale_bytes_per_token_head = 0;
    size_t k_codes_offset = 0;
    size_t v_codes_offset = 0;
    size_t k_scales_offset = 0;
    size_t v_scales_offset = 0;
    size_t k_codes_bytes = 0;
    size_t v_codes_bytes = 0;
    size_t k_scales_bytes = 0;
    size_t v_scales_bytes = 0;
    size_t page_size_bytes = 0;
};

TQPageLayout make_tq_page_layout(const TQConfig& cfg);

inline TQ_HD size_t token_head_code_offset(
    const TQPageLayout& layout,
    int token_in_block,
    int head_idx,
    int num_heads) {
    return static_cast<size_t>(token_in_block * num_heads + head_idx) *
           layout.code_bytes_per_token_head;
}

inline TQ_HD size_t token_head_scale_offset(
    const TQPageLayout& layout,
    int token_in_block,
    int head_idx,
    int num_heads) {
    return static_cast<size_t>(token_in_block * num_heads + head_idx) *
           layout.scale_bytes_per_token_head;
}

#undef TQ_HD
