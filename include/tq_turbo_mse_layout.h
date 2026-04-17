#pragma once
#include <cstddef>
#include "tq_config.h"

struct TQTurbomsePageLayout {
    size_t code_bytes_per_token_head = 0;
    size_t norm_bytes_per_token_head = 0;

    size_t k_codes_offset = 0;
    size_t v_codes_offset = 0;
    size_t k_norms_offset = 0;
    size_t v_norms_offset = 0;

    size_t page_size_bytes = 0;
};

TQTurbomsePageLayout make_tq_turbo_mse_page_layout(const TQConfig& cfg);

#ifdef __CUDACC__
#define TQ_HD __host__ __device__
#else
#define TQ_HD
#endif

inline TQ_HD size_t turbo_token_head_code_offset(
    const TQTurbomsePageLayout& layout,
    int token_in_block,
    int head_idx,
    int num_kv_heads) {
    return static_cast<size_t>(token_in_block * num_kv_heads + head_idx) *
           layout.code_bytes_per_token_head;
}

inline TQ_HD size_t turbo_token_head_norm_offset(
    const TQTurbomsePageLayout& layout,
    int token_in_block,
    int head_idx,
    int num_kv_heads) {
    return static_cast<size_t>(token_in_block * num_kv_heads + head_idx) *
           layout.norm_bytes_per_token_head;
}

#undef TQ_HD
