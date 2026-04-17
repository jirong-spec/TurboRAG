#include "tq_qjl.cuh"

static inline size_t align_up_size(size_t x, size_t a) {
    return ((x + a - 1) / a) * a;
}

TQQJLPageLayout make_tq_qjl_page_layout(const TQConfig& cfg) {
    TQQJLPageLayout layout{};

    layout.k1_bytes_per_token_head    = (cfg.head_dim + 7) / 8;       // 1 bit/dim
    layout.v4_bytes_per_token_head    = (cfg.head_dim * 4 + 7) / 8;   // 4 bits/dim
    layout.scale_bytes_per_token_head = sizeof(half);                  // FP16

    const size_t token_heads = (size_t)cfg.block_size * cfg.num_kv_heads;

    const size_t k1_region = token_heads * layout.k1_bytes_per_token_head;
    const size_t ks_region = token_heads * layout.scale_bytes_per_token_head;
    const size_t v4_region = token_heads * layout.v4_bytes_per_token_head;
    const size_t vs_region = token_heads * layout.scale_bytes_per_token_head;

    size_t cursor = 0;
    layout.k1_codes_offset = cursor; cursor += k1_region;

    cursor = align_up_size(cursor, alignof(half));
    layout.k_scales_offset = cursor; cursor += ks_region;

    layout.v4_codes_offset = cursor; cursor += v4_region;

    cursor = align_up_size(cursor, alignof(half));
    layout.v_scales_offset = cursor; cursor += vs_region;

    layout.page_size_bytes = align_up_size(cursor, alignof(half));
    return layout;
}
