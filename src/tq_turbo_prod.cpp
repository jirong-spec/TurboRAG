#include "tq_turbo_prod.cuh"

static inline size_t align_up_size(size_t x, size_t a) {
    return ((x + a - 1) / a) * a;
}

TQTurboProdPageLayout make_tq_turbo_prod_page_layout(const TQConfig& cfg) {
    TQTurboProdPageLayout layout{};

    layout.k3_bytes_per_token_head = (cfg.head_dim * 3 + 7) / 8;
    layout.kres_bytes_per_token_head = (cfg.head_dim + 7) / 8;
    layout.v4_bytes_per_token_head = (cfg.head_dim * 4 + 7) / 8;
    layout.scale_bytes_per_token_head = sizeof(half);

    const size_t token_heads = (size_t)cfg.block_size * cfg.num_kv_heads;

    const size_t k3_region        = token_heads * layout.k3_bytes_per_token_head;
    const size_t kres_region      = token_heads * layout.kres_bytes_per_token_head;
    const size_t kres_scale_region = token_heads * layout.scale_bytes_per_token_head;
    const size_t ks_region        = token_heads * layout.scale_bytes_per_token_head;
    const size_t v4_region        = token_heads * layout.v4_bytes_per_token_head;
    const size_t vs_region        = token_heads * layout.scale_bytes_per_token_head;

    size_t cursor = 0;
    layout.k3_codes_offset   = cursor; cursor += k3_region;
    layout.k_residual_offset = cursor; cursor += kres_region;

    // Align to half before FP16 scale regions
    cursor = align_up_size(cursor, alignof(half));
    layout.k_residual_scales_offset = cursor; cursor += kres_scale_region;
    layout.k_scales_offset          = cursor; cursor += ks_region;

    layout.v4_codes_offset = cursor; cursor += v4_region;

    cursor = align_up_size(cursor, alignof(half));
    layout.v_scales_offset = cursor; cursor += vs_region;

    layout.page_size_bytes = align_up_size(cursor, alignof(half));
    return layout;
}
