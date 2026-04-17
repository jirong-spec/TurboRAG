#include "tq_dense_layout.h"

static size_t align_up(size_t x, size_t a) {
    return ((x + a - 1) / a) * a;
}

TQDensePageLayout make_tq_dense_page_layout(const TQConfig& cfg) {
    TQDensePageLayout l;
    const size_t token_heads = static_cast<size_t>(cfg.block_size) * cfg.num_kv_heads;
    const size_t bytes_per_token_head = static_cast<size_t>(cfg.head_dim) * l.elem_bytes;

    l.k_offset = 0;
    l.k_bytes = token_heads * bytes_per_token_head;
    l.v_offset = align_up(l.k_offset + l.k_bytes, cfg.page_alignment);
    l.v_bytes = token_heads * bytes_per_token_head;
    l.page_size_bytes = align_up(l.v_offset + l.v_bytes, cfg.page_alignment);
    return l;
}
