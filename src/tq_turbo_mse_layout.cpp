#include "tq_turbo_mse_layout.h"

static size_t align_up(size_t x, size_t a) {
    return ((x + a - 1) / a) * a;
}

TQTurbomsePageLayout make_tq_turbo_mse_page_layout(const TQConfig& cfg) {
    TQTurbomsePageLayout l;

    l.code_bytes_per_token_head = (cfg.head_dim * cfg.nbits + 7) / 8;
    l.norm_bytes_per_token_head = sizeof(uint16_t);

    const size_t num_token_heads =
        static_cast<size_t>(cfg.block_size) * cfg.num_kv_heads;

    const size_t k_codes_bytes = num_token_heads * l.code_bytes_per_token_head;
    const size_t v_codes_bytes = num_token_heads * l.code_bytes_per_token_head;
    const size_t k_norms_bytes = num_token_heads * l.norm_bytes_per_token_head;
    const size_t v_norms_bytes = num_token_heads * l.norm_bytes_per_token_head;

    l.k_codes_offset = 0;
    l.v_codes_offset = align_up(l.k_codes_offset + k_codes_bytes, cfg.page_alignment);
    l.k_norms_offset = align_up(l.v_codes_offset + v_codes_bytes, cfg.page_alignment);
    l.v_norms_offset = align_up(l.k_norms_offset + k_norms_bytes, cfg.page_alignment);
    l.page_size_bytes = align_up(l.v_norms_offset + v_norms_bytes, cfg.page_alignment);

    return l;
}
