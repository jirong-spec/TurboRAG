#include "tq_layout.h"
#include <stdexcept>

static size_t align_up(size_t x, size_t a) {
    return ((x + a - 1) / a) * a;
}

TQPageLayout make_tq_page_layout(const TQConfig& cfg) {
    if (cfg.nbits != 4) {
        throw std::runtime_error("Only int4 is supported in MVP");
    }
    if (cfg.group_size != cfg.head_dim) {
        throw std::runtime_error("MVP expects group_size == head_dim");
    }

    TQPageLayout l;
    l.code_bytes_per_token_head = static_cast<size_t>(cfg.head_dim * cfg.nbits) / 8;
    l.scale_bytes_per_token_head = (cfg.scale_type == TQScaleType::FP16) ? 2 : 4;

    const size_t token_heads = static_cast<size_t>(cfg.block_size) * cfg.num_kv_heads;

    l.k_codes_offset = 0;
    l.k_codes_bytes = token_heads * l.code_bytes_per_token_head;

    l.v_codes_offset = align_up(l.k_codes_offset + l.k_codes_bytes, cfg.page_alignment);
    l.v_codes_bytes = token_heads * l.code_bytes_per_token_head;

    l.k_scales_offset = align_up(l.v_codes_offset + l.v_codes_bytes, cfg.page_alignment);
    l.k_scales_bytes = token_heads * l.scale_bytes_per_token_head;

    l.v_scales_offset = align_up(l.k_scales_offset + l.k_scales_bytes, cfg.page_alignment);
    l.v_scales_bytes = token_heads * l.scale_bytes_per_token_head;

    l.page_size_bytes = align_up(l.v_scales_offset + l.v_scales_bytes, cfg.page_alignment);
    return l;
}
