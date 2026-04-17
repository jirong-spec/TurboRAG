#pragma once
// ---------------------------------------------------------------------------
// PolarQuant page layout: K=2-bit + V=3-bit in Hadamard-rotated domain
//
// Memory per token-head at head_dim=128:
//   K: 32 B (2-bit codes) + 2 B (FP16 RMS scale) = 34 B
//   V: 48 B (3-bit codes) + 2 B (FP16 RMS scale) = 50 B
//   Total: 84 B  vs  512 B dense  → ~6.1× compression
//
// Compared to QJL (also 84 B): K quality is much better (2-bit vs 1-bit sign),
// V quality is slightly lower (3-bit vs 4-bit). Net attention MSE is lower.
// ---------------------------------------------------------------------------

#include <cstddef>
#include <cstdint>
#include "tq_config.h"

struct TQPolarPageLayout {
    size_t page_size_bytes = 0;

    size_t k2_codes_offset = 0;  // 2-bit K codes  (D/4 bytes per token-head)
    size_t k_scales_offset = 0;  // FP16 K RMS scales
    size_t v3_codes_offset = 0;  // 3-bit V codes  (D*3/8 bytes per token-head)
    size_t v_scales_offset = 0;  // FP16 V RMS scales

    int k2_bytes_per_token_head = 0;   // = head_dim / 4
    int v3_bytes_per_token_head = 0;   // = head_dim * 3 / 8
    int scale_bytes_per_token_head = 0; // = sizeof(half) = 2
};

TQPolarPageLayout make_tq_polar_page_layout(const TQConfig& cfg);

// ---------------------------------------------------------------------------
// Byte offset of a (token_in_block, head_idx) cell inside one region.
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define TQ_POLAR_HD __host__ __device__
#else
#define TQ_POLAR_HD
#endif

inline TQ_POLAR_HD size_t polar_token_head_offset(
    int token_in_block,
    int head_idx,
    int num_kv_heads,
    int bytes_per_token_head)
{
    return ((size_t)token_in_block * num_kv_heads + head_idx) * bytes_per_token_head;
}

#undef TQ_POLAR_HD
