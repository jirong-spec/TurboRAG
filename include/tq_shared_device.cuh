#pragma once
// Bit-manipulation helpers shared across V6, QJL, and turbo_mse kernels.
// All functions are __device__ __forceinline__; safe to include from multiple
// translation units (no ODR issues — each TU gets its own inlined copy).

#include <cstdint>

// Sentinel values used in kernel initialisation.
static constexpr float kMinRMS      = 1e-8f;   // floor for per-token-head RMS scale
static constexpr float kInitMaxLogit = -1e30f; // initial max logit for online softmax

// ---------------------------------------------------------------------------
// 1-bit helpers
// ---------------------------------------------------------------------------

// Read one bit from a packed byte array.
__device__ __forceinline__ uint8_t unpack_1bit_get(const uint8_t* src, int idx) {
    return (src[idx >> 3] >> (idx & 7)) & 1u;
}

// Assemble one output byte from D sign bits stored in bits[].
// byte_idx selects which byte (covers dimensions byte_idx*8 .. byte_idx*8+7).
__device__ __forceinline__ uint8_t pack_1bit_byte_from_bits(const int* bits, int byte_idx, int D) {
    uint8_t out = 0;
    int base = byte_idx * 8;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        int idx = base + i;
        if (idx < D) out |= (uint8_t)((bits[idx] & 1) << i);
    }
    return out;
}

// ---------------------------------------------------------------------------
// 2-bit helpers
// ---------------------------------------------------------------------------

// Read one 2-bit code from a packed byte array (4 codes per byte).
__device__ __forceinline__ uint8_t unpack_2bit_get(const uint8_t* src, int idx) {
    return (src[idx >> 2] >> ((idx & 3) * 2)) & 3u;
}

// Assemble one output byte covering 4 consecutive 2-bit codes (byte_idx*4 .. byte_idx*4+3).
__device__ __forceinline__ uint8_t pack_2bit_byte_from_codes(const int* codes, int byte_idx, int D) {
    uint8_t out = 0;
    int base = byte_idx * 4;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int idx = base + i;
        if (idx < D) out |= (uint8_t)((codes[idx] & 3) << (i * 2));
    }
    return out;
}

// ---------------------------------------------------------------------------
// 4-bit (nibble) helpers
// ---------------------------------------------------------------------------

// Read one 4-bit nibble from a packed byte array.
// Even indices → lower nibble; odd indices → upper nibble.
__device__ __forceinline__ uint8_t unpack_4bit_get(const uint8_t* src, int idx) {
    uint8_t b = src[idx >> 1];
    return ((idx & 1) == 0) ? (b & 0x0F) : ((b >> 4) & 0x0F);
}
