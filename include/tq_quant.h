#pragma once
#include <cstdint>

inline __host__ __device__ int clamp_int(int x, int lo, int hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

inline __host__ __device__ uint8_t pack_int4(int lo, int hi) {
    return static_cast<uint8_t>((lo & 0xF) | ((hi & 0xF) << 4));
}

inline __host__ __device__ int unpack_int4_low(uint8_t x) {
    int v = x & 0xF;
    return (v >= 8) ? (v - 16) : v;
}

inline __host__ __device__ int unpack_int4_high(uint8_t x) {
    int v = (x >> 4) & 0xF;
    return (v >= 8) ? (v - 16) : v;
}
