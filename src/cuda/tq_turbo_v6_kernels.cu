#include "tq_turbo_v6_kernels.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

namespace {

__device__ __forceinline__ float h2f(half x) { return __half2float(x); }
__device__ __forceinline__ half f2h(float x) { return __float2half(x); }

template <typename T>
__device__ __forceinline__ T clamp_val(T x, T lo, T hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

__device__ __forceinline__ float sign_flip(int idx) {
    unsigned x = static_cast<unsigned>(idx) * 1103515245u + 12345u;
    return (x & 1u) ? 1.0f : -1.0f;
}

/*
K: 3-bit base codebook
*/
__device__ __constant__ float kK3Codebook[8] = {
    -2.40f, -1.45f, -0.82f, -0.24f,
     0.24f,  0.82f,  1.45f,  2.40f
};

/*
V: 4-bit codebook
*/
__device__ __constant__ float kV4Codebook[16] = {
    -2.7326f, -2.0690f, -1.6180f, -1.2562f,
    -0.9423f, -0.6568f, -0.3880f, -0.1284f,
     0.1284f,  0.3880f,  0.6568f,  0.9423f,
     1.2562f,  1.6180f,  2.0690f,  2.7326f
};

__device__ __forceinline__ int nearest_k3_idx(float x) {
    int best = 0;
    float best_dist = fabsf(x - kK3Codebook[0]);
    #pragma unroll
    for (int i = 1; i < 8; ++i) {
        float d = fabsf(x - kK3Codebook[i]);
        if (d < best_dist) {
            best_dist = d;
            best = i;
        }
    }
    return best;
}

__device__ __forceinline__ int nearest_v4_idx(float x) {
    int best = 0;
    float best_dist = fabsf(x - kV4Codebook[0]);
    #pragma unroll
    for (int i = 1; i < 16; ++i) {
        float d = fabsf(x - kV4Codebook[i]);
        if (d < best_dist) {
            best_dist = d;
            best = i;
        }
    }
    return best;
}

__device__ __forceinline__ void pack_3bit_set(uint8_t* dst, int idx, uint8_t val) {
    int bit = idx * 3;
    int byte = bit >> 3;
    int off = bit & 7;

    unsigned int x = (unsigned int)(val & 0x7);

    if (off <= 5) {
        uint8_t mask = (uint8_t)(0x7u << off);
        dst[byte] = (uint8_t)((dst[byte] & ~mask) | ((x << off) & mask));
    } else {
        int lo_bits = 8 - off;
        int hi_bits = 3 - lo_bits;

        uint8_t mask0 = (uint8_t)(((1u << lo_bits) - 1u) << off);
        uint8_t mask1 = (uint8_t)((1u << hi_bits) - 1u);

        dst[byte]     = (uint8_t)((dst[byte] & ~mask0) | (((x & ((1u << lo_bits) - 1u)) << off) & mask0));
        dst[byte + 1] = (uint8_t)((dst[byte + 1] & ~mask1) | ((x >> lo_bits) & mask1));
    }
}

__device__ __forceinline__ uint8_t unpack_3bit_get(const uint8_t* src, int idx) {
    int bit = idx * 3;
    int byte = bit >> 3;
    int off = bit & 7;
    unsigned int x = src[byte];
    if (off > 5) {
        x |= ((unsigned int)src[byte + 1] << 8);
    }
    return (x >> off) & 0x7;
}

__device__ __forceinline__ void pack_1bit_set(uint8_t* dst, int idx, uint8_t bitv) {
    int byte = idx >> 3;
    int off = idx & 7;
    if (bitv) dst[byte] |= (1u << off);
}

__device__ __forceinline__ uint8_t unpack_1bit_get(const uint8_t* src, int idx) {
    int byte = idx >> 3;
    int off = idx & 7;
    return (src[byte] >> off) & 1u;
}

__device__ __forceinline__ void pack_4bit_set(uint8_t* dst, int idx, uint8_t val) {
    int byte = idx >> 1;
    if ((idx & 1) == 0) {
        dst[byte] = (dst[byte] & 0xF0) | (val & 0x0F);
    } else {
        dst[byte] = (dst[byte] & 0x0F) | ((val & 0x0F) << 4);
    }
}

__device__ __forceinline__ uint8_t unpack_4bit_get(const uint8_t* src, int idx) {
    int byte = idx >> 1;
    return ((idx & 1) == 0) ? (src[byte] & 0x0F) : ((src[byte] >> 4) & 0x0F);
}

__device__ __forceinline__ uint8_t pack_3bit_byte_from_codes(const int* codes, int byte_idx, int D) {
    uint8_t out = 0;
    int bit_base = byte_idx * 8;

    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        int global_bit = bit_base + k;
        int code_idx = global_bit / 3;
        if (code_idx < D) {
            int bit_in_code = global_bit % 3;
            uint8_t bit = (uint8_t)((codes[code_idx] >> bit_in_code) & 1);
            out |= (uint8_t)(bit << k);
        }
    }
    return out;
}

__device__ __forceinline__ uint8_t pack_1bit_byte_from_bits(const int* bits, int byte_idx, int D) {
    uint8_t out = 0;
    int base = byte_idx * 8;

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        int idx = base + i;
        if (idx < D) {
            out |= (uint8_t)((bits[idx] & 1) << i);
        }
    }
    return out;
}


template<int MAX_D>
__device__ void hadamard_inplace(float* x, int D) {
    for (int len = 1; len < D; len <<= 1) {
        int tid = threadIdx.x;
        int butterfly = D >> 1;
        if (tid < butterfly) {
            int group = tid / len;
            int offset = tid % len;
            int i0 = group * (len << 1) + offset;
            int i1 = i0 + len;
            float a = x[i0];
            float b = x[i1];
            x[i0] = a + b;
            x[i1] = a - b;
        }
        __syncthreads();
    }
}

template<int MAX_D>
__global__ void turbo_v6_pack_kv_kernel(
    const half* __restrict__ key,
    const half* __restrict__ value,
    const int32_t* __restrict__ slot_mapping,
    uint8_t* __restrict__ page_pool,
    TQTurboV6PageLayout layout,
    TQConfig cfg,
    int num_tokens,
    float* __restrict__ debug_k_rot,
    float* __restrict__ debug_v_rot,
    float* __restrict__ debug_kn,
    float* __restrict__ debug_vn,
    int* __restrict__ debug_kidx,
    int* __restrict__ debug_vidx) {

    int token_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int tid = threadIdx.x;

    if (token_idx >= num_tokens || head_idx >= cfg.num_kv_heads) return;
    if (cfg.head_dim > MAX_D) return;

    int D = cfg.head_dim;
    if (tid >= D) return;

    int slot = slot_mapping[token_idx];
    int physical_block = slot / cfg.block_size;
    int token_in_block = slot % cfg.block_size;

    uint8_t* page_base = page_pool + (size_t)physical_block * layout.page_size_bytes;

    uint8_t* k3_codes = page_base + layout.k3_codes_offset +
        turbo_v6_token_head_offset(token_in_block, head_idx, cfg.num_kv_heads, layout.k3_bytes_per_token_head);
    uint8_t* kres = page_base + layout.k_residual_offset +
        turbo_v6_token_head_offset(token_in_block, head_idx, cfg.num_kv_heads, layout.kres_bytes_per_token_head);
    half* kscale = reinterpret_cast<half*>(
        page_base + layout.k_scales_offset +
        turbo_v6_token_head_offset(token_in_block, head_idx, cfg.num_kv_heads, layout.scale_bytes_per_token_head));

    uint8_t* v4_codes = page_base + layout.v4_codes_offset +
        turbo_v6_token_head_offset(token_in_block, head_idx, cfg.num_kv_heads, layout.v4_bytes_per_token_head);
    half* vscale = reinterpret_cast<half*>(
        page_base + layout.v_scales_offset +
        turbo_v6_token_head_offset(token_in_block, head_idx, cfg.num_kv_heads, layout.scale_bytes_per_token_head));

    extern __shared__ float smem[];
    float* sk = smem;
    float* sv = smem + MAX_D;
    float* red = smem + 2 * MAX_D;

    __shared__ int kidx_s[256];
    __shared__ int vidx_s[256];
    __shared__ int krbit_s[256];

    int base = (token_idx * cfg.num_kv_heads + head_idx) * D;

    if (tid < layout.k3_bytes_per_token_head) k3_codes[tid] = 0;
    if (tid < layout.kres_bytes_per_token_head) kres[tid] = 0;
    if (tid < layout.v4_bytes_per_token_head) v4_codes[tid] = 0;
    red[tid] = 0.0f;
    __syncthreads();

    sk[tid] = h2f(key[base + tid]) * sign_flip(tid);
    sv[tid] = h2f(value[base + tid]) * sign_flip(tid);
    __syncthreads();

    hadamard_inplace<MAX_D>(sk, D);
    hadamard_inplace<MAX_D>(sv, D);

    float inv_sqrt_d = rsqrtf((float)D);
    sk[tid] *= inv_sqrt_d;
    sv[tid] *= inv_sqrt_d;

    red[tid] = sk[tid] * sk[tid];
    __syncthreads();
    for (int stride = D >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) red[tid] += red[tid + stride];
        __syncthreads();
    }
    float krms = sqrtf(red[0] / (float)D);
    if (krms < 1e-8f) krms = 1e-8f;
    if (tid == 0) *kscale = f2h(krms);
    __syncthreads();

    red[tid] = sv[tid] * sv[tid];
    __syncthreads();
    for (int stride = D >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) red[tid] += red[tid + stride];
        __syncthreads();
    }
    float vrms = sqrtf(red[0] / (float)D);
    if (vrms < 1e-8f) vrms = 1e-8f;
    if (tid == 0) *vscale = f2h(vrms);
    __syncthreads();

    if (debug_k_rot) debug_k_rot[base + tid] = sk[tid];
    if (debug_v_rot) debug_v_rot[base + tid] = sv[tid];

    float kn = clamp_val(sk[tid] / krms, -2.5f, 2.5f);
    float vn = clamp_val(sv[tid] / vrms, -2.75f, 2.75f);

    int kidx = nearest_k3_idx(kn);
    float kbase = kK3Codebook[kidx];
    int krbit = (kn >= kbase) ? 1 : 0;

    int vidx = nearest_v4_idx(vn);

    kidx_s[tid] = kidx;
    krbit_s[tid] = krbit;
    vidx_s[tid] = vidx;

    if (debug_kn) debug_kn[base + tid] = kn;
    if (debug_vn) debug_vn[base + tid] = vn;
    if (debug_kidx) debug_kidx[base + tid] = kidx;
    if (debug_vidx) debug_vidx[base + tid] = vidx;

    __syncthreads();

        // K: byte-owner parallel pack, one thread owns one output byte.
    if (tid < layout.k3_bytes_per_token_head) {
        k3_codes[tid] = pack_3bit_byte_from_codes(kidx_s, tid, D);
    }

    if (tid < layout.kres_bytes_per_token_head) {
        kres[tid] = pack_1bit_byte_from_bits(krbit_s, tid, D);
    }

    // V: pair-wise pack, one thread owns one byte.
    if (tid < (D >> 1)) {
        int i0 = 2 * tid;
        int i1 = i0 + 1;
        uint8_t v0 = (uint8_t)vidx_s[i0];
        uint8_t v1 = (uint8_t)vidx_s[i1];
        v4_codes[tid] = (uint8_t)((v1 << 4) | v0);
    }
}

template<int MAX_D>
__global__ void turbo_v6_dequant_kv_kernel(
    const uint8_t* __restrict__ page_pool,
    const int32_t* __restrict__ slot_mapping,
    half* __restrict__ out_key,
    half* __restrict__ out_value,
    TQTurboV6PageLayout layout,
    TQConfig cfg,
    int num_tokens) {

    int token_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int tid = threadIdx.x;

    if (token_idx >= num_tokens || head_idx >= cfg.num_kv_heads) return;
    if (cfg.head_dim > MAX_D) return;

    int D = cfg.head_dim;
    if (tid >= D) return;

    int slot = slot_mapping[token_idx];
    int physical_block = slot / cfg.block_size;
    int token_in_block = slot % cfg.block_size;

    const uint8_t* page_base = page_pool + (size_t)physical_block * layout.page_size_bytes;

    const uint8_t* k3_codes = page_base + layout.k3_codes_offset +
        turbo_v6_token_head_offset(token_in_block, head_idx, cfg.num_kv_heads, layout.k3_bytes_per_token_head);
    const uint8_t* kres = page_base + layout.k_residual_offset +
        turbo_v6_token_head_offset(token_in_block, head_idx, cfg.num_kv_heads, layout.kres_bytes_per_token_head);
    const half* kscale = reinterpret_cast<const half*>(
        page_base + layout.k_scales_offset +
        turbo_v6_token_head_offset(token_in_block, head_idx, cfg.num_kv_heads, layout.scale_bytes_per_token_head));

    const uint8_t* v4_codes = page_base + layout.v4_codes_offset +
        turbo_v6_token_head_offset(token_in_block, head_idx, cfg.num_kv_heads, layout.v4_bytes_per_token_head);
    const half* vscale = reinterpret_cast<const half*>(
        page_base + layout.v_scales_offset +
        turbo_v6_token_head_offset(token_in_block, head_idx, cfg.num_kv_heads, layout.scale_bytes_per_token_head));

    extern __shared__ float smem[];
    float* sk = smem;
    float* sv = smem + MAX_D;

    float ks = h2f(*kscale);
    float vs = h2f(*vscale);

    uint8_t kidx = unpack_3bit_get(k3_codes, tid);
    uint8_t kr = unpack_1bit_get(kres, tid);
    uint8_t vidx = unpack_4bit_get(v4_codes, tid);
    
    float kresidual = kr ? 0.125f : -0.125f;
    float kval = kK3Codebook[kidx] + kresidual;
    float vval = kV4Codebook[vidx];

    sk[tid] = kval * ks;
    sv[tid] = vval * vs;
    __syncthreads();

    hadamard_inplace<MAX_D>(sk, D);
    hadamard_inplace<MAX_D>(sv, D);

    float inv_sqrt_d = rsqrtf((float)D);
    int base = (token_idx * cfg.num_kv_heads + head_idx) * D;

    out_key[base + tid] = f2h(sk[tid] * inv_sqrt_d * sign_flip(tid));
    out_value[base + tid] = f2h(sv[tid] * inv_sqrt_d * sign_flip(tid));
}

template<int MAX_D>
__global__ void turbo_v6_fused_attention_logits_kernel(
    const half* __restrict__ query,
    const uint8_t* __restrict__ page_pool,
    const int32_t* __restrict__ slot_mapping,
    float* __restrict__ logits,
    TQTurboV6PageLayout layout,
    TQConfig cfg,
    int num_queries,
    int num_kv_tokens) {

    /*
    Skeleton version:
    - blockIdx.x = query token
    - blockIdx.y = kv head
    - each block computes logits[q, :, head] over kv tokens
    - one thread handles one dim
    - directly consumes compressed K, does NOT materialize full K in global memory
    */

    int q_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int tid = threadIdx.x;

    if (q_idx >= num_queries || head_idx >= cfg.num_kv_heads) return;
    if (cfg.head_dim > MAX_D) return;

    int D = cfg.head_dim;
    if (tid >= D) return;

    extern __shared__ float smem[];
    float* qrot = smem;
    float* kred = smem + MAX_D;
    float* red = smem + 2 * MAX_D;

    int qbase = (q_idx * cfg.num_kv_heads + head_idx) * D;
    qrot[tid] = h2f(query[qbase + tid]) * sign_flip(tid);
    __syncthreads();

    hadamard_inplace<MAX_D>(qrot, D);
    float inv_sqrt_d = rsqrtf((float)D);
    qrot[tid] *= inv_sqrt_d;
    __syncthreads();

    for (int kv_idx = 0; kv_idx < num_kv_tokens; ++kv_idx) {
        int slot = slot_mapping[kv_idx];
        int physical_block = slot / cfg.block_size;
        int token_in_block = slot % cfg.block_size;

        const uint8_t* page_base = page_pool + (size_t)physical_block * layout.page_size_bytes;

        const uint8_t* k3_codes = page_base + layout.k3_codes_offset +
            turbo_v6_token_head_offset(token_in_block, head_idx, cfg.num_kv_heads, layout.k3_bytes_per_token_head);
        const uint8_t* kres = page_base + layout.k_residual_offset +
            turbo_v6_token_head_offset(token_in_block, head_idx, cfg.num_kv_heads, layout.kres_bytes_per_token_head);
        const half* kscale = reinterpret_cast<const half*>(
            page_base + layout.k_scales_offset +
            turbo_v6_token_head_offset(token_in_block, head_idx, cfg.num_kv_heads, layout.scale_bytes_per_token_head));

        float ks = h2f(*kscale);
        uint8_t kidx = unpack_3bit_get(k3_codes, tid);
        uint8_t kr = unpack_1bit_get(kres, tid);

        float kresidual = kr ? 0.125f : -0.125f;
        float kval = kK3Codebook[kidx] + kresidual;

        /*
        Approximate rotated-domain K
        */
        kred[tid] = kval * ks;
        red[tid] = qrot[tid] * kred[tid];
        __syncthreads();

        for (int stride = D >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) red[tid] += red[tid + stride];
            __syncthreads();
        }

        if (tid == 0) {
            logits[(q_idx * cfg.num_kv_heads + head_idx) * num_kv_tokens + kv_idx] = red[0];
        }
        __syncthreads();
    }
}

} // namespace

void launch_tq_turbo_v6_pack_kv(
    const half* key,
    const half* value,
    const int32_t* slot_mapping,
    uint8_t* page_pool,
    const TQTurboV6PageLayout& layout,
    const TQConfig& cfg,
    int num_tokens,
    cudaStream_t stream,
    float* debug_k_rot,
    float* debug_v_rot,
    float* debug_kn,
    float* debug_vn,
    int* debug_kidx,
    int* debug_vidx) {

    dim3 grid(num_tokens, cfg.num_kv_heads);
    if (cfg.head_dim <= 128) {
        int threads = cfg.head_dim;
        size_t shmem = sizeof(float) * (3 * 128);
        turbo_v6_pack_kv_kernel<128><<<grid, threads, shmem, stream>>>(
            key, value, slot_mapping, page_pool, layout, cfg, num_tokens,
            debug_k_rot, debug_v_rot, debug_kn, debug_vn, debug_kidx, debug_vidx);
    }
}

void launch_tq_turbo_v6_dequant_kv(
    const uint8_t* page_pool,
    const int32_t* slot_mapping,
    half* out_key,
    half* out_value,
    const TQTurboV6PageLayout& layout,
    const TQConfig& cfg,
    int num_tokens,
    cudaStream_t stream) {

    dim3 grid(num_tokens, cfg.num_kv_heads);
    if (cfg.head_dim <= 128) {
        int threads = cfg.head_dim;
        size_t shmem = sizeof(float) * (2 * 128);
        turbo_v6_dequant_kv_kernel<128><<<grid, threads, shmem, stream>>>(
            page_pool, slot_mapping, out_key, out_value, layout, cfg, num_tokens);
    }
}

void launch_tq_turbo_v6_fused_attention_logits(
    const half* query,
    const uint8_t* page_pool,
    const int32_t* slot_mapping,
    float* logits,
    const TQTurboV6PageLayout& layout,
    const TQConfig& cfg,
    int num_queries,
    int num_kv_tokens,
    cudaStream_t stream) {

    dim3 grid(num_queries, cfg.num_kv_heads);
    if (cfg.head_dim <= 128) {
        int threads = cfg.head_dim;
        size_t shmem = sizeof(float) * (3 * 128);
        turbo_v6_fused_attention_logits_kernel<128><<<grid, threads, shmem, stream>>>(
            query, page_pool, slot_mapping, logits, layout, cfg, num_queries, num_kv_tokens);
    }
}
