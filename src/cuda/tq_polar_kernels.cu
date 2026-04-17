#include "tq_polar.cuh"
#include "tq_shared_device.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdexcept>

namespace {

__device__ __forceinline__ float h2f(half x) { return __half2float(x); }
__device__ __forceinline__ half  f2h(float x) { return __float2half(x); }

template<typename T>
__device__ __forceinline__ T clamp_val(T x, T lo, T hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

__device__ __forceinline__ float sign_flip(int idx) {
    unsigned x = static_cast<unsigned>(idx) * 1103515245u + 12345u;
    return (x & 1u) ? 1.0f : -1.0f;
}

// ---------------------------------------------------------------------------
// Codebooks
// ---------------------------------------------------------------------------

// K 2-bit: 4-level Lloyd-Max optimal for N(0,1)
__device__ __constant__ float kK2Codebook[4] = {
    -1.5104f, -0.4528f, 0.4528f, 1.5104f
};

// V 3-bit: 8-level Lloyd-Max optimal for N(0,1) — same as turbo_prod K3
__device__ __constant__ float kV3Codebook[8] = {
    -2.1513f, -1.34326f, -0.755526f, -0.244919f,
     0.244919f, 0.755526f,  1.34326f,  2.1513f
};

// Nearest K2 centroid via threshold comparison (faster than brute-force for 4 levels).
// Optimal threshold for Lloyd-Max N(0,1) 4-level: ±0.9816.
__device__ __forceinline__ int nearest_k2_idx(float x) {
    if (x < -0.9816f) return 0;
    if (x <  0.0f)    return 1;
    if (x <  0.9816f) return 2;
    return 3;
}

// Nearest V3 centroid via brute-force (8 levels).
__device__ __forceinline__ int nearest_v3_idx(float x) {
    int best = 0;
    float best_dist = fabsf(x - kV3Codebook[0]);
    #pragma unroll
    for (int i = 1; i < 8; ++i) {
        float d = fabsf(x - kV3Codebook[i]);
        if (d < best_dist) { best_dist = d; best = i; }
    }
    return best;
}

// ---------------------------------------------------------------------------
// 3-bit pack / unpack (bit-stream format: 3 consecutive bits per code).
// Replicated here per CUDA anonymous-namespace requirement.
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint8_t unpack_3bit_get(const uint8_t* src, int idx) {
    int bit  = idx * 3;
    int byte = bit >> 3;
    int off  = bit & 7;
    unsigned int x = src[byte];
    if (off > 5) x |= ((unsigned int)src[byte + 1] << 8);
    return (uint8_t)((x >> off) & 0x7u);
}

__device__ __forceinline__ uint8_t pack_3bit_byte_from_codes(
    const int* codes, int byte_idx, int D)
{
    uint8_t out = 0;
    int bit_base = byte_idx * 8;
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        int global_bit = bit_base + k;
        int code_idx   = global_bit / 3;
        if (code_idx < D) {
            int bit_in_code = global_bit % 3;
            out |= (uint8_t)(((codes[code_idx] >> bit_in_code) & 1) << k);
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// In-place normalised Walsh-Hadamard Transform (butterfly on shared memory).
// Requires all D threads alive; leaves them synchronised at exit.
// Replicated per translation unit (same as V6 / QJL).
// ---------------------------------------------------------------------------
template<int MAX_D>
__device__ void hadamard_inplace(float* x, int D) {
    for (int len = 1; len < D; len <<= 1) {
        int tid      = threadIdx.x;
        int butterfly = D >> 1;
        if (tid < butterfly) {
            int group  = tid / len;
            int offset = tid % len;
            int i0 = group * (len << 1) + offset;
            int i1 = i0 + len;
            float a = x[i0], b = x[i1];
            x[i0] = a + b;
            x[i1] = a - b;
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// Pack kernel
//
// grid = (num_tokens, num_kv_heads),  threads = head_dim
// Shared memory (extern): sk[MAX_D] | sv[MAX_D] | red[MAX_D]
// ---------------------------------------------------------------------------
template<int MAX_D>
__global__ void polar_pack_kv_kernel(
    const half* __restrict__   key,
    const half* __restrict__   value,
    const int32_t* __restrict__ slot_mapping,
    uint8_t* __restrict__      page_pool,
    TQPolarPageLayout          layout,
    TQConfig                   cfg,
    int                        num_tokens)
{
    int token_idx = blockIdx.x;
    int head_idx  = blockIdx.y;
    int tid       = threadIdx.x;

    if (token_idx >= num_tokens || head_idx >= cfg.num_kv_heads) return;
    if (cfg.head_dim > MAX_D) return;

    int D = cfg.head_dim;
    if (tid >= D) return;

    int slot           = slot_mapping[token_idx];
    int physical_block = slot / cfg.block_size;
    int token_in_block = slot % cfg.block_size;

    uint8_t* page_base = page_pool + (size_t)physical_block * layout.page_size_bytes;

    uint8_t* k2_codes = page_base + layout.k2_codes_offset +
        polar_token_head_offset(token_in_block, head_idx, cfg.num_kv_heads,
                                layout.k2_bytes_per_token_head);
    half* kscale = reinterpret_cast<half*>(
        page_base + layout.k_scales_offset +
        polar_token_head_offset(token_in_block, head_idx, cfg.num_kv_heads,
                                layout.scale_bytes_per_token_head));

    uint8_t* v3_codes = page_base + layout.v3_codes_offset +
        polar_token_head_offset(token_in_block, head_idx, cfg.num_kv_heads,
                                layout.v3_bytes_per_token_head);
    half* vscale = reinterpret_cast<half*>(
        page_base + layout.v_scales_offset +
        polar_token_head_offset(token_in_block, head_idx, cfg.num_kv_heads,
                                layout.scale_bytes_per_token_head));

    extern __shared__ float smem[];
    float* sk  = smem;           // [MAX_D] rotated K
    float* sv  = smem + MAX_D;   // [MAX_D] rotated V
    float* red = smem + 2*MAX_D; // [MAX_D] reduction scratch

    __shared__ int kidx_s[MAX_D]; // 2-bit K indices
    __shared__ int vidx_s[MAX_D]; // 3-bit V indices

    const int base = (token_idx * cfg.num_kv_heads + head_idx) * D;
    const float inv_sqrt_d = rsqrtf((float)D);

    // ---- Step 1: sign-flip + WHT + normalise K and V ----------------------
    sk[tid] = h2f(key  [base + tid]) * sign_flip(tid);
    sv[tid] = h2f(value[base + tid]) * sign_flip(tid);
    __syncthreads();

    hadamard_inplace<MAX_D>(sk, D);
    hadamard_inplace<MAX_D>(sv, D);

    sk[tid] *= inv_sqrt_d;
    sv[tid] *= inv_sqrt_d;

    // ---- Step 2: K RMS scale ----------------------------------------------
    red[tid] = sk[tid] * sk[tid];
    __syncthreads();
    for (int stride = D >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) red[tid] += red[tid + stride];
        __syncthreads();
    }
    float krms = sqrtf(red[0] / (float)D);
    if (krms < kMinRMS) krms = kMinRMS;
    if (tid == 0) *kscale = f2h(krms);
    __syncthreads();

    // ---- Step 3: V RMS scale ----------------------------------------------
    red[tid] = sv[tid] * sv[tid];
    __syncthreads();
    for (int stride = D >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) red[tid] += red[tid + stride];
        __syncthreads();
    }
    float vrms = sqrtf(red[0] / (float)D);
    if (vrms < kMinRMS) vrms = kMinRMS;
    if (tid == 0) *vscale = f2h(vrms);
    __syncthreads();

    // ---- Step 4: quantise -------------------------------------------------
    kidx_s[tid] = nearest_k2_idx(sk[tid] / krms);
    vidx_s[tid] = nearest_v3_idx(sv[tid] / vrms);
    __syncthreads();

    // ---- Step 5: pack K (2-bit, 4 codes/byte) ----------------------------
    if (tid < layout.k2_bytes_per_token_head)
        k2_codes[tid] = pack_2bit_byte_from_codes(kidx_s, tid, D);

    // ---- Step 6: pack V (3-bit bit-stream) --------------------------------
    for (int byte_i = tid; byte_i < layout.v3_bytes_per_token_head; byte_i += D)
        v3_codes[byte_i] = pack_3bit_byte_from_codes(vidx_s, byte_i, D);
}

// ---------------------------------------------------------------------------
// Dequant kernel
//
// grid = (num_tokens, num_kv_heads),  threads = head_dim
// Shared memory (extern): smem[MAX_D]  (reused for K then V)
// ---------------------------------------------------------------------------
template<int MAX_D>
__global__ void polar_dequant_kv_kernel(
    const uint8_t* __restrict__  page_pool,
    const int32_t* __restrict__  slot_mapping,
    half* __restrict__           out_key,
    half* __restrict__           out_value,
    TQPolarPageLayout            layout,
    TQConfig                     cfg,
    int                          num_tokens)
{
    int token_idx = blockIdx.x;
    int head_idx  = blockIdx.y;
    int tid       = threadIdx.x;

    if (token_idx >= num_tokens || head_idx >= cfg.num_kv_heads) return;
    if (cfg.head_dim > MAX_D) return;

    int D = cfg.head_dim;
    if (tid >= D) return;

    int slot           = slot_mapping[token_idx];
    int physical_block = slot / cfg.block_size;
    int token_in_block = slot % cfg.block_size;

    const uint8_t* page_base = page_pool + (size_t)physical_block * layout.page_size_bytes;

    const uint8_t* k2_codes = page_base + layout.k2_codes_offset +
        polar_token_head_offset(token_in_block, head_idx, cfg.num_kv_heads,
                                layout.k2_bytes_per_token_head);
    const half* k_scale_ptr = reinterpret_cast<const half*>(
        page_base + layout.k_scales_offset +
        polar_token_head_offset(token_in_block, head_idx, cfg.num_kv_heads,
                                layout.scale_bytes_per_token_head));

    const uint8_t* v3_codes = page_base + layout.v3_codes_offset +
        polar_token_head_offset(token_in_block, head_idx, cfg.num_kv_heads,
                                layout.v3_bytes_per_token_head);
    const half* v_scale_ptr = reinterpret_cast<const half*>(
        page_base + layout.v_scales_offset +
        polar_token_head_offset(token_in_block, head_idx, cfg.num_kv_heads,
                                layout.scale_bytes_per_token_head));

    extern __shared__ float smem[]; // [MAX_D]

    const int base        = (token_idx * cfg.num_kv_heads + head_idx) * D;
    const float inv_sqrt_d = rsqrtf((float)D);

    // ---- Decode K: 2-bit → WHT⁻¹ → sign-unflip → FP16 -------------------
    float krms = h2f(*k_scale_ptr);
    smem[tid]  = kK2Codebook[unpack_2bit_get(k2_codes, tid)] * krms;
    __syncthreads();

    hadamard_inplace<MAX_D>(smem, D);

    out_key[base + tid] = f2h(smem[tid] * inv_sqrt_d * sign_flip(tid));
    __syncthreads();

    // ---- Decode V: 3-bit → WHT⁻¹ → sign-unflip → FP16 -------------------
    float vrms = h2f(*v_scale_ptr);
    smem[tid]  = kV3Codebook[unpack_3bit_get(v3_codes, tid)] * vrms;
    __syncthreads();

    hadamard_inplace<MAX_D>(smem, D);

    out_value[base + tid] = f2h(smem[tid] * inv_sqrt_d * sign_flip(tid));
}

// ---------------------------------------------------------------------------
// Fused attention: online softmax → weighted-V sum → inverse rotation.
//
// grid = (num_queries, num_kv_heads),  threads = head_dim
// Shared memory (extern): qrot[MAX_D] | vaccum[MAX_D] | red[MAX_D]
//
// K decoded as 2-bit, V decoded as 3-bit (both in Hadamard-rotated domain).
// Logit convention: <q, k>  (no 1/sqrt(D)), matching V6 / QJL convention.
// ---------------------------------------------------------------------------
template<int MAX_D>
__global__ void polar_fused_attn_online_kernel(
    const half* __restrict__    query,
    const uint8_t* __restrict__ page_pool,
    const int32_t* __restrict__ slot_mapping,
    half* __restrict__          output,
    TQPolarPageLayout           layout,
    TQConfig                    cfg,
    int                         num_queries,
    int                         num_kv_tokens)
{
    int q_idx    = blockIdx.x;
    int head_idx = blockIdx.y;
    int tid      = threadIdx.x;

    if (q_idx >= num_queries || head_idx >= cfg.num_kv_heads) return;
    if (cfg.head_dim > MAX_D) return;

    int D = cfg.head_dim;
    if (tid >= D) return;

    const float inv_sqrt_d = rsqrtf((float)D);

    extern __shared__ float smem[];
    float* qrot   = smem;           // [MAX_D] rotated query
    float* vaccum = smem + MAX_D;   // [MAX_D] weighted V accumulation
    float* red    = smem + 2*MAX_D; // [MAX_D] reduction scratch

    __shared__ float sh_m, sh_l, sh_a, sh_b;

    // ---- Rotate query: sign_flip + WHT + 1/sqrt(D) ------------------------
    int qbase = (q_idx * cfg.num_kv_heads + head_idx) * D;
    qrot[tid] = h2f(query[qbase + tid]) * sign_flip(tid);
    __syncthreads();

    hadamard_inplace<MAX_D>(qrot, D);
    qrot[tid] *= inv_sqrt_d;
    __syncthreads();

    vaccum[tid] = 0.0f;
    if (tid == 0) { sh_m = kInitMaxLogit; sh_l = 0.0f; }
    __syncthreads();

    for (int t = 0; t < num_kv_tokens; ++t) {
        int slot           = slot_mapping[t];
        int physical_block = slot / cfg.block_size;
        int token_in_block = slot % cfg.block_size;

        const uint8_t* page_base = page_pool +
            (size_t)physical_block * layout.page_size_bytes;

        // ---- 2-bit K decode → dot product ----------------------------------
        const uint8_t* k2_codes = page_base + layout.k2_codes_offset +
            polar_token_head_offset(token_in_block, head_idx, cfg.num_kv_heads,
                                    layout.k2_bytes_per_token_head);
        const half* k_scale_ptr = reinterpret_cast<const half*>(
            page_base + layout.k_scales_offset +
            polar_token_head_offset(token_in_block, head_idx, cfg.num_kv_heads,
                                    layout.scale_bytes_per_token_head));

        float kval = kK2Codebook[unpack_2bit_get(k2_codes, tid)] * h2f(*k_scale_ptr);

        red[tid] = qrot[tid] * kval;
        __syncthreads();
        for (int stride = D >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) red[tid] += red[tid + stride];
            __syncthreads();
        }
        // red[0] = logit = <qrot, krot>

        // ---- Online softmax update (thread 0) ------------------------------
        if (tid == 0) {
            float logit = red[0];
            float m_new = fmaxf(sh_m, logit);
            sh_a = expf(sh_m - m_new);
            sh_b = expf(logit - m_new);
            sh_l = sh_l * sh_a + sh_b;
            sh_m = m_new;
        }
        __syncthreads();

        vaccum[tid] *= sh_a;

        // ---- 3-bit V decode → accumulate -----------------------------------
        const uint8_t* v3_codes = page_base + layout.v3_codes_offset +
            polar_token_head_offset(token_in_block, head_idx, cfg.num_kv_heads,
                                    layout.v3_bytes_per_token_head);
        const half* v_scale_ptr = reinterpret_cast<const half*>(
            page_base + layout.v_scales_offset +
            polar_token_head_offset(token_in_block, head_idx, cfg.num_kv_heads,
                                    layout.scale_bytes_per_token_head));

        float vval = kV3Codebook[unpack_3bit_get(v3_codes, tid)] * h2f(*v_scale_ptr);

        vaccum[tid] += sh_b * vval;
        __syncthreads();
    }

    // ---- Normalise + inverse WHT + sign-unflip → output -------------------
    __shared__ float sh_inv_l;
    if (tid == 0) sh_inv_l = (sh_l > 0.0f) ? (1.0f / sh_l) : 1.0f;
    __syncthreads();
    vaccum[tid] *= sh_inv_l;

    __syncthreads();
    hadamard_inplace<MAX_D>(vaccum, D);

    int obase = (q_idx * cfg.num_kv_heads + head_idx) * D;
    output[obase + tid] = f2h(vaccum[tid] * inv_sqrt_d * sign_flip(tid));
}

} // namespace

// ---------------------------------------------------------------------------
// Launch functions
// ---------------------------------------------------------------------------

void launch_tq_polar_pack_kv(
    const half*              key,
    const half*              value,
    const int32_t*           slot_mapping,
    uint8_t*                 page_pool,
    const TQPolarPageLayout& layout,
    const TQConfig&          cfg,
    int                      num_tokens,
    cudaStream_t             stream)
{
    if (cfg.head_dim > 128)
        throw std::runtime_error("Polar pack: head_dim > 128 not supported");
    dim3 grid(num_tokens, cfg.num_kv_heads);
    int threads = cfg.head_dim;
    // smem: sk[128] + sv[128] + red[128]  (static __shared__ is separate)
    size_t shmem = sizeof(float) * 3 * 128;
    polar_pack_kv_kernel<128><<<grid, threads, shmem, stream>>>(
        key, value, slot_mapping, page_pool, layout, cfg, num_tokens);
}

void launch_tq_polar_dequant_kv(
    const uint8_t*           page_pool,
    const int32_t*           slot_mapping,
    half*                    out_key,
    half*                    out_value,
    const TQPolarPageLayout& layout,
    const TQConfig&          cfg,
    int                      num_tokens,
    cudaStream_t             stream)
{
    if (cfg.head_dim > 128)
        throw std::runtime_error("Polar dequant: head_dim > 128 not supported");
    dim3 grid(num_tokens, cfg.num_kv_heads);
    int threads = cfg.head_dim;
    size_t shmem = sizeof(float) * 128;
    polar_dequant_kv_kernel<128><<<grid, threads, shmem, stream>>>(
        page_pool, slot_mapping, out_key, out_value, layout, cfg, num_tokens);
}

void launch_tq_polar_fused_attention_output(
    const half*              query,
    const uint8_t*           page_pool,
    const int32_t*           slot_mapping,
    half*                    output,
    const TQPolarPageLayout& layout,
    const TQConfig&          cfg,
    int                      num_queries,
    int                      num_kv_tokens,
    cudaStream_t             stream)
{
    if (cfg.head_dim > 128)
        throw std::runtime_error("Polar fused attn: head_dim > 128 not supported");
    dim3 grid(num_queries, cfg.num_kv_heads);
    int threads = cfg.head_dim;
    // smem: qrot[128] + vaccum[128] + red[128]  (sh_* are static __shared__)
    size_t shmem = sizeof(float) * 3 * 128;
    polar_fused_attn_online_kernel<128><<<grid, threads, shmem, stream>>>(
        query, page_pool, slot_mapping, output,
        layout, cfg, num_queries, num_kv_tokens);
}
