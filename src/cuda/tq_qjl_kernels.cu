#include "tq_qjl.cuh"
#include "tq_shared_device.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdexcept>

namespace {

__device__ __forceinline__ float h2f(half x) { return __half2float(x); }
__device__ __forceinline__ half  f2h(float x) { return __float2half(x); }

template <typename T>
__device__ __forceinline__ T clamp_val(T x, T lo, T hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

// Deterministic per-dimension sign flip (same hash as turbo_prod).
__device__ __forceinline__ float sign_flip(int idx) {
    unsigned x = static_cast<unsigned>(idx) * 1103515245u + 12345u;
    return (x & 1u) ? 1.0f : -1.0f;
}

// 4-bit V codebook – Gaussian Lloyd-Max optimal, same as turbo_prod.
__device__ __constant__ float kV4Codebook[16] = {
    -2.7326f, -2.0690f, -1.6180f, -1.2562f,
    -0.9423f, -0.6568f, -0.3880f, -0.1284f,
     0.1284f,  0.3880f,  0.6568f,  0.9423f,
     1.2562f,  1.6180f,  2.0690f,  2.7326f
};

__device__ __forceinline__ int nearest_v4_idx(float x) {
    int best = 0;
    float best_dist = fabsf(x - kV4Codebook[0]);
    #pragma unroll
    for (int i = 1; i < 16; ++i) {
        float d = fabsf(x - kV4Codebook[i]);
        if (d < best_dist) { best_dist = d; best = i; }
    }
    return best;
}


// In-place normalised Walsh-Hadamard Transform (butterfly on shared memory).
// Requires all participating threads to be alive; leaves them synchronised.
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
__global__ void qjl_pack_kv_kernel(
    const half* __restrict__ key,
    const half* __restrict__ value,
    const int32_t* __restrict__ slot_mapping,
    uint8_t* __restrict__ page_pool,
    TQQJLPageLayout layout,
    TQConfig        cfg,
    int             num_tokens)
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

    uint8_t* k1_codes = page_base + layout.k1_codes_offset +
        qjl_token_head_offset(token_in_block, head_idx, cfg.num_kv_heads,
                              layout.k1_bytes_per_token_head);
    half* kscale = reinterpret_cast<half*>(
        page_base + layout.k_scales_offset +
        qjl_token_head_offset(token_in_block, head_idx, cfg.num_kv_heads,
                              layout.scale_bytes_per_token_head));

    uint8_t* v4_codes = page_base + layout.v4_codes_offset +
        qjl_token_head_offset(token_in_block, head_idx, cfg.num_kv_heads,
                              layout.v4_bytes_per_token_head);
    half* vscale = reinterpret_cast<half*>(
        page_base + layout.v_scales_offset +
        qjl_token_head_offset(token_in_block, head_idx, cfg.num_kv_heads,
                              layout.scale_bytes_per_token_head));

    extern __shared__ float smem[];
    float* sk  = smem;           // [MAX_D]
    float* sv  = smem + MAX_D;   // [MAX_D]
    float* red = smem + 2*MAX_D; // [MAX_D]

    __shared__ int kbit_s[256]; // sign bits  (0 or 1)
    __shared__ int vidx_s[256]; // 4-bit V indices

    int base = (token_idx * cfg.num_kv_heads + head_idx) * D;

    // Zero code bytes before byte-owner write
    if (tid < layout.k1_bytes_per_token_head) k1_codes[tid] = 0;
    if (tid < layout.v4_bytes_per_token_head)  v4_codes[tid] = 0;
    __syncthreads();

    // Step 1 – sign-flip + WHT + normalise
    sk[tid] = h2f(key  [base + tid]) * sign_flip(tid);
    sv[tid] = h2f(value[base + tid]) * sign_flip(tid);
    __syncthreads();

    hadamard_inplace<MAX_D>(sk, D);
    hadamard_inplace<MAX_D>(sv, D);

    const float inv_sqrt_d = rsqrtf((float)D);
    sk[tid] *= inv_sqrt_d;
    sv[tid] *= inv_sqrt_d;

    // Step 2 – K RMS scale
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

    // Step 3 – V RMS scale
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

    // Step 4 – quantise
    kbit_s[tid] = (sk[tid] / krms >= 0.0f) ? 1 : 0;                  // 1-bit K
    vidx_s[tid] = nearest_v4_idx(clamp_val(sv[tid] / vrms, -2.75f, 2.75f)); // 4-bit V
    __syncthreads();

    // Step 5 – pack K bits (byte-owner: each thread writes one byte)
    if (tid < layout.k1_bytes_per_token_head)
        k1_codes[tid] = pack_1bit_byte_from_bits(kbit_s, tid, D);

    // Pack V nibbles: thread tid writes one byte covering dims 2*tid and 2*tid+1
    if (tid < (D >> 1)) {
        int i0 = 2 * tid, i1 = i0 + 1;
        v4_codes[tid] = (uint8_t)(((uint8_t)vidx_s[i1] << 4) | (uint8_t)vidx_s[i0]);
    }
}

// ---------------------------------------------------------------------------
// Dequant kernel
//
// grid = (num_tokens, num_kv_heads),  threads = head_dim
// Shared memory (extern): sk[MAX_D] | sv[MAX_D]
// ---------------------------------------------------------------------------
template<int MAX_D>
__global__ void qjl_dequant_kv_kernel(
    const uint8_t* __restrict__ page_pool,
    const int32_t* __restrict__ slot_mapping,
    half* __restrict__ out_key,
    half* __restrict__ out_value,
    TQQJLPageLayout layout,
    TQConfig        cfg,
    int             num_tokens)
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

    const uint8_t* k1_codes = page_base + layout.k1_codes_offset +
        qjl_token_head_offset(token_in_block, head_idx, cfg.num_kv_heads,
                              layout.k1_bytes_per_token_head);
    const half* kscale_ptr = reinterpret_cast<const half*>(
        page_base + layout.k_scales_offset +
        qjl_token_head_offset(token_in_block, head_idx, cfg.num_kv_heads,
                              layout.scale_bytes_per_token_head));

    const uint8_t* v4_codes = page_base + layout.v4_codes_offset +
        qjl_token_head_offset(token_in_block, head_idx, cfg.num_kv_heads,
                              layout.v4_bytes_per_token_head);
    const half* vscale_ptr = reinterpret_cast<const half*>(
        page_base + layout.v_scales_offset +
        qjl_token_head_offset(token_in_block, head_idx, cfg.num_kv_heads,
                              layout.scale_bytes_per_token_head));

    extern __shared__ float smem[];
    float* sk = smem;          // [MAX_D] rotated K reconstruction
    float* sv = smem + MAX_D;  // [MAX_D] rotated V reconstruction

    // Reconstruct in rotated domain
    float ks = h2f(*kscale_ptr);
    float vs = h2f(*vscale_ptr);

    sk[tid] = (unpack_1bit_get(k1_codes, tid) ? +1.0f : -1.0f) * ks;
    sv[tid] = kV4Codebook[unpack_4bit_get(v4_codes, tid)] * vs;
    __syncthreads();

    // Inverse rotation: WHT + 1/sqrt(D) + sign-unflip
    hadamard_inplace<MAX_D>(sk, D);
    hadamard_inplace<MAX_D>(sv, D);

    const float inv_sqrt_d = rsqrtf((float)D);
    int base = (token_idx * cfg.num_kv_heads + head_idx) * D;
    out_key  [base + tid] = f2h(sk[tid] * inv_sqrt_d * sign_flip(tid));
    out_value[base + tid] = f2h(sv[tid] * inv_sqrt_d * sign_flip(tid));
}

// ---------------------------------------------------------------------------
// Fused attention output kernel — online softmax (no logit_scratch)
//
// grid = (num_queries, num_kv_heads),  threads = head_dim
// Shared memory (extern): qrot[MAX_D] | vaccum[MAX_D] | red[MAX_D]
//
// Single pass over KV tokens (FlashAttention-style online softmax).
// Logit convention: <q, k>  (no 1/sqrt(D)), matching turbo_prod fused kernel.
// K is reconstructed as (sign_bit ? +1 : -1) * kscale.
// ---------------------------------------------------------------------------
template<int MAX_D>
__global__ void qjl_fused_attn_online_kernel(
    const half* __restrict__ query,
    const uint8_t* __restrict__ page_pool,
    const int32_t* __restrict__ slot_mapping,
    half* __restrict__ output,
    TQQJLPageLayout layout,
    TQConfig        cfg,
    int             num_queries,
    int             num_kv_tokens)
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
    float* qrot   = smem;           // [MAX_D]
    float* vaccum = smem + MAX_D;   // [MAX_D]
    float* red    = smem + 2*MAX_D; // [MAX_D]

    // Online softmax scalars — thread 0 writes, all threads read after sync
    __shared__ float sh_m;
    __shared__ float sh_l;
    __shared__ float sh_a;   // rescale: exp(m_old - m_new)
    __shared__ float sh_b;   // new-token weight: exp(logit - m_new)

    // Rotate query: sign_flip + WHT + 1/sqrt(D)
    int qbase = (q_idx * cfg.num_kv_heads + head_idx) * D;
    qrot[tid] = h2f(query[qbase + tid]) * sign_flip(tid);
    __syncthreads();

    hadamard_inplace<MAX_D>(qrot, D);
    qrot[tid] *= inv_sqrt_d;
    __syncthreads();

    vaccum[tid] = 0.0f;
    if (tid == 0) { sh_m = -1e30f; sh_l = 0.0f; }
    __syncthreads();

    for (int t = 0; t < num_kv_tokens; ++t) {
        int slot           = slot_mapping[t];
        int physical_block = slot / cfg.block_size;
        int token_in_block = slot % cfg.block_size;

        const uint8_t* page_base = page_pool +
            (size_t)physical_block * layout.page_size_bytes;

        // ---- 1-bit K decode → dot product ----------------------------------
        const uint8_t* k1_codes = page_base + layout.k1_codes_offset +
            qjl_token_head_offset(token_in_block, head_idx, cfg.num_kv_heads,
                                  layout.k1_bytes_per_token_head);
        const half* kscale_ptr = reinterpret_cast<const half*>(
            page_base + layout.k_scales_offset +
            qjl_token_head_offset(token_in_block, head_idx, cfg.num_kv_heads,
                                  layout.scale_bytes_per_token_head));

        float ks   = h2f(*kscale_ptr);
        float kval = (unpack_1bit_get(k1_codes, tid) ? +1.0f : -1.0f) * ks;

        red[tid] = qrot[tid] * kval;
        __syncthreads();
        for (int stride = D >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) red[tid] += red[tid + stride];
            __syncthreads();
        }
        // red[0] = logit

        // ---- Online softmax update (thread 0) ------------------------------
        if (tid == 0) {
            float logit = red[0];
            float m_new = fmaxf(sh_m, logit);
            sh_a = expf(sh_m - m_new);
            sh_b = expf(logit - m_new);
            sh_l = sh_l * sh_a + sh_b;
            sh_m = m_new;
        }
        __syncthreads();  // broadcast sh_a, sh_b

        vaccum[tid] *= sh_a;

        // ---- 4-bit V decode → accumulate -----------------------------------
        const uint8_t* v4_codes = page_base + layout.v4_codes_offset +
            qjl_token_head_offset(token_in_block, head_idx, cfg.num_kv_heads,
                                  layout.v4_bytes_per_token_head);
        const half* vscale_ptr = reinterpret_cast<const half*>(
            page_base + layout.v_scales_offset +
            qjl_token_head_offset(token_in_block, head_idx, cfg.num_kv_heads,
                                  layout.scale_bytes_per_token_head));

        float vs   = h2f(*vscale_ptr);
        float vval = kV4Codebook[unpack_4bit_get(v4_codes, tid)] * vs;
        vaccum[tid] += sh_b * vval;
        __syncthreads();  // barrier before next iteration's reduction
    }

    // Normalise
    __shared__ float sh_inv_l;
    if (tid == 0) sh_inv_l = (sh_l > 0.0f) ? (1.0f / sh_l) : 1.0f;
    __syncthreads();
    vaccum[tid] *= sh_inv_l;

    // Inverse rotation → FP16 output
    __syncthreads();
    hadamard_inplace<MAX_D>(vaccum, D);

    int obase = (q_idx * cfg.num_kv_heads + head_idx) * D;
    output[obase + tid] = f2h(vaccum[tid] * inv_sqrt_d * sign_flip(tid));
}

} // namespace

// ---------------------------------------------------------------------------
// Host launch functions
// ---------------------------------------------------------------------------

void launch_tq_qjl_pack_kv(
    const half*            key,
    const half*            value,
    const int32_t*         slot_mapping,
    uint8_t*               page_pool,
    const TQQJLPageLayout& layout,
    const TQConfig&        cfg,
    int                    num_tokens,
    cudaStream_t           stream)
{
    if (cfg.head_dim > 128)
        throw std::runtime_error("QJL: head_dim > 128 is not supported");
    dim3 grid(num_tokens, cfg.num_kv_heads);
    int    threads = cfg.head_dim;
    size_t shmem   = sizeof(float) * (3 * 128);
    qjl_pack_kv_kernel<128><<<grid, threads, shmem, stream>>>(
        key, value, slot_mapping, page_pool, layout, cfg, num_tokens);
}

void launch_tq_qjl_dequant_kv(
    const uint8_t*         page_pool,
    const int32_t*         slot_mapping,
    half*                  out_key,
    half*                  out_value,
    const TQQJLPageLayout& layout,
    const TQConfig&        cfg,
    int                    num_tokens,
    cudaStream_t           stream)
{
    if (cfg.head_dim > 128)
        throw std::runtime_error("QJL: head_dim > 128 is not supported");
    dim3 grid(num_tokens, cfg.num_kv_heads);
    int    threads = cfg.head_dim;
    size_t shmem   = sizeof(float) * (2 * 128);
    qjl_dequant_kv_kernel<128><<<grid, threads, shmem, stream>>>(
        page_pool, slot_mapping, out_key, out_value, layout, cfg, num_tokens);
}

void launch_tq_qjl_fused_attention_output(
    const half*            query,
    const uint8_t*         page_pool,
    const int32_t*         slot_mapping,
    half*                  output,
    const TQQJLPageLayout& layout,
    const TQConfig&        cfg,
    int                    num_queries,
    int                    num_kv_tokens,
    cudaStream_t           stream)
{
    if (cfg.head_dim > 128)
        throw std::runtime_error("QJL: head_dim > 128 is not supported");
    dim3 grid(num_queries, cfg.num_kv_heads);
    int    threads = cfg.head_dim;
    size_t shmem   = sizeof(float) * (3 * 128);
    qjl_fused_attn_online_kernel<128><<<grid, threads, shmem, stream>>>(
        query, page_pool, slot_mapping, output,
        layout, cfg, num_queries, num_kv_tokens);
}
