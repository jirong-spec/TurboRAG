#include "tq_turbo_prod_kernels.cuh"
#include "tq_shared_device.cuh"
#include "tq_cuda_check.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <stdexcept>

namespace {

__device__ __forceinline__ float h2f(half x) { return __half2float(x); }
__device__ __forceinline__ half f2h(float x) { return __float2half(x); }

template <typename T>
__device__ __forceinline__ T clamp_val(T x, T lo, T hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

__device__ __forceinline__ float sign_flip(int idx, int head_idx) {
    unsigned x = (static_cast<unsigned>(head_idx) * 2654435761u)
               ^ (static_cast<unsigned>(idx)      * 1103515245u + 12345u);
    return (x & 1u) ? 1.0f : -1.0f;
}

// K: 3-bit base codebook — Lloyd-Max optimal for N(0,1)
__device__ __constant__ float kK3Codebook[8] = {
    -2.1513f, -1.34326f, -0.755526f, -0.244919f,
     0.244919f, 0.755526f, 1.34326f, 2.1513f
};

// V: 4-bit codebook
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
        if (d < best_dist) { best_dist = d; best = i; }
    }
    return best;
}

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

// ---------------------------------------------------------------------------
// K interleaved nibble helpers
//
// Each 4-bit nibble encodes one head-dim element:
//   bits [2:0] = 3-bit Lloyd-Max code   (0–7)
//   bit  [3]   = 1-bit QJL residual sign
//
// Packing two elements per byte: elem[2*i] in lower nibble, elem[2*i+1] in upper.
// ---------------------------------------------------------------------------

__device__ __forceinline__ uint8_t make_k4_nibble(int code3, int resbit) {
    return (uint8_t)(((resbit & 1) << 3) | (code3 & 7));
}

// Assemble one output byte from the shared-mem code/resbit arrays.
// byte_idx owns elements at positions 2*byte_idx and 2*byte_idx+1.
__device__ __forceinline__ uint8_t pack_k4_byte(const int* codes, const int* rbits,
                                                 int byte_idx, int D) {
    int i0 = 2 * byte_idx;
    int i1 = i0 + 1;
    uint8_t out = 0;
    if (i0 < D) out  = make_k4_nibble(codes[i0], rbits[i0]);
    if (i1 < D) out |= (uint8_t)(make_k4_nibble(codes[i1], rbits[i1]) << 4);
    return out;
}

// ---------------------------------------------------------------------------

template<int MAX_D>
__device__ void hadamard_inplace(float* x, int D) {
    for (int len = 1; len < D; len <<= 1) {
        int tid = threadIdx.x;
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
// ---------------------------------------------------------------------------
template<int MAX_D>
__global__ void turbo_prod_pack_kv_kernel(
    const half* __restrict__ key,
    const half* __restrict__ value,
    const int32_t* __restrict__ slot_mapping,
    uint8_t* __restrict__ page_pool,
    TQTurboProdPageLayout layout,
    TQConfig cfg,
    int num_tokens,
    float* __restrict__ debug_k_rot,
    float* __restrict__ debug_v_rot,
    float* __restrict__ debug_kn,
    float* __restrict__ debug_vn,
    int* __restrict__ debug_kidx,
    int* __restrict__ debug_vidx) {

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

    // K interleaved nibble stream
    uint8_t* k4_codes = page_base + layout.k4_codes_offset +
        turbo_prod_token_head_offset(token_in_block, head_idx,
                                     cfg.num_kv_heads, layout.k4_bytes_per_token_head);

    half* kres_scale = reinterpret_cast<half*>(
        page_base + layout.k_residual_scales_offset +
        turbo_prod_token_head_offset(token_in_block, head_idx,
                                     cfg.num_kv_heads, layout.scale_bytes_per_token_head));
    half* kscale = reinterpret_cast<half*>(
        page_base + layout.k_scales_offset +
        turbo_prod_token_head_offset(token_in_block, head_idx,
                                     cfg.num_kv_heads, layout.scale_bytes_per_token_head));

    uint8_t* v4_codes = page_base + layout.v4_codes_offset +
        turbo_prod_token_head_offset(token_in_block, head_idx,
                                     cfg.num_kv_heads, layout.v4_bytes_per_token_head);
    half* vscale = reinterpret_cast<half*>(
        page_base + layout.v_scales_offset +
        turbo_prod_token_head_offset(token_in_block, head_idx,
                                     cfg.num_kv_heads, layout.scale_bytes_per_token_head));

    extern __shared__ float smem[];
    float* sk  = smem;
    float* sv  = smem + MAX_D;
    float* red = smem + 2 * MAX_D;

    __shared__ int kidx_s[256];
    __shared__ int krbit_s[256];
    __shared__ int vidx_s[256];

    // Zero-initialise output buffers (pack_k4_byte does full assignment, but
    // keep the pattern consistent with v4 for clarity)
    if (tid < layout.k4_bytes_per_token_head) k4_codes[tid] = 0;
    if (tid < layout.v4_bytes_per_token_head)  v4_codes[tid] = 0;
    __syncthreads();

    // Load, sign-flip, WHT
    int base = (token_idx * cfg.num_kv_heads + head_idx) * D;
    sk[tid] = h2f(key[base + tid])   * sign_flip(tid, head_idx);
    sv[tid] = h2f(value[base + tid]) * sign_flip(tid, head_idx);
    __syncthreads();

    hadamard_inplace<MAX_D>(sk, D);
    hadamard_inplace<MAX_D>(sv, D);

    float inv_sqrt_d = rsqrtf((float)D);
    sk[tid] *= inv_sqrt_d;
    sv[tid] *= inv_sqrt_d;

    // K RMS
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

    // V RMS
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

    int kidx  = nearest_k3_idx(kn);
    float e   = kn - kK3Codebook[kidx];   // QJL residual in normalised space
    int krbit = (e >= 0.0f) ? 1 : 0;

    int vidx = nearest_v4_idx(vn);

    // QJL residual RMS (re-use red[])
    red[tid] = e * e;
    __syncthreads();
    for (int stride = D >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) red[tid] += red[tid + stride];
        __syncthreads();
    }
    float rms_e = sqrtf(red[0] / (float)D);
    if (rms_e < 1e-8f) rms_e = 1e-8f;
    if (tid == 0) *kres_scale = f2h(rms_e);

    kidx_s[tid]  = kidx;
    krbit_s[tid] = krbit;
    vidx_s[tid]  = vidx;

    if (debug_kn)   debug_kn[base + tid]   = kn;
    if (debug_vn)   debug_vn[base + tid]   = vn;
    if (debug_kidx) debug_kidx[base + tid] = kidx;
    if (debug_vidx) debug_vidx[base + tid] = vidx;

    __syncthreads();

    // K: interleaved nibble pack — each thread owns one byte (two elements)
    if (tid < layout.k4_bytes_per_token_head) {
        k4_codes[tid] = pack_k4_byte(kidx_s, krbit_s, tid, D);
    }

    // V: unchanged nibble pack
    if (tid < (D >> 1)) {
        int i0 = 2 * tid, i1 = i0 + 1;
        v4_codes[tid] = (uint8_t)(((uint8_t)vidx_s[i1] << 4) | (uint8_t)vidx_s[i0]);
    }
}

// ---------------------------------------------------------------------------
// Dequant kernel
// ---------------------------------------------------------------------------
template<int MAX_D>
__global__ void turbo_prod_dequant_kv_kernel(
    const uint8_t* __restrict__ page_pool,
    const int32_t* __restrict__ slot_mapping,
    half* __restrict__ out_key,
    half* __restrict__ out_value,
    TQTurboProdPageLayout layout,
    TQConfig cfg,
    int num_tokens) {

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

    const uint8_t* k4_codes = page_base + layout.k4_codes_offset +
        turbo_prod_token_head_offset(token_in_block, head_idx,
                                     cfg.num_kv_heads, layout.k4_bytes_per_token_head);
    const half* kres_scale = reinterpret_cast<const half*>(
        page_base + layout.k_residual_scales_offset +
        turbo_prod_token_head_offset(token_in_block, head_idx,
                                     cfg.num_kv_heads, layout.scale_bytes_per_token_head));
    const half* kscale = reinterpret_cast<const half*>(
        page_base + layout.k_scales_offset +
        turbo_prod_token_head_offset(token_in_block, head_idx,
                                     cfg.num_kv_heads, layout.scale_bytes_per_token_head));

    const uint8_t* v4_codes = page_base + layout.v4_codes_offset +
        turbo_prod_token_head_offset(token_in_block, head_idx,
                                     cfg.num_kv_heads, layout.v4_bytes_per_token_head);
    const half* vscale = reinterpret_cast<const half*>(
        page_base + layout.v_scales_offset +
        turbo_prod_token_head_offset(token_in_block, head_idx,
                                     cfg.num_kv_heads, layout.scale_bytes_per_token_head));

    extern __shared__ float smem[];
    float* sk = smem;
    float* sv = smem + MAX_D;

    float ks = h2f(*kscale);
    float re = h2f(*kres_scale);
    float vs = h2f(*vscale);

    // Decode K interleaved nibble: bits[2:0]=code3, bit[3]=resbit
    uint8_t nibble = unpack_4bit_get(k4_codes, tid);
    uint8_t kidx   = nibble & 0x7u;
    float   sign_e = ((nibble >> 3) & 0x1u) ? 1.0f : -1.0f;

    uint8_t vidx = unpack_4bit_get(v4_codes, tid);

    float kval = kK3Codebook[kidx] + re * sign_e;
    float vval = kV4Codebook[vidx];

    sk[tid] = kval * ks;
    sv[tid] = vval * vs;
    __syncthreads();

    hadamard_inplace<MAX_D>(sk, D);
    hadamard_inplace<MAX_D>(sv, D);

    float inv_sqrt_d = rsqrtf((float)D);
    int base = (token_idx * cfg.num_kv_heads + head_idx) * D;

    out_key[base + tid]   = f2h(sk[tid] * inv_sqrt_d * sign_flip(tid, head_idx));
    out_value[base + tid] = f2h(sv[tid] * inv_sqrt_d * sign_flip(tid, head_idx));
}

// ---------------------------------------------------------------------------
// Fused attention logits kernel
// ---------------------------------------------------------------------------
template<int MAX_D>
__global__ void turbo_prod_fused_attention_logits_kernel(
    const half* __restrict__ query,
    const uint8_t* __restrict__ page_pool,
    const int32_t* __restrict__ slot_mapping,
    float* __restrict__ logits,
    TQTurboProdPageLayout layout,
    TQConfig cfg,
    int num_queries,
    int num_kv_tokens) {

    int q_idx    = blockIdx.x;
    int head_idx = blockIdx.y;
    int tid      = threadIdx.x;

    if (q_idx >= num_queries || head_idx >= cfg.num_kv_heads) return;
    if (cfg.head_dim > MAX_D) return;

    int D = cfg.head_dim;
    if (tid >= D) return;

    extern __shared__ float smem[];
    float* qrot = smem;
    float* kred = smem + MAX_D;   // reserved, not used — keeps shmem layout stable
    float* red  = smem + 2 * MAX_D;

    int qbase = (q_idx * cfg.num_kv_heads + head_idx) * D;
    qrot[tid] = h2f(query[qbase + tid]) * sign_flip(tid, head_idx);
    __syncthreads();

    hadamard_inplace<MAX_D>(qrot, D);
    float inv_sqrt_d = rsqrtf((float)D);
    qrot[tid] *= inv_sqrt_d;
    __syncthreads();

    for (int kv_idx = 0; kv_idx < num_kv_tokens; ++kv_idx) {
        int slot           = slot_mapping[kv_idx];
        int physical_block = slot / cfg.block_size;
        int token_in_block = slot % cfg.block_size;

        const uint8_t* page_base = page_pool + (size_t)physical_block * layout.page_size_bytes;

        const uint8_t* k4_codes = page_base + layout.k4_codes_offset +
            turbo_prod_token_head_offset(token_in_block, head_idx,
                                         cfg.num_kv_heads, layout.k4_bytes_per_token_head);
        const half* kres_scale_ptr = reinterpret_cast<const half*>(
            page_base + layout.k_residual_scales_offset +
            turbo_prod_token_head_offset(token_in_block, head_idx,
                                         cfg.num_kv_heads, layout.scale_bytes_per_token_head));
        const half* kscale = reinterpret_cast<const half*>(
            page_base + layout.k_scales_offset +
            turbo_prod_token_head_offset(token_in_block, head_idx,
                                         cfg.num_kv_heads, layout.scale_bytes_per_token_head));

        float ks     = h2f(*kscale);
        float re     = h2f(*kres_scale_ptr);
        float res_ks = ks * re;

        // Decode K nibble once; split into base code and residual sign
        uint8_t nibble = unpack_4bit_get(k4_codes, tid);
        float   base_d = kK3Codebook[nibble & 0x7u];
        float   sign_e = ((nibble >> 3) & 0x1u) ? 1.0f : -1.0f;

        // Stage 1: base inner product from 3-bit code
        red[tid] = qrot[tid] * base_d * ks;
        __syncthreads();
        for (int stride = D >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) red[tid] += red[tid + stride];
            __syncthreads();
        }
        float logit_base = red[0];
        __syncthreads();

        // Stage 2: QJL residual correction
        red[tid] = qrot[tid] * sign_e;
        __syncthreads();
        for (int stride = D >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) red[tid] += red[tid + stride];
            __syncthreads();
        }
        float logit = (logit_base + red[0] * res_ks) * inv_sqrt_d;

        if (tid == 0) {
            logits[(q_idx * cfg.num_kv_heads + head_idx) * num_kv_tokens + kv_idx] = logit;
        }
        __syncthreads();
    }
    (void)kred;  // suppress unused-variable warning
}

// ---------------------------------------------------------------------------
// Fused attention output kernel — online softmax (FlashAttention-style)
// ---------------------------------------------------------------------------
template<int MAX_D>
__global__ void turbo_prod_fused_attn_online_kernel(
    const half* __restrict__ query,
    const uint8_t* __restrict__ page_pool,
    const int32_t* __restrict__ slot_mapping,
    half* __restrict__ output,
    TQTurboProdPageLayout layout,
    TQConfig cfg,
    int num_queries,
    int num_kv_tokens) {

    int q_idx    = blockIdx.x;
    int head_idx = blockIdx.y;
    int tid      = threadIdx.x;

    if (q_idx >= num_queries || head_idx >= cfg.num_kv_heads) return;
    if (cfg.head_dim > MAX_D) return;

    int D = cfg.head_dim;
    if (tid >= D) return;

    const float inv_sqrt_d = rsqrtf((float)D);

    extern __shared__ float smem[];
    float* qrot   = smem;
    float* vaccum = smem + MAX_D;
    float* red    = smem + 2 * MAX_D;

    __shared__ float sh_m;    // running max logit
    __shared__ float sh_l;    // running softmax denominator
    __shared__ float sh_a;    // rescale factor: exp(m_old - m_new)
    __shared__ float sh_b;    // new-token weight: exp(logit - m_new)

    int qbase = (q_idx * cfg.num_kv_heads + head_idx) * D;
    qrot[tid] = h2f(query[qbase + tid]) * sign_flip(tid, head_idx);
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

        // ---- K decode — single nibble load, two-stage logit ----------------
        const uint8_t* k4_codes = page_base + layout.k4_codes_offset +
            turbo_prod_token_head_offset(token_in_block, head_idx,
                                         cfg.num_kv_heads, layout.k4_bytes_per_token_head);
        const half* kres_scale_ptr = reinterpret_cast<const half*>(
            page_base + layout.k_residual_scales_offset +
            turbo_prod_token_head_offset(token_in_block, head_idx,
                                         cfg.num_kv_heads, layout.scale_bytes_per_token_head));
        const half* kscale = reinterpret_cast<const half*>(
            page_base + layout.k_scales_offset +
            turbo_prod_token_head_offset(token_in_block, head_idx,
                                         cfg.num_kv_heads, layout.scale_bytes_per_token_head));

        float ks     = h2f(*kscale);
        float re     = h2f(*kres_scale_ptr);
        float res_ks = ks * re;

        // One nibble load → code3 + resbit
        uint8_t nibble = unpack_4bit_get(k4_codes, tid);
        float   base_d = kK3Codebook[nibble & 0x7u];
        float   sign_e = ((nibble >> 3) & 0x1u) ? 1.0f : -1.0f;

        // Stage 1: base dot product
        red[tid] = qrot[tid] * base_d * ks;
        __syncthreads();
        for (int stride = D >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) red[tid] += red[tid + stride];
            __syncthreads();
        }
        float logit_base = red[0];
        __syncthreads();

        // Stage 2: QJL residual correction
        red[tid] = qrot[tid] * sign_e;
        __syncthreads();
        for (int stride = D >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) red[tid] += red[tid + stride];
            __syncthreads();
        }
        float logit = (logit_base + red[0] * res_ks) * inv_sqrt_d;

        // ---- Online softmax update (thread 0) --------------------------------
        if (tid == 0) {
            float m_new = fmaxf(sh_m, logit);
            sh_a = expf(sh_m - m_new);
            sh_b = expf(logit - m_new);
            sh_l = sh_l * sh_a + sh_b;
            sh_m = m_new;
        }
        __syncthreads();

        vaccum[tid] *= sh_a;

        // ---- V decode → accumulate ------------------------------------------
        const uint8_t* v4_codes = page_base + layout.v4_codes_offset +
            turbo_prod_token_head_offset(token_in_block, head_idx,
                                         cfg.num_kv_heads, layout.v4_bytes_per_token_head);
        const half* vscale_ptr = reinterpret_cast<const half*>(
            page_base + layout.v_scales_offset +
            turbo_prod_token_head_offset(token_in_block, head_idx,
                                         cfg.num_kv_heads, layout.scale_bytes_per_token_head));

        float vs   = h2f(*vscale_ptr);
        float vval = kV4Codebook[unpack_4bit_get(v4_codes, tid)] * vs;
        vaccum[tid] += sh_b * vval;
        __syncthreads();
    }

    // Normalise and inverse-WHT
    __shared__ float sh_inv_l;
    if (tid == 0) sh_inv_l = (sh_l > 0.0f) ? (1.0f / sh_l) : 1.0f;
    __syncthreads();
    vaccum[tid] *= sh_inv_l;

    __syncthreads();
    hadamard_inplace<MAX_D>(vaccum, D);

    int obase = (q_idx * cfg.num_kv_heads + head_idx) * D;
    output[obase + tid] = f2h(vaccum[tid] * inv_sqrt_d * sign_flip(tid, head_idx));
}

} // namespace

// ---------------------------------------------------------------------------
// Launch wrappers
// ---------------------------------------------------------------------------

void launch_tq_turbo_prod_pack_kv(
    const half* key,
    const half* value,
    const int32_t* slot_mapping,
    uint8_t* page_pool,
    const TQTurboProdPageLayout& layout,
    const TQConfig& cfg,
    int num_tokens,
    cudaStream_t stream,
    float* debug_k_rot,
    float* debug_v_rot,
    float* debug_kn,
    float* debug_vn,
    int* debug_kidx,
    int* debug_vidx) {

    if (cfg.head_dim > 128)
        throw std::runtime_error("turbo_prod: head_dim > 128 is not supported");
    dim3 grid(num_tokens, cfg.num_kv_heads);
    int threads = cfg.head_dim;
    size_t shmem = sizeof(float) * (3 * 128);
    turbo_prod_pack_kv_kernel<128><<<grid, threads, shmem, stream>>>(
        key, value, slot_mapping, page_pool, layout, cfg, num_tokens,
        debug_k_rot, debug_v_rot, debug_kn, debug_vn, debug_kidx, debug_vidx);
    TQ_CHECK_LAUNCH("turbo_prod_pack_kv_kernel");
    TQ_CHECK_ASYNC(stream);
}

void launch_tq_turbo_prod_dequant_kv(
    const uint8_t* page_pool,
    const int32_t* slot_mapping,
    half* out_key,
    half* out_value,
    const TQTurboProdPageLayout& layout,
    const TQConfig& cfg,
    int num_tokens,
    cudaStream_t stream) {

    if (cfg.head_dim > 128)
        throw std::runtime_error("turbo_prod: head_dim > 128 is not supported");
    dim3 grid(num_tokens, cfg.num_kv_heads);
    int threads = cfg.head_dim;
    size_t shmem = sizeof(float) * (2 * 128);
    turbo_prod_dequant_kv_kernel<128><<<grid, threads, shmem, stream>>>(
        page_pool, slot_mapping, out_key, out_value, layout, cfg, num_tokens);
    TQ_CHECK_LAUNCH("turbo_prod_dequant_kv_kernel");
    TQ_CHECK_ASYNC(stream);
}

void launch_tq_turbo_prod_fused_attention_logits(
    const half* query,
    const uint8_t* page_pool,
    const int32_t* slot_mapping,
    float* logits,
    const TQTurboProdPageLayout& layout,
    const TQConfig& cfg,
    int num_queries,
    int num_kv_tokens,
    cudaStream_t stream) {

    if (cfg.head_dim > 128)
        throw std::runtime_error("turbo_prod: head_dim > 128 is not supported");
    dim3 grid(num_queries, cfg.num_kv_heads);
    int threads = cfg.head_dim;
    size_t shmem = sizeof(float) * (3 * 128);
    turbo_prod_fused_attention_logits_kernel<128><<<grid, threads, shmem, stream>>>(
        query, page_pool, slot_mapping, logits, layout, cfg, num_queries, num_kv_tokens);
    TQ_CHECK_LAUNCH("turbo_prod_fused_attention_logits_kernel");
    TQ_CHECK_ASYNC(stream);
}

void launch_tq_turbo_prod_fused_attention_output(
    const half* query,
    const uint8_t* page_pool,
    const int32_t* slot_mapping,
    half* output,
    const TQTurboProdPageLayout& layout,
    const TQConfig& cfg,
    int num_queries,
    int num_kv_tokens,
    cudaStream_t stream) {

    if (cfg.head_dim > 128)
        throw std::runtime_error("turbo_prod: head_dim > 128 is not supported");
    dim3 grid(num_queries, cfg.num_kv_heads);
    int threads = cfg.head_dim;
    size_t shmem = sizeof(float) * (3 * 128);
    turbo_prod_fused_attn_online_kernel<128><<<grid, threads, shmem, stream>>>(
        query, page_pool, slot_mapping, output,
        layout, cfg, num_queries, num_kv_tokens);
    TQ_CHECK_LAUNCH("turbo_prod_fused_attn_online_kernel");
    TQ_CHECK_ASYNC(stream);
}
