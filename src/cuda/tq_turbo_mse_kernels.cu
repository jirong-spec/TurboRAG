#include "tq_turbo_mse_kernels.cuh"
#include "tq_shared_device.cuh"
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

__device__ __constant__ float kTurboCodebook[16] = {
    -2.7326f, -2.0690f, -1.6180f, -1.2562f,
    -0.9423f, -0.6568f, -0.3880f, -0.1284f,
     0.1284f,  0.3880f,  0.6568f,  0.9423f,
     1.2562f,  1.6180f,  2.0690f,  2.7326f
};

__device__ __forceinline__ int nearest_codebook_idx(float x) {
    int best = 0;
    float best_dist = fabsf(x - kTurboCodebook[0]);
    #pragma unroll
    for (int i = 1; i < 16; ++i) {
        float d = fabsf(x - kTurboCodebook[i]);
        if (d < best_dist) {
            best_dist = d;
            best = i;
        }
    }
    return best;
}

// In-place normalised Walsh-Hadamard Transform (butterfly on shared memory).
// Requires all D threads alive; leaves them synchronised at exit.
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

template<int MAX_D>
__device__ void cooperative_forward_transform_and_quantize(
    const half* src,
    uint8_t* codes,
    half* scale_ptr,
    int D,
    float* smem_vec,
    float* smem_red,
    int head_idx) {

    int tid = threadIdx.x;
    int pair_count = D / 2;

    if (tid < pair_count) {
        int i0 = 2 * tid;
        int i1 = i0 + 1;
        smem_vec[i0] = h2f(src[i0]) * sign_flip(i0, head_idx);
        smem_vec[i1] = h2f(src[i1]) * sign_flip(i1, head_idx);
    }
    __syncthreads();

    for (int len = 1; len < D; len <<= 1) {
        int butterfly = D >> 1;
        if (tid < butterfly) {
            int group = tid / len;
            int offset = tid % len;
            int j0 = group * (len << 1) + offset;
            int j1 = j0 + len;
            float a = smem_vec[j0];
            float b = smem_vec[j1];
            smem_vec[j0] = a + b;
            smem_vec[j1] = a - b;
        }
        __syncthreads();
    }

    const float inv_sqrt_d = rsqrtf((float)D);

    if (tid < pair_count) {
        int i0 = 2 * tid;
        int i1 = i0 + 1;
        smem_vec[i0] *= inv_sqrt_d;
        smem_vec[i1] *= inv_sqrt_d;
        smem_red[tid] = smem_vec[i0] * smem_vec[i0] + smem_vec[i1] * smem_vec[i1];
    }
    __syncthreads();

    for (int stride = pair_count >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem_red[tid] += smem_red[tid + stride];
        }
        __syncthreads();
    }

    float rms = sqrtf(smem_red[0] / (float)D);
    if (rms < 1e-8f) rms = 1e-8f;

    if (tid == 0) {
        *scale_ptr = f2h(rms);
    }
    __syncthreads();

    if (tid < pair_count) {
        int i0 = 2 * tid;
        int i1 = i0 + 1;
        float a = clamp_val(smem_vec[i0] / rms, -2.75f, 2.75f);
        float b = clamp_val(smem_vec[i1] / rms, -2.75f, 2.75f);
        int qa = nearest_codebook_idx(a);
        int qb = nearest_codebook_idx(b);
        codes[tid] = static_cast<uint8_t>((qb << 4) | qa);
    }
}

template<int MAX_D>
__device__ void cooperative_inverse_transform_and_store(
    const uint8_t* codes,
    const half* scale_ptr,
    half* dst,
    int D,
    float* smem,
    int head_idx) {

    int tid = threadIdx.x;
    int pair_count = D / 2;
    float rms = h2f(*scale_ptr);

    if (tid < pair_count) {
        uint8_t packed = codes[tid];
        int qa = packed & 0xF;
        int qb = (packed >> 4) & 0xF;
        smem[2 * tid]     = kTurboCodebook[qa] * rms;
        smem[2 * tid + 1] = kTurboCodebook[qb] * rms;
    }
    __syncthreads();

    for (int len = 1; len < D; len <<= 1) {
        int butterfly = D >> 1;
        if (tid < butterfly) {
            int group = tid / len;
            int offset = tid % len;
            int i0 = group * (len << 1) + offset;
            int i1 = i0 + len;
            float a = smem[i0];
            float b = smem[i1];
            smem[i0] = a + b;
            smem[i1] = a - b;
        }
        __syncthreads();
    }

    const float inv_sqrt_d = rsqrtf((float)D);
    if (tid < pair_count) {
        int i0 = 2 * tid;
        int i1 = i0 + 1;
        float v0 = smem[i0] * inv_sqrt_d * sign_flip(i0, head_idx);
        float v1 = smem[i1] * inv_sqrt_d * sign_flip(i1, head_idx);
        dst[i0] = f2h(v0);
        dst[i1] = f2h(v1);
    }
}

template<int MAX_D>
__global__ void turbo_mse_pack_kv_kernel(
    const half* __restrict__ key,
    const half* __restrict__ value,
    const int32_t* __restrict__ slot_mapping,
    uint8_t* __restrict__ page_pool,
    TQTurbomsePageLayout layout,
    TQConfig cfg,
    int num_tokens) {

    int token_idx = blockIdx.x;
    int head_idx = blockIdx.y;

    if (token_idx >= num_tokens || head_idx >= cfg.num_kv_heads) return;
    if (cfg.head_dim > MAX_D) return;

    int D = cfg.head_dim;
    int pair_count = D / 2;
    if (threadIdx.x >= pair_count) return;

    int slot = slot_mapping[token_idx];
    int physical_block = slot / cfg.block_size;
    int token_in_block = slot % cfg.block_size;

    uint8_t* page_base =
        page_pool + static_cast<size_t>(physical_block) * layout.page_size_bytes;

    uint8_t* k_codes = page_base + layout.k_codes_offset +
        turbo_token_head_code_offset(layout, token_in_block, head_idx, cfg.num_kv_heads);
    uint8_t* v_codes = page_base + layout.v_codes_offset +
        turbo_token_head_code_offset(layout, token_in_block, head_idx, cfg.num_kv_heads);

    half* k_scale = reinterpret_cast<half*>(
        page_base + layout.k_norms_offset +
        turbo_token_head_norm_offset(layout, token_in_block, head_idx, cfg.num_kv_heads));
    half* v_scale = reinterpret_cast<half*>(
        page_base + layout.v_norms_offset +
        turbo_token_head_norm_offset(layout, token_in_block, head_idx, cfg.num_kv_heads));

    int base = (token_idx * cfg.num_kv_heads + head_idx) * D;

    extern __shared__ float smem[];
    float* smem_k = smem;
    float* smem_v = smem + MAX_D;
    float* smem_red = smem + 2 * MAX_D;

    cooperative_forward_transform_and_quantize<MAX_D>(
        key + base, k_codes, k_scale, D, smem_k, smem_red, head_idx);
    __syncthreads();
    cooperative_forward_transform_and_quantize<MAX_D>(
        value + base, v_codes, v_scale, D, smem_v, smem_red, head_idx);
}

template<int MAX_D>
__global__ void turbo_mse_dequant_kv_kernel(
    const uint8_t* __restrict__ page_pool,
    const int32_t* __restrict__ slot_mapping,
    half* __restrict__ out_key,
    half* __restrict__ out_value,
    TQTurbomsePageLayout layout,
    TQConfig cfg,
    int num_tokens) {

    int token_idx = blockIdx.x;
    int head_idx = blockIdx.y;

    if (token_idx >= num_tokens || head_idx >= cfg.num_kv_heads) return;
    if (cfg.head_dim > MAX_D) return;

    int D = cfg.head_dim;
    int pair_count = D / 2;
    if (threadIdx.x >= pair_count) return;

    int slot = slot_mapping[token_idx];
    int physical_block = slot / cfg.block_size;
    int token_in_block = slot % cfg.block_size;

    const uint8_t* page_base =
        page_pool + static_cast<size_t>(physical_block) * layout.page_size_bytes;

    const uint8_t* k_codes = page_base + layout.k_codes_offset +
        turbo_token_head_code_offset(layout, token_in_block, head_idx, cfg.num_kv_heads);
    const uint8_t* v_codes = page_base + layout.v_codes_offset +
        turbo_token_head_code_offset(layout, token_in_block, head_idx, cfg.num_kv_heads);

    const half* k_scale = reinterpret_cast<const half*>(
        page_base + layout.k_norms_offset +
        turbo_token_head_norm_offset(layout, token_in_block, head_idx, cfg.num_kv_heads));
    const half* v_scale = reinterpret_cast<const half*>(
        page_base + layout.v_norms_offset +
        turbo_token_head_norm_offset(layout, token_in_block, head_idx, cfg.num_kv_heads));

    int base = (token_idx * cfg.num_kv_heads + head_idx) * D;

    extern __shared__ float smem[];
    float* smem_k = smem;
    float* smem_v = smem + MAX_D;

    cooperative_inverse_transform_and_store<MAX_D>(k_codes, k_scale, out_key + base, D, smem_k, head_idx);
    __syncthreads();
    cooperative_inverse_transform_and_store<MAX_D>(v_codes, v_scale, out_value + base, D, smem_v, head_idx);
}

// ---------------------------------------------------------------------------
// turbo_mse fused attention — online softmax, single KV fetch per token
//
// grid = (num_queries, num_kv_heads),  threads = head_dim
// Shared memory (extern): qrot[MAX_D] | vaccum[MAX_D] | red[MAX_D]
//
// turbo_mse uses the same sign_flip + WHT preprocessing as turbo_prod and QJL,
// with a single 4-bit Gaussian codebook (kTurboCodebook) for both K and V.
// The identity <qrot, krot> = <q, k> holds, so attention logits are computed
// entirely in the rotated domain without materialising FP16 KV in global mem.
//
// Logit convention: <q, k> / sqrt(D)  (standard scaled dot-product attention).
// ---------------------------------------------------------------------------
template<int MAX_D>
__global__ void turbo_mse_fused_attn_online_kernel(
    const half* __restrict__ query,
    const uint8_t* __restrict__ page_pool,
    const int32_t* __restrict__ slot_mapping,
    half* __restrict__ output,
    TQTurbomsePageLayout layout,
    TQConfig cfg,
    int num_queries,
    int num_kv_tokens)
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

    __shared__ float sh_m;
    __shared__ float sh_l;
    __shared__ float sh_a;
    __shared__ float sh_b;

    // Rotate query: sign_flip + WHT + 1/sqrt(D)
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

        // ---- 4-bit K decode → dot product ----------------------------------
        const uint8_t* k_codes = page_base + layout.k_codes_offset +
            turbo_token_head_code_offset(layout, token_in_block, head_idx,
                                         cfg.num_kv_heads);
        const half* k_scale_ptr = reinterpret_cast<const half*>(
            page_base + layout.k_norms_offset +
            turbo_token_head_norm_offset(layout, token_in_block, head_idx,
                                         cfg.num_kv_heads));

        float ks   = h2f(*k_scale_ptr);
        float kval = kTurboCodebook[unpack_4bit_get(k_codes, tid)] * ks;

        red[tid] = qrot[tid] * kval;
        __syncthreads();
        for (int stride = D >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) red[tid] += red[tid + stride];
            __syncthreads();
        }
        // red[0] * inv_sqrt_d = logit = <qrot, krot> / sqrt(D)

        // ---- Online softmax update (thread 0) ------------------------------
        if (tid == 0) {
            float logit = red[0] * inv_sqrt_d;
            float m_new = fmaxf(sh_m, logit);
            sh_a = expf(sh_m - m_new);
            sh_b = expf(logit - m_new);
            sh_l = sh_l * sh_a + sh_b;
            sh_m = m_new;
        }
        __syncthreads();

        vaccum[tid] *= sh_a;

        // ---- 4-bit V decode → accumulate -----------------------------------
        const uint8_t* v_codes = page_base + layout.v_codes_offset +
            turbo_token_head_code_offset(layout, token_in_block, head_idx,
                                         cfg.num_kv_heads);
        const half* v_scale_ptr = reinterpret_cast<const half*>(
            page_base + layout.v_norms_offset +
            turbo_token_head_norm_offset(layout, token_in_block, head_idx,
                                         cfg.num_kv_heads));

        float vs   = h2f(*v_scale_ptr);
        float vval = kTurboCodebook[unpack_4bit_get(v_codes, tid)] * vs;

        vaccum[tid] += sh_b * vval;
        __syncthreads();
    }

    // Normalise
    __shared__ float sh_inv_l;
    if (tid == 0) sh_inv_l = (sh_l > 0.0f) ? (1.0f / sh_l) : 1.0f;
    __syncthreads();
    vaccum[tid] *= sh_inv_l;

    // Inverse Hadamard + sign-unflip → FP16 output
    __syncthreads();
    hadamard_inplace<MAX_D>(vaccum, D);

    int obase = (q_idx * cfg.num_kv_heads + head_idx) * D;
    output[obase + tid] = f2h(vaccum[tid] * inv_sqrt_d * sign_flip(tid, head_idx));
}

} // namespace

void launch_tq_turbo_mse_pack_kv(
    const half* key,
    const half* value,
    const int32_t* slot_mapping,
    uint8_t* page_pool,
    const TQTurbomsePageLayout& layout,
    const TQConfig& cfg,
    int num_tokens,
    cudaStream_t stream) {

    if (cfg.head_dim > 128)
        throw std::runtime_error("turbo_mse pack: head_dim > 128 not supported");
    dim3 grid(num_tokens, cfg.num_kv_heads);
    int threads = cfg.head_dim / 2;
    size_t shmem = sizeof(float) * (2 * 128 + 64);
    turbo_mse_pack_kv_kernel<128><<<grid, threads, shmem, stream>>>(
        key, value, slot_mapping, page_pool, layout, cfg, num_tokens);
}

void launch_tq_turbo_mse_dequant_kv(
    const uint8_t* page_pool,
    const int32_t* slot_mapping,
    half* out_key,
    half* out_value,
    const TQTurbomsePageLayout& layout,
    const TQConfig& cfg,
    int num_tokens,
    cudaStream_t stream) {

    if (cfg.head_dim > 128)
        throw std::runtime_error("turbo_mse dequant: head_dim > 128 not supported");
    dim3 grid(num_tokens, cfg.num_kv_heads);
    int threads = cfg.head_dim / 2;
    size_t shmem = sizeof(float) * 2 * 128;
    turbo_mse_dequant_kv_kernel<128><<<grid, threads, shmem, stream>>>(
        page_pool, slot_mapping, out_key, out_value, layout, cfg, num_tokens);
}

void launch_tq_turbo_mse_fused_attention_output(
    const half* query,
    const uint8_t* page_pool,
    const int32_t* slot_mapping,
    half* output,
    const TQTurbomsePageLayout& layout,
    const TQConfig& cfg,
    int num_queries,
    int num_kv_tokens,
    cudaStream_t stream) {

    if (cfg.head_dim > 128)
        throw std::runtime_error("turbo_mse fused attn: head_dim > 128 not supported");
    dim3 grid(num_queries, cfg.num_kv_heads);
    int threads = cfg.head_dim;
    // smem: qrot[128] + vaccum[128] + red[128]  (sh_* are static __shared__)
    size_t shmem = sizeof(float) * (3 * 128);
    turbo_mse_fused_attn_online_kernel<128><<<grid, threads, shmem, stream>>>(
        query, page_pool, slot_mapping, output,
        layout, cfg, num_queries, num_kv_tokens);
}
