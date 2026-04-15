#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>
#include <algorithm>
#include <iomanip>

#include "tq_turbo_v6.cuh"

#define CUDA_CHECK(x) do {                                           \
    cudaError_t err__ = (x);                                         \
    if (err__ != cudaSuccess) {                                      \
        std::cerr << "CUDA error: " << cudaGetErrorString(err__)     \
                  << " @ " << __FILE__ << ":" << __LINE__ << "\n";   \
        std::exit(1);                                                \
    }                                                                \
} while (0)

static float frand(std::mt19937& rng) {
    static std::normal_distribution<float> dist(0.0f, 1.0f);
    return dist(rng);
}

static float sign_flip_host(int idx) {
    unsigned x = static_cast<unsigned>(idx) * 1103515245u + 12345u;
    return (x & 1u) ? 1.0f : -1.0f;
}

static void hadamard_inplace_host(std::vector<float>& x, int D) {
    for (int len = 1; len < D; len <<= 1) {
        for (int i = 0; i < D; i += (len << 1)) {
            for (int j = 0; j < len; ++j) {
                float a = x[i + j];
                float b = x[i + j + len];
                x[i + j] = a + b;
                x[i + j + len] = a - b;
            }
        }
    }
}

static float mse_half(const std::vector<half>& a, const std::vector<half>& b) {
    double acc = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        float x = __half2float(a[i]);
        float y = __half2float(b[i]);
        double d = double(x) - double(y);
        acc += d * d;
    }
    return float(acc / std::max<size_t>(1, a.size()));
}

static void ref_logits_rot_domain(
    const std::vector<half>& q,
    const std::vector<half>& k,
    int num_queries,
    int num_kv_tokens,
    int num_heads,
    int D,
    std::vector<float>& out_logits) {

    out_logits.assign((size_t)num_queries * num_heads * num_kv_tokens, 0.0f);

    for (int qi = 0; qi < num_queries; ++qi) {
        for (int h = 0; h < num_heads; ++h) {
            const int qbase = (qi * num_heads + h) * D;
            for (int ki = 0; ki < num_kv_tokens; ++ki) {
                const int kbase = (ki * num_heads + h) * D;
                float acc = 0.0f;
                for (int d = 0; d < D; ++d) {
                    acc += __half2float(q[qbase + d]) * __half2float(k[kbase + d]);
                }
                out_logits[(qi * num_heads + h) * num_kv_tokens + ki] = acc;
            }
        }
    }
}

static uint32_t unpack_3bit_host(const uint8_t* buf, int idx) {
    int bit_pos = idx * 3;
    int byte_idx = bit_pos >> 3;
    int bit_off = bit_pos & 7;
    uint32_t word = uint32_t(buf[byte_idx]);
    word |= (uint32_t(buf[byte_idx + 1]) << 8);
    word |= (uint32_t(buf[byte_idx + 2]) << 16);
    return (word >> bit_off) & 0x7u;
}

static uint32_t unpack_4bit_host(const uint8_t* buf, int idx) {
    uint8_t x = buf[idx >> 1];
    return (idx & 1) ? ((x >> 4) & 0xFu) : (x & 0xFu);
}

static uint32_t unpack_1bit_host(const uint8_t* buf, int idx) {
    int byte_idx = idx >> 3;
    int bit_off = idx & 7;
    return (buf[byte_idx] >> bit_off) & 0x1u;
}

static void make_rot_domain_host(
    const std::vector<half>& src,
    int token_idx,
    int head_idx,
    const TQConfig& cfg,
    std::vector<float>& out_rot) {

    const int D = cfg.head_dim;
    const int base = (token_idx * cfg.num_kv_heads + head_idx) * D;
    out_rot.resize(D);

    for (int d = 0; d < D; ++d) {
        out_rot[d] = __half2float(src[base + d]) * sign_flip_host(d);
    }
    hadamard_inplace_host(out_rot, D);

    float inv_sqrt_d = 1.0f / std::sqrt(float(D));
    for (int d = 0; d < D; ++d) out_rot[d] *= inv_sqrt_d;
}

static void dump_debug_token_head(
    const std::vector<uint8_t>& h_page_pool,
    const TQTurboV6PageLayout& layout,
    const TQConfig& cfg,
    const std::vector<half>& h_k,
    const std::vector<half>& h_v,
    const std::vector<half>& h_k_deq,
    const std::vector<half>& h_v_deq,
    const std::vector<float>& h_k_rot_dbg,
    const std::vector<float>& h_v_rot_dbg,
    const std::vector<float>& h_kn_dbg,
    const std::vector<float>& h_vn_dbg,
    const std::vector<int>& h_kidx_dbg,
    const std::vector<int>& h_vidx_dbg,
    int token_idx,
    int head_idx) {

    static const float kK3Codebook[8] = {
        -2.40f, -1.45f, -0.82f, -0.24f,
         0.24f,  0.82f,  1.45f,  2.40f
    };

    static const float kV4Codebook[16] = {
        -2.7326f, -2.0690f, -1.6180f, -1.2562f,
        -0.9423f, -0.6568f, -0.3880f, -0.1284f,
         0.1284f,  0.3880f,  0.6568f,  0.9423f,
         1.2562f,  1.6180f,  2.0690f,  2.7326f
    };

    const int block_idx = token_idx / cfg.block_size;
    const int token_in_block = token_idx % cfg.block_size;
    const size_t page_base = (size_t)block_idx * layout.page_size_bytes;

    const size_t off_k3 = turbo_v6_token_head_offset(
        token_in_block, head_idx, cfg.num_kv_heads, layout.k3_bytes_per_token_head);
    const size_t off_kres = turbo_v6_token_head_offset(
        token_in_block, head_idx, cfg.num_kv_heads, layout.kres_bytes_per_token_head);
    const size_t off_ks = turbo_v6_token_head_offset(
        token_in_block, head_idx, cfg.num_kv_heads, layout.scale_bytes_per_token_head);
    const size_t off_v4 = turbo_v6_token_head_offset(
        token_in_block, head_idx, cfg.num_kv_heads, layout.v4_bytes_per_token_head);
    const size_t off_vs = turbo_v6_token_head_offset(
        token_in_block, head_idx, cfg.num_kv_heads, layout.scale_bytes_per_token_head);

    const uint8_t* k3 = h_page_pool.data() + page_base + layout.k3_codes_offset + off_k3;
    const uint8_t* kres = h_page_pool.data() + page_base + layout.k_residual_offset + off_kres;
    const uint8_t* v4 = h_page_pool.data() + page_base + layout.v4_codes_offset + off_v4;

    const half* ks = reinterpret_cast<const half*>(
        h_page_pool.data() + page_base + layout.k_scales_offset + off_ks);
    const half* vs = reinterpret_cast<const half*>(
        h_page_pool.data() + page_base + layout.v_scales_offset + off_vs);

    const int D = cfg.head_dim;
    const int base = (token_idx * cfg.num_kv_heads + head_idx) * D;

    std::vector<float> k_rot_orig, v_rot_orig, k_rot_deq, v_rot_deq;
    make_rot_domain_host(h_k, token_idx, head_idx, cfg, k_rot_orig);
    make_rot_domain_host(h_v, token_idx, head_idx, cfg, v_rot_orig);
    make_rot_domain_host(h_k_deq, token_idx, head_idx, cfg, k_rot_deq);
    make_rot_domain_host(h_v_deq, token_idx, head_idx, cfg, v_rot_deq);

    float k_scale = __half2float(ks[0]);
    float v_scale = __half2float(vs[0]);

    std::cout << "\n=== DEBUG token=" << token_idx << " head=" << head_idx << " ===\n";
    std::cout << "k_scale=" << k_scale << "\n";
    std::cout << "v_scale=" << v_scale << "\n";

    std::cout << "\nK original-domain first 16 dims:\n";
    std::cout << "dim  orig        code3  rbit  deq\n";
    for (int d = 0; d < 16; ++d) {
        uint32_t c = unpack_3bit_host(k3, d);
        uint32_t r = unpack_1bit_host(kres, d);
        float x0 = __half2float(h_k[base + d]);
        float x1 = __half2float(h_k_deq[base + d]);
        std::cout << std::setw(3) << d << "  "
                << std::setw(10) << x0 << "  "
                << std::setw(5) << c << "  "
                << std::setw(4) << r << "  "
                << std::setw(10) << x1 << "\n";
    }

    std::cout << "\\nK rotated first 16 dims:\\n";
    std::cout << "dim  rot_orig     pack_rot      kn       kidx  code3  rbit  recon*scale  rot_deq\n";
    for (int d = 0; d < 16; ++d) {
        uint32_t c = unpack_3bit_host(k3, d);
        uint32_t r = unpack_1bit_host(kres, d);
        float kresidual = r ? 0.125f : -0.125f;
        float rec = (kK3Codebook[c] + kresidual) * k_scale;

        std::cout << std::setw(3) << d << "  "
                << std::setw(10) << k_rot_orig[d] << "  "
                << std::setw(10) << h_k_rot_dbg[base + d] << "  "
                << std::setw(8) << h_kn_dbg[base + d] << "  "
                << std::setw(5) << h_kidx_dbg[base + d] << "  "
                << std::setw(5) << c << "  "
                << std::setw(4) << r << "  "
                << std::setw(11) << rec << "  "
                << std::setw(10) << k_rot_deq[d] << "\n";
    }

    std::cout << "\nK residual bytes first 8:\n";
    for (int i = 0; i < std::min(8, layout.kres_bytes_per_token_head); ++i) {
        std::cout << int(kres[i]) << (i + 1 == std::min(8, layout.kres_bytes_per_token_head) ? '\n' : ' ');
    }

    std::cout << "K residual bits first 16:\n";
    for (int d = 0; d < std::min(16, D); ++d) {
        std::cout << int(unpack_1bit_host(kres, d))
                << (d + 1 == std::min(16, D) ? '\n' : ' ');
    }


    std::cout << "\nV original-domain first 16 dims:\n";
    std::cout << "dim  orig        code4  deq\n";
    for (int d = 0; d < 16; ++d) {
        uint32_t c = unpack_4bit_host(v4, d);
        float x0 = __half2float(h_v[base + d]);
        float x1 = __half2float(h_v_deq[base + d]);
        std::cout << std::setw(3) << d << "  "
                  << std::setw(10) << x0 << "  "
                  << std::setw(5) << c << "  "
                  << std::setw(10) << x1 << "\n";
    }

    std::cout << "\\nV rotated first 16 dims:\\n";
    std::cout << "dim  rot_orig     pack_rot      vn       vidx  code4  code*scale  rot_deq\\n";
    for (int d = 0; d < 16; ++d) {
        uint32_t c = unpack_4bit_host(v4, d);
        float rec = kV4Codebook[c] * v_scale;
        std::cout << std::setw(3) << d << "  "
                  << std::setw(10) << v_rot_orig[d] << "  "
                  << std::setw(10) << h_v_rot_dbg[base + d] << "  "
                  << std::setw(8) << h_vn_dbg[base + d] << "  "
                  << std::setw(5) << h_vidx_dbg[base + d] << "  "
                  << std::setw(5) << c << "  "
                  << std::setw(10) << rec << "  "
                  << std::setw(10) << v_rot_deq[d] << "\\n";

    }

    std::cout << "\nK code bytes first 8:\n";
    for (int i = 0; i < std::min(8, layout.k3_bytes_per_token_head); ++i) {
        std::cout << int(k3[i]) << (i + 1 == std::min(8, layout.k3_bytes_per_token_head) ? '\n' : ' ');
    }

    std::cout << "V code bytes first 8:\n";
    for (int i = 0; i < std::min(8, layout.v4_bytes_per_token_head); ++i) {
        std::cout << int(v4[i]) << (i + 1 == std::min(8, layout.v4_bytes_per_token_head) ? '\n' : ' ');
    }
}

int main() {
    TQConfig cfg{};
    cfg.num_kv_heads = 8;
    cfg.head_dim = 128;
    cfg.block_size = 16;
    cfg.group_size = 128;

    const int num_kv_tokens = 128;
    const int num_queries = 16;
    const int H = cfg.num_kv_heads;
    const int D = cfg.head_dim;

    std::mt19937 rng(1234);

    std::vector<half> h_k((size_t)num_kv_tokens * H * D);
    std::vector<half> h_v((size_t)num_kv_tokens * H * D);
    std::vector<half> h_q((size_t)num_queries * H * D);

    for (auto& x : h_k) x = __float2half(frand(rng));
    for (auto& x : h_v) x = __float2half(frand(rng));
    for (auto& x : h_q) x = __float2half(frand(rng));

    std::vector<int32_t> h_slot_mapping(num_kv_tokens);
    for (int i = 0; i < num_kv_tokens; ++i) h_slot_mapping[i] = i;

    TQTurboV6PageLayout layout = make_tq_turbo_v6_page_layout(cfg);
    const int num_blocks = (num_kv_tokens + cfg.block_size - 1) / cfg.block_size;
    const size_t page_pool_bytes = (size_t)num_blocks * layout.page_size_bytes;

    std::cout << "=== V6b Layout (3-bit K + 1-bit residual, 4-bit V, residual ON) ===\n";
    std::cout << "head_dim=" << D
              << " num_heads=" << H
              << " block_size=" << cfg.block_size
              << " num_kv_tokens=" << num_kv_tokens
              << " num_queries=" << num_queries << "\n";
    std::cout << "k3_bytes_per_token_head=" << layout.k3_bytes_per_token_head << "\n";
    std::cout << "kres_bytes_per_token_head=" << layout.kres_bytes_per_token_head << "\n";
    std::cout << "v4_bytes_per_token_head=" << layout.v4_bytes_per_token_head << "\n";
    std::cout << "scale_bytes_per_token_head=" << layout.scale_bytes_per_token_head << "\n";
    std::cout << "page_size_bytes=" << layout.page_size_bytes << "\n";
    std::cout << "page_pool_bytes=" << page_pool_bytes << "\n";

    const float bf16_kv_bytes_per_token = float(H * D * 2 * sizeof(half));
    const float v6_bytes_per_token =
        float(H * (layout.k3_bytes_per_token_head +
                   layout.kres_bytes_per_token_head +
                   layout.scale_bytes_per_token_head +
                   layout.v4_bytes_per_token_head +
                   layout.scale_bytes_per_token_head));

    std::cout << "bf16_kv_bytes_per_token=" << bf16_kv_bytes_per_token << "\n";
    std::cout << "v6_bytes_per_token=" << v6_bytes_per_token << "\n";
    std::cout << "compression_ratio_vs_bf16=" << (bf16_kv_bytes_per_token / v6_bytes_per_token) << "\n";

    half* d_k = nullptr;
    half* d_v = nullptr;
    half* d_q = nullptr;
    half* d_k_deq = nullptr;
    half* d_v_deq = nullptr;
    int32_t* d_slot = nullptr;
    uint8_t* d_page_pool = nullptr;
    float* d_logits = nullptr;
    float* d_k_rot_dbg = nullptr;
    float* d_v_rot_dbg = nullptr;
    float* d_kn_dbg = nullptr;
    float* d_vn_dbg = nullptr;
    int* d_kidx_dbg = nullptr;
    int* d_vidx_dbg = nullptr;

    CUDA_CHECK(cudaMalloc(&d_k, h_k.size() * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_v, h_v.size() * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_q, h_q.size() * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_k_deq, h_k.size() * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_v_deq, h_v.size() * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_slot, h_slot_mapping.size() * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_page_pool, page_pool_bytes));
    CUDA_CHECK(cudaMalloc(&d_logits, (size_t)num_queries * H * num_kv_tokens * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_k_rot_dbg, h_k.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v_rot_dbg, h_v.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_kn_dbg, h_k.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_vn_dbg, h_v.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_kidx_dbg, h_k.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_vidx_dbg, h_v.size() * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_k, h_k.data(), h_k.size() * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, h_v.data(), h_v.size() * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), h_q.size() * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_slot, h_slot_mapping.data(), h_slot_mapping.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_page_pool, 0, page_pool_bytes));
    CUDA_CHECK(cudaMemset(d_logits, 0, (size_t)num_queries * H * num_kv_tokens * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_k_rot_dbg, 0, h_k.size() * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_v_rot_dbg, 0, h_v.size() * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_kn_dbg, 0, h_k.size() * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_vn_dbg, 0, h_v.size() * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_kidx_dbg, 0, h_k.size() * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_vidx_dbg, 0, h_v.size() * sizeof(int)));

    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));

    CUDA_CHECK(cudaEventRecord(e0));
    launch_tq_turbo_v6_pack_kv(
        d_k, d_v, d_slot, d_page_pool, layout, cfg, num_kv_tokens, 0,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float pack_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&pack_ms, e0, e1));

    CUDA_CHECK(cudaEventRecord(e0));
    launch_tq_turbo_v6_dequant_kv(
        d_page_pool, d_slot, d_k_deq, d_v_deq, layout, cfg, num_kv_tokens, 0);
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float dequant_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&dequant_ms, e0, e1));

    CUDA_CHECK(cudaEventRecord(e0));
    launch_tq_turbo_v6_fused_attention_logits(
        d_q, d_page_pool, d_slot, d_logits, layout, cfg, num_queries, num_kv_tokens, 0);
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float fused_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&fused_ms, e0, e1));

    std::vector<half> h_k_deq(h_k.size());
    std::vector<half> h_v_deq(h_v.size());
    std::vector<float> h_logits((size_t)num_queries * H * num_kv_tokens);
    std::vector<uint8_t> h_page_pool(page_pool_bytes);
    std::vector<float> h_k_rot_dbg(h_k.size());
    std::vector<float> h_v_rot_dbg(h_v.size());
    std::vector<float> h_kn_dbg(h_k.size());
    std::vector<float> h_vn_dbg(h_v.size());
    std::vector<int> h_kidx_dbg(h_k.size());
    std::vector<int> h_vidx_dbg(h_v.size());

    CUDA_CHECK(cudaMemcpy(h_k_deq.data(), d_k_deq, h_k_deq.size() * sizeof(half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_v_deq.data(), d_v_deq, h_v_deq.size() * sizeof(half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_logits.data(), d_logits, h_logits.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_page_pool.data(), d_page_pool, h_page_pool.size() * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_k_rot_dbg.data(), d_k_rot_dbg, h_k_rot_dbg.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_v_rot_dbg.data(), d_v_rot_dbg, h_v_rot_dbg.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_kn_dbg.data(), d_kn_dbg, h_kn_dbg.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vn_dbg.data(), d_vn_dbg, h_vn_dbg.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_kidx_dbg.data(), d_kidx_dbg, h_kidx_dbg.size() * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vidx_dbg.data(), d_vidx_dbg, h_vidx_dbg.size() * sizeof(int), cudaMemcpyDeviceToHost));

    float k_mse = mse_half(h_k, h_k_deq);
    float v_mse = mse_half(h_v, h_v_deq);

    std::vector<float> ref_logits_original;
    ref_logits_rot_domain(h_q, h_k, num_queries, num_kv_tokens, H, D, ref_logits_original);

    std::vector<float> ref_logits_deq;
    ref_logits_rot_domain(h_q, h_k_deq, num_queries, num_kv_tokens, H, D, ref_logits_deq);

    double mse_vs_original = 0.0;
    double abs_mean_vs_original = 0.0;
    double abs_max_vs_original = 0.0;

    double mse_vs_deq = 0.0;
    double abs_mean_vs_deq = 0.0;
    double abs_max_vs_deq = 0.0;

    for (size_t i = 0; i < h_logits.size(); ++i) {
        double d0 = double(h_logits[i]) - double(ref_logits_original[i]);
        mse_vs_original += d0 * d0;
        abs_mean_vs_original += std::abs(d0);
        abs_max_vs_original = std::max(abs_max_vs_original, std::abs(d0));

        double d1 = double(h_logits[i]) - double(ref_logits_deq[i]);
        mse_vs_deq += d1 * d1;
        abs_mean_vs_deq += std::abs(d1);
        abs_max_vs_deq = std::max(abs_max_vs_deq, std::abs(d1));
    }

    mse_vs_original /= std::max<size_t>(1, h_logits.size());
    abs_mean_vs_original /= std::max<size_t>(1, h_logits.size());
    mse_vs_deq /= std::max<size_t>(1, h_logits.size());
    abs_mean_vs_deq /= std::max<size_t>(1, h_logits.size());

    std::cout << "\n=== V6b Results ===\n";
    std::cout << "pack_ms=" << pack_ms << "\n";
    std::cout << "dequant_ms=" << dequant_ms << "\n";
    std::cout << "fused_logits_ms=" << fused_ms << "\n";
    std::cout << "k_mse=" << k_mse << "\n";
    std::cout << "v_mse=" << v_mse << "\n";
    std::cout << "logit_mse_vs_original=" << mse_vs_original << "\n";
    std::cout << "logit_abs_mean_vs_original=" << abs_mean_vs_original << "\n";
    std::cout << "logit_abs_max_vs_original=" << abs_max_vs_original << "\n";
    std::cout << "logit_mse_vs_dequant=" << mse_vs_deq << "\n";
    std::cout << "logit_abs_mean_vs_dequant=" << abs_mean_vs_deq << "\n";
    std::cout << "logit_abs_max_vs_dequant=" << abs_max_vs_deq << "\n";

    dump_debug_token_head(h_page_pool, layout, cfg, h_k, h_v, h_k_deq, h_v_deq,
                          h_k_rot_dbg, h_v_rot_dbg, h_kn_dbg, h_vn_dbg,
                          h_kidx_dbg, h_vidx_dbg, 0, 0);

    CUDA_CHECK(cudaFree(d_k));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_k_deq));
    CUDA_CHECK(cudaFree(d_v_deq));
    CUDA_CHECK(cudaFree(d_slot));
    CUDA_CHECK(cudaFree(d_page_pool));
    CUDA_CHECK(cudaFree(d_logits));
    CUDA_CHECK(cudaFree(d_k_rot_dbg));
    CUDA_CHECK(cudaFree(d_v_rot_dbg));
    CUDA_CHECK(cudaFree(d_kn_dbg));
    CUDA_CHECK(cudaFree(d_vn_dbg));
    CUDA_CHECK(cudaFree(d_kidx_dbg));
    CUDA_CHECK(cudaFree(d_vidx_dbg));
    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));

    return 0;
}
