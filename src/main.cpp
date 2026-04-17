#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <cstring>
#include <fstream>
#include <functional>
#include <numeric>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "tq_layout.h"
#include "tq_dense_layout.h"
#include "tq_turbo_mse_layout.h"
#include "tq_allocator.h"
#include "tq_block_table.h"
#include "tq_pack_kernels.cuh"
#include "tq_dequant_kernels.cuh"
#include "tq_dense_kernels.cuh"
#include "tq_turbo_mse_kernels.cuh"
#include "tq_turbo_prod.cuh"
#include "tq_qjl.cuh"
#include "tq_polar.cuh"
#include "tq_attention_ref.h"

#define CUDA_CHECK(call)                                                          \
do {                                                                              \
    cudaError_t err__ = (call);                                                   \
    if (err__ != cudaSuccess) {                                                   \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__              \
                  << " code=" << static_cast<int>(err__)                          \
                  << " (" << cudaGetErrorString(err__) << ")" << std::endl;       \
        std::exit(EXIT_FAILURE);                                                  \
    }                                                                             \
} while (0)

struct StatSummary {
    double mean = 0.0;
    double p50 = 0.0;
    double p95 = 0.0;
    double min = 0.0;
    double max = 0.0;
};

struct ErrorSummary {
    double k_mse = 0.0;
    double v_mse = 0.0;
    double attn_mse_mean = 0.0;
    double attn_mse_max = 0.0;
    double attn_max_abs_mean = 0.0;
    double attn_max_abs_max = 0.0;
};

struct BenchResult {
    int num_tokens = 0;
    int block_size = 0;
    int num_kv_heads = 0;
    int head_dim = 0;
    int group_size = 0;
    int warmup_iters = 0;
    int measure_iters = 0;

    size_t dense_page_bytes = 0;
    size_t uniform_page_bytes = 0;
    size_t turbo_page_bytes = 0;
    size_t v6_page_bytes = 0;

    double uniform_compression_ratio = 0.0;
    double turbo_compression_ratio = 0.0;
    double v6_compression_ratio = 0.0;

    double dense_k_mse = 0.0;
    double dense_v_mse = 0.0;

    ErrorSummary uniform_err;
    ErrorSummary turbo_err;
    ErrorSummary v6_err;           // dequant-based error (vs dense KV)

    // V6 fused attention: measured against unscaled dense reference
    // (logit = <q,k>, no 1/sqrt(D), matching the fused kernel convention)
    double v6_fused_attn_mse = 0.0;
    double v6_fused_attn_max_abs = 0.0;

    size_t qjl_page_bytes = 0;
    double qjl_compression_ratio = 0.0;
    ErrorSummary qjl_err;
    double qjl_fused_attn_mse = 0.0;
    double qjl_fused_attn_max_abs = 0.0;

    StatSummary dense_store_ms;
    StatSummary dense_load_ms;
    StatSummary uniform_pack_ms;
    StatSummary uniform_dequant_ms;
    StatSummary turbo_pack_ms;
    StatSummary turbo_dequant_ms;
    StatSummary v6_pack_ms;
    StatSummary v6_dequant_ms;
    StatSummary v6_fused_attn_ms;
    StatSummary qjl_pack_ms;
    StatSummary qjl_dequant_ms;
    StatSummary qjl_fused_attn_ms;

    // turbo_mse fused attention (reads directly from compressed page pool)
    double turbo_fused_attn_mse     = 0.0;
    double turbo_fused_attn_max_abs = 0.0;
    StatSummary turbo_fused_attn_ms;

    // Polar (K=2-bit + V=3-bit, 6.1× compression, same memory as QJL but better K)
    size_t polar_page_bytes = 0;
    double polar_compression_ratio = 0.0;
    ErrorSummary polar_err;
    double polar_fused_attn_mse     = 0.0;
    double polar_fused_attn_max_abs = 0.0;
    StatSummary polar_pack_ms;
    StatSummary polar_dequant_ms;
    StatSummary polar_fused_attn_ms;
};

static float h2f(half x) { return __half2float(x); }

static std::vector<half> gather_head_tokens(
    const std::vector<half>& x,
    int num_tokens,
    int num_heads,
    int head_dim,
    int head_idx) {
    std::vector<half> out(num_tokens * head_dim);
    for (int t = 0; t < num_tokens; ++t) {
        int src = (t * num_heads + head_idx) * head_dim;
        int dst = t * head_dim;
        for (int d = 0; d < head_dim; ++d) out[dst + d] = x[src + d];
    }
    return out;
}

static double percentile_sorted(const std::vector<float>& sorted_vals, double p) {
    if (sorted_vals.empty()) return 0.0;
    if (sorted_vals.size() == 1) return sorted_vals[0];
    double idx = p * (sorted_vals.size() - 1);
    size_t lo = static_cast<size_t>(std::floor(idx));
    size_t hi = static_cast<size_t>(std::ceil(idx));
    double frac = idx - lo;
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac;
}

static StatSummary summarize(std::vector<float> vals) {
    StatSummary s;
    if (vals.empty()) return s;
    std::sort(vals.begin(), vals.end());
    s.min = vals.front();
    s.max = vals.back();
    s.p50 = percentile_sorted(vals, 0.50);
    s.p95 = percentile_sorted(vals, 0.95);
    s.mean = std::accumulate(vals.begin(), vals.end(), 0.0) / vals.size();
    return s;
}

static float time_cuda_op_once(const std::function<void()>& fn) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, 0));
    fn();
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms;
}

static StatSummary benchmark_cuda_op(
    const std::function<void()>& fn,
    int warmup_iters,
    int measure_iters) {

    for (int i = 0; i < warmup_iters; ++i) {
        fn();
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    std::vector<float> samples;
    samples.reserve(measure_iters);
    for (int i = 0; i < measure_iters; ++i) {
        float ms = time_cuda_op_once(fn);
        samples.push_back(ms);
    }
    return summarize(samples);
}

// Unscaled single-head attention reference.
// Logit convention: logit_t = <q, k_t>  (no 1/sqrt(D)).
// This matches the V6 fused attention kernel, allowing a fair comparison that
// measures only quantization error rather than a formula mismatch.
static std::vector<float> attention_unscaled_single_head(
    const std::vector<half>& q,
    const std::vector<half>& k_all,
    const std::vector<half>& v_all,
    int num_tokens,
    int head_dim) {

    std::vector<float> scores(num_tokens, 0.f);
    for (int t = 0; t < num_tokens; ++t) {
        float acc = 0.f;
        for (int d = 0; d < head_dim; ++d)
            acc += h2f(q[d]) * h2f(k_all[t * head_dim + d]);
        scores[t] = acc;
    }

    float mx = *std::max_element(scores.begin(), scores.end());
    float denom = 0.f;
    for (int t = 0; t < num_tokens; ++t) {
        scores[t] = std::exp(scores[t] - mx);
        denom += scores[t];
    }
    for (float& s : scores) s /= denom;

    std::vector<float> out(head_dim, 0.f);
    for (int t = 0; t < num_tokens; ++t)
        for (int d = 0; d < head_dim; ++d)
            out[d] += scores[t] * h2f(v_all[t * head_dim + d]);
    return out;
}

// Compute mean MSE and mean max-abs-error for a single-pass fused attention
// output vs the unscaled dense reference (logit = <q,k>, no 1/sqrt(D)).
static std::pair<double,double> compute_fused_attn_error(
    const std::vector<half>& h_fused_out,   // [H * D] fused output
    const std::vector<half>& h_query,       // [H * D] query
    const std::vector<half>& h_dense_out_k, // [T * H * D] dense K
    const std::vector<half>& h_dense_out_v, // [T * H * D] dense V
    int T, int H, int D) {

    double mse_sum = 0.0, max_abs_sum = 0.0;
    for (int head_idx = 0; head_idx < H; ++head_idx) {
        auto q_head = std::vector<half>(
            h_query.begin() + head_idx * D,
            h_query.begin() + (head_idx + 1) * D);
        auto dense_k_head = gather_head_tokens(h_dense_out_k, T, H, D, head_idx);
        auto dense_v_head = gather_head_tokens(h_dense_out_v, T, H, D, head_idx);
        auto ref_out = attention_unscaled_single_head(q_head, dense_k_head, dense_v_head, T, D);

        std::vector<float> fused_head(D);
        for (int d = 0; d < D; ++d)
            fused_head[d] = h2f(h_fused_out[head_idx * D + d]);

        mse_sum     += mse_vec(ref_out, fused_head);
        max_abs_sum += max_abs_vec(ref_out, fused_head);
    }
    return { mse_sum / H, max_abs_sum / H };
}

static ErrorSummary compute_error_summary(
    const std::vector<half>& h_key,
    const std::vector<half>& h_value,
    const std::vector<half>& h_query,
    const std::vector<half>& out_k,
    const std::vector<half>& out_v,
    const std::vector<half>& dense_k,
    const std::vector<half>& dense_v,
    int T,
    int H,
    int D) {

    ErrorSummary e;

    double mse_k = 0.0;
    double mse_v = 0.0;
    for (int i = 0; i < T * H * D; ++i) {
        double a = h2f(h_key[i]);
        double b = h2f(out_k[i]);
        double c = h2f(h_value[i]);
        double d = h2f(out_v[i]);
        mse_k += (a - b) * (a - b);
        mse_v += (c - d) * (c - d);
    }
    e.k_mse = mse_k / (T * H * D);
    e.v_mse = mse_v / (T * H * D);

    double attn_mse_sum = 0.0;
    double attn_mse_max = 0.0;
    double attn_max_abs_sum = 0.0;
    double attn_max_abs_max = 0.0;

    for (int head_idx = 0; head_idx < H; ++head_idx) {
        auto q_head = std::vector<half>(
            h_query.begin() + head_idx * D,
            h_query.begin() + (head_idx + 1) * D);

        auto dense_k_head = gather_head_tokens(dense_k, T, H, D, head_idx);
        auto dense_v_head = gather_head_tokens(dense_v, T, H, D, head_idx);
        auto out_k_head = gather_head_tokens(out_k, T, H, D, head_idx);
        auto out_v_head = gather_head_tokens(out_v, T, H, D, head_idx);

        auto attn_dense = attention_ref_single_head(q_head, dense_k_head, dense_v_head, T, D);
        auto attn_out = attention_ref_single_head(q_head, out_k_head, out_v_head, T, D);

        double mse = mse_vec(attn_dense, attn_out);
        double mx = max_abs_vec(attn_dense, attn_out);

        attn_mse_sum += mse;
        attn_max_abs_sum += mx;
        attn_mse_max = std::max(attn_mse_max, mse);
        attn_max_abs_max = std::max(attn_max_abs_max, mx);
    }

    e.attn_mse_mean = attn_mse_sum / H;
    e.attn_mse_max = attn_mse_max;
    e.attn_max_abs_mean = attn_max_abs_sum / H;
    e.attn_max_abs_max = attn_max_abs_max;
    return e;
}

static BenchResult run_one_case(
    int num_tokens,
    int block_size,
    int num_kv_heads,
    int head_dim,
    int group_size,
    int warmup_iters,
    int measure_iters) {

    TQConfig cfg;
    cfg.block_size = block_size;
    cfg.num_kv_heads = num_kv_heads;
    cfg.head_dim = head_dim;
    cfg.nbits = 4;
    cfg.group_size = group_size;

    TQPageLayout uniform_layout = make_tq_page_layout(cfg);
    TQDensePageLayout dense_layout = make_tq_dense_page_layout(cfg);
    TQTurbomsePageLayout turbo_layout = make_tq_turbo_mse_page_layout(cfg);
    TQTurboProdPageLayout v6_layout  = make_tq_turbo_prod_page_layout(cfg);
    TQQJLPageLayout     qjl_layout = make_tq_qjl_page_layout(cfg);
    TQPolarPageLayout   polar_layout = make_tq_polar_page_layout(cfg);

    TQAllocator uniform_alloc(cfg, uniform_layout, 256);
    TQAllocator dense_alloc(cfg, TQPageLayout{0,0,0,0,0,0,0,0,0,0,dense_layout.page_size_bytes}, 256);
    TQAllocator turbo_alloc(cfg, TQPageLayout{0,0,0,0,0,0,0,0,0,0,turbo_layout.page_size_bytes}, 256);
    TQAllocator v6_alloc(cfg, TQPageLayout{0,0,0,0,0,0,0,0,0,0,v6_layout.page_size_bytes}, 256);
    TQAllocator qjl_alloc(cfg, TQPageLayout{0,0,0,0,0,0,0,0,0,0,qjl_layout.page_size_bytes}, 256);
    TQAllocator polar_alloc(cfg, TQPageLayout{0,0,0,0,0,0,0,0,0,0,polar_layout.page_size_bytes}, 256);

    TQBlockTable bt_uniform(uniform_alloc, cfg);
    TQBlockTable bt_dense(dense_alloc, cfg);
    TQBlockTable bt_turbo(turbo_alloc, cfg);
    TQBlockTable bt_v6(v6_alloc, cfg);
    TQBlockTable bt_qjl(qjl_alloc, cfg);
    TQBlockTable bt_polar(polar_alloc, cfg);

    int T = num_tokens;
    int H = cfg.num_kv_heads;
    int D = cfg.head_dim;

    std::vector<float> h_key_f(T * H * D), h_value_f(T * H * D), h_query_f(H * D);
    std::mt19937 rng(0);
    std::normal_distribution<float> dist(0.f, 1.f);
    for (auto& x : h_key_f) x = dist(rng);
    for (auto& x : h_value_f) x = dist(rng);
    for (auto& x : h_query_f) x = dist(rng);

    std::vector<half> h_key(T * H * D), h_value(T * H * D), h_query(H * D);
    for (int i = 0; i < T * H * D; ++i) {
        h_key[i] = __float2half(h_key_f[i]);
        h_value[i] = __float2half(h_value_f[i]);
    }
    for (int i = 0; i < H * D; ++i) {
        h_query[i] = __float2half(h_query_f[i]);
    }

    auto slots_uniform = bt_uniform.build_slot_map(0, T);
    auto slots_dense = bt_dense.build_slot_map(0, T);
    auto slots_turbo = bt_turbo.build_slot_map(0, T);
    auto slots_v6    = bt_v6.build_slot_map(0, T);
    auto slots_qjl   = bt_qjl.build_slot_map(0, T);
    auto slots_polar = bt_polar.build_slot_map(0, T);

    half *d_key=nullptr, *d_value=nullptr;
    half *d_uniform_out_k=nullptr, *d_uniform_out_v=nullptr;
    half *d_dense_out_k=nullptr, *d_dense_out_v=nullptr;
    half *d_turbo_out_k=nullptr, *d_turbo_out_v=nullptr;
    half *d_v6_out_k=nullptr, *d_v6_out_v=nullptr;
    half *d_qjl_out_k=nullptr, *d_qjl_out_v=nullptr;
    half *d_polar_out_k=nullptr, *d_polar_out_v=nullptr;
    half *d_query=nullptr;
    half *d_v6_fused_out=nullptr;
    half *d_qjl_fused_out=nullptr;
    half *d_turbo_fused_out=nullptr;
    half *d_polar_fused_out=nullptr;
    int32_t *d_slots_uniform=nullptr, *d_slots_dense=nullptr, *d_slots_turbo=nullptr;
    int32_t *d_slots_v6=nullptr;
    int32_t *d_slots_qjl=nullptr;
    int32_t *d_slots_polar=nullptr;

    CUDA_CHECK(cudaFree(0)); // Warm up CUDA context

    CUDA_CHECK(cudaMalloc(&d_key, sizeof(half) * T * H * D));
    CUDA_CHECK(cudaMalloc(&d_value, sizeof(half) * T * H * D));
    CUDA_CHECK(cudaMalloc(&d_uniform_out_k, sizeof(half) * T * H * D));
    CUDA_CHECK(cudaMalloc(&d_uniform_out_v, sizeof(half) * T * H * D));
    CUDA_CHECK(cudaMalloc(&d_dense_out_k, sizeof(half) * T * H * D));
    CUDA_CHECK(cudaMalloc(&d_dense_out_v, sizeof(half) * T * H * D));
    CUDA_CHECK(cudaMalloc(&d_turbo_out_k, sizeof(half) * T * H * D));
    CUDA_CHECK(cudaMalloc(&d_turbo_out_v, sizeof(half) * T * H * D));
    CUDA_CHECK(cudaMalloc(&d_v6_out_k, sizeof(half) * T * H * D));
    CUDA_CHECK(cudaMalloc(&d_v6_out_v, sizeof(half) * T * H * D));
    CUDA_CHECK(cudaMalloc(&d_query, sizeof(half) * H * D));
    CUDA_CHECK(cudaMalloc(&d_v6_fused_out, sizeof(half) * H * D));   // 1 query output
    CUDA_CHECK(cudaMalloc(&d_slots_uniform, sizeof(int32_t) * T));
    CUDA_CHECK(cudaMalloc(&d_slots_dense, sizeof(int32_t) * T));
    CUDA_CHECK(cudaMalloc(&d_slots_turbo, sizeof(int32_t) * T));
    CUDA_CHECK(cudaMalloc(&d_slots_v6, sizeof(int32_t) * T));
    CUDA_CHECK(cudaMalloc(&d_qjl_out_k, sizeof(half) * T * H * D));
    CUDA_CHECK(cudaMalloc(&d_qjl_out_v, sizeof(half) * T * H * D));
    CUDA_CHECK(cudaMalloc(&d_qjl_fused_out, sizeof(half) * H * D));
    CUDA_CHECK(cudaMalloc(&d_slots_qjl, sizeof(int32_t) * T));
    CUDA_CHECK(cudaMalloc(&d_turbo_fused_out, sizeof(half) * H * D));
    CUDA_CHECK(cudaMalloc(&d_polar_out_k, sizeof(half) * T * H * D));
    CUDA_CHECK(cudaMalloc(&d_polar_out_v, sizeof(half) * T * H * D));
    CUDA_CHECK(cudaMalloc(&d_polar_fused_out, sizeof(half) * H * D));
    CUDA_CHECK(cudaMalloc(&d_slots_polar, sizeof(int32_t) * T));

    CUDA_CHECK(cudaMemcpy(d_key, h_key.data(), sizeof(half) * T * H * D, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_value, h_value.data(), sizeof(half) * T * H * D, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_slots_uniform, slots_uniform.data(), sizeof(int32_t) * T, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_slots_dense, slots_dense.data(), sizeof(int32_t) * T, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_slots_turbo, slots_turbo.data(), sizeof(int32_t) * T, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_slots_v6, slots_v6.data(), sizeof(int32_t) * T, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_slots_qjl, slots_qjl.data(), sizeof(int32_t) * T, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_slots_polar, slots_polar.data(), sizeof(int32_t) * T, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_query, h_query.data(), sizeof(half) * H * D, cudaMemcpyHostToDevice));

    auto dense_store_fn = [&]() {
        launch_dense_store_kv(d_key, d_value, d_slots_dense, dense_alloc.device_page_pool(), dense_layout, cfg, T, 0);
        CUDA_CHECK(cudaPeekAtLastError());
    };
    auto dense_load_fn = [&]() {
        launch_dense_load_kv(dense_alloc.device_page_pool(), d_slots_dense, d_dense_out_k, d_dense_out_v, dense_layout, cfg, T, 0);
        CUDA_CHECK(cudaPeekAtLastError());
    };

    auto uniform_pack_fn = [&]() {
        launch_tq_pack_kv(d_key, d_value, d_slots_uniform, uniform_alloc.device_page_pool(), uniform_layout, cfg, T, 0);
        CUDA_CHECK(cudaPeekAtLastError());
    };
    auto uniform_dequant_fn = [&]() {
        launch_tq_dequant_kv(uniform_alloc.device_page_pool(), d_slots_uniform, d_uniform_out_k, d_uniform_out_v, uniform_layout, cfg, T, 0);
        CUDA_CHECK(cudaPeekAtLastError());
    };

    auto turbo_pack_fn = [&]() {
        launch_tq_turbo_mse_pack_kv(d_key, d_value, d_slots_turbo, turbo_alloc.device_page_pool(), turbo_layout, cfg, T, 0);
        CUDA_CHECK(cudaPeekAtLastError());
    };
    auto turbo_dequant_fn = [&]() {
        launch_tq_turbo_mse_dequant_kv(turbo_alloc.device_page_pool(), d_slots_turbo, d_turbo_out_k, d_turbo_out_v, turbo_layout, cfg, T, 0);
        CUDA_CHECK(cudaPeekAtLastError());
    };

    auto v6_pack_fn = [&]() {
        launch_tq_turbo_prod_pack_kv(d_key, d_value, d_slots_v6,
            v6_alloc.device_page_pool(), v6_layout, cfg, T, 0);
        CUDA_CHECK(cudaPeekAtLastError());
    };
    auto v6_dequant_fn = [&]() {
        launch_tq_turbo_prod_dequant_kv(v6_alloc.device_page_pool(), d_slots_v6,
            d_v6_out_k, d_v6_out_v, v6_layout, cfg, T, 0);
        CUDA_CHECK(cudaPeekAtLastError());
    };
    auto v6_fused_attn_fn = [&]() {
        launch_tq_turbo_prod_fused_attention_output(
            d_query, v6_alloc.device_page_pool(), d_slots_v6,
            d_v6_fused_out, v6_layout, cfg, 1 /*num_queries*/, T, 0);
        CUDA_CHECK(cudaPeekAtLastError());
    };

    auto qjl_pack_fn = [&]() {
        launch_tq_qjl_pack_kv(d_key, d_value, d_slots_qjl,
            qjl_alloc.device_page_pool(), qjl_layout, cfg, T, 0);
        CUDA_CHECK(cudaPeekAtLastError());
    };
    auto qjl_dequant_fn = [&]() {
        launch_tq_qjl_dequant_kv(qjl_alloc.device_page_pool(), d_slots_qjl,
            d_qjl_out_k, d_qjl_out_v, qjl_layout, cfg, T, 0);
        CUDA_CHECK(cudaPeekAtLastError());
    };
    auto qjl_fused_attn_fn = [&]() {
        launch_tq_qjl_fused_attention_output(
            d_query, qjl_alloc.device_page_pool(), d_slots_qjl,
            d_qjl_fused_out, qjl_layout, cfg, 1 /*num_queries*/, T, 0);
        CUDA_CHECK(cudaPeekAtLastError());
    };
    auto turbo_fused_attn_fn = [&]() {
        launch_tq_turbo_mse_fused_attention_output(
            d_query, turbo_alloc.device_page_pool(), d_slots_turbo,
            d_turbo_fused_out, turbo_layout, cfg, 1 /*num_queries*/, T, 0);
        CUDA_CHECK(cudaPeekAtLastError());
    };

    auto polar_pack_fn = [&]() {
        launch_tq_polar_pack_kv(d_key, d_value, d_slots_polar,
            polar_alloc.device_page_pool(), polar_layout, cfg, T, 0);
        CUDA_CHECK(cudaPeekAtLastError());
    };
    auto polar_dequant_fn = [&]() {
        launch_tq_polar_dequant_kv(polar_alloc.device_page_pool(), d_slots_polar,
            d_polar_out_k, d_polar_out_v, polar_layout, cfg, T, 0);
        CUDA_CHECK(cudaPeekAtLastError());
    };
    auto polar_fused_attn_fn = [&]() {
        launch_tq_polar_fused_attention_output(
            d_query, polar_alloc.device_page_pool(), d_slots_polar,
            d_polar_fused_out, polar_layout, cfg, 1 /*num_queries*/, T, 0);
        CUDA_CHECK(cudaPeekAtLastError());
    };

    StatSummary dense_store_ms = benchmark_cuda_op(dense_store_fn, warmup_iters, measure_iters);
    StatSummary dense_load_ms = benchmark_cuda_op(dense_load_fn, warmup_iters, measure_iters);
    StatSummary uniform_pack_ms = benchmark_cuda_op(uniform_pack_fn, warmup_iters, measure_iters);
    StatSummary uniform_dequant_ms = benchmark_cuda_op(uniform_dequant_fn, warmup_iters, measure_iters);
    StatSummary turbo_pack_ms = benchmark_cuda_op(turbo_pack_fn, warmup_iters, measure_iters);
    StatSummary turbo_dequant_ms = benchmark_cuda_op(turbo_dequant_fn, warmup_iters, measure_iters);
    StatSummary v6_pack_ms = benchmark_cuda_op(v6_pack_fn, warmup_iters, measure_iters);
    StatSummary v6_dequant_ms = benchmark_cuda_op(v6_dequant_fn, warmup_iters, measure_iters);
    StatSummary v6_fused_attn_ms = benchmark_cuda_op(v6_fused_attn_fn, warmup_iters, measure_iters);
    StatSummary qjl_pack_ms = benchmark_cuda_op(qjl_pack_fn, warmup_iters, measure_iters);
    StatSummary qjl_dequant_ms = benchmark_cuda_op(qjl_dequant_fn, warmup_iters, measure_iters);
    StatSummary qjl_fused_attn_ms = benchmark_cuda_op(qjl_fused_attn_fn, warmup_iters, measure_iters);
    StatSummary turbo_fused_attn_ms = benchmark_cuda_op(turbo_fused_attn_fn, warmup_iters, measure_iters);
    StatSummary polar_pack_ms      = benchmark_cuda_op(polar_pack_fn, warmup_iters, measure_iters);
    StatSummary polar_dequant_ms   = benchmark_cuda_op(polar_dequant_fn, warmup_iters, measure_iters);
    StatSummary polar_fused_attn_ms = benchmark_cuda_op(polar_fused_attn_fn, warmup_iters, measure_iters);

    dense_store_fn();
    dense_load_fn();
    uniform_pack_fn();
    uniform_dequant_fn();
    turbo_pack_fn();
    turbo_dequant_fn();
    v6_pack_fn();
    v6_dequant_fn();
    v6_fused_attn_fn();
    qjl_pack_fn();
    qjl_dequant_fn();
    qjl_fused_attn_fn();
    turbo_fused_attn_fn();
    polar_pack_fn();
    polar_dequant_fn();
    polar_fused_attn_fn();
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<half> h_dense_out_k(T * H * D), h_dense_out_v(T * H * D);
    std::vector<half> h_uniform_out_k(T * H * D), h_uniform_out_v(T * H * D);
    std::vector<half> h_turbo_out_k(T * H * D), h_turbo_out_v(T * H * D);
    std::vector<half> h_v6_out_k(T * H * D), h_v6_out_v(T * H * D);
    std::vector<half> h_v6_fused_out(H * D);
    std::vector<half> h_qjl_out_k(T * H * D), h_qjl_out_v(T * H * D);
    std::vector<half> h_qjl_fused_out(H * D);
    std::vector<half> h_turbo_fused_out(H * D);
    std::vector<half> h_polar_out_k(T * H * D), h_polar_out_v(T * H * D);
    std::vector<half> h_polar_fused_out(H * D);

    CUDA_CHECK(cudaMemcpy(h_dense_out_k.data(), d_dense_out_k, sizeof(half) * T * H * D, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_dense_out_v.data(), d_dense_out_v, sizeof(half) * T * H * D, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_uniform_out_k.data(), d_uniform_out_k, sizeof(half) * T * H * D, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_uniform_out_v.data(), d_uniform_out_v, sizeof(half) * T * H * D, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_turbo_out_k.data(), d_turbo_out_k, sizeof(half) * T * H * D, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_turbo_out_v.data(), d_turbo_out_v, sizeof(half) * T * H * D, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_v6_out_k.data(), d_v6_out_k, sizeof(half) * T * H * D, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_v6_out_v.data(), d_v6_out_v, sizeof(half) * T * H * D, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_v6_fused_out.data(), d_v6_fused_out, sizeof(half) * H * D, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_qjl_out_k.data(), d_qjl_out_k, sizeof(half) * T * H * D, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_qjl_out_v.data(), d_qjl_out_v, sizeof(half) * T * H * D, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_qjl_fused_out.data(), d_qjl_fused_out, sizeof(half) * H * D, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_turbo_fused_out.data(), d_turbo_fused_out, sizeof(half) * H * D, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_polar_out_k.data(), d_polar_out_k, sizeof(half) * T * H * D, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_polar_out_v.data(), d_polar_out_v, sizeof(half) * T * H * D, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_polar_fused_out.data(), d_polar_fused_out, sizeof(half) * H * D, cudaMemcpyDeviceToHost));

    double dense_k_mse = 0.0, dense_v_mse = 0.0;
    for (int i = 0; i < T * H * D; ++i) {
        double a = h2f(h_key[i]);
        double b = h2f(h_dense_out_k[i]);
        double c = h2f(h_value[i]);
        double d = h2f(h_dense_out_v[i]);
        dense_k_mse += (a - b) * (a - b);
        dense_v_mse += (c - d) * (c - d);
    }
    dense_k_mse /= (T * H * D);
    dense_v_mse /= (T * H * D);

    ErrorSummary uniform_err = compute_error_summary(
        h_key, h_value, h_query, h_uniform_out_k, h_uniform_out_v, h_dense_out_k, h_dense_out_v, T, H, D);

    ErrorSummary turbo_err = compute_error_summary(
        h_key, h_value, h_query, h_turbo_out_k, h_turbo_out_v, h_dense_out_k, h_dense_out_v, T, H, D);

    ErrorSummary v6_err = compute_error_summary(
        h_key, h_value, h_query, h_v6_out_k, h_v6_out_v, h_dense_out_k, h_dense_out_v, T, H, D);

    ErrorSummary qjl_err = compute_error_summary(
        h_key, h_value, h_query, h_qjl_out_k, h_qjl_out_v, h_dense_out_k, h_dense_out_v, T, H, D);

    ErrorSummary polar_err = compute_error_summary(
        h_key, h_value, h_query, h_polar_out_k, h_polar_out_v, h_dense_out_k, h_dense_out_v, T, H, D);

    // Fused attention accuracy vs unscaled dense reference
    // logit = <q, k>  (no 1/sqrt(D)) — matches all fused kernel conventions
    auto [v6_fused_attn_mse, v6_fused_attn_max_abs] =
        compute_fused_attn_error(h_v6_fused_out, h_query, h_dense_out_k, h_dense_out_v, T, H, D);
    auto [turbo_fused_attn_mse, turbo_fused_attn_max_abs] =
        compute_fused_attn_error(h_turbo_fused_out, h_query, h_dense_out_k, h_dense_out_v, T, H, D);
    auto [qjl_fused_attn_mse, qjl_fused_attn_max_abs] =
        compute_fused_attn_error(h_qjl_fused_out, h_query, h_dense_out_k, h_dense_out_v, T, H, D);
    auto [polar_fused_attn_mse, polar_fused_attn_max_abs] =
        compute_fused_attn_error(h_polar_fused_out, h_query, h_dense_out_k, h_dense_out_v, T, H, D);

    BenchResult r;
    r.num_tokens = T;
    r.block_size = block_size;
    r.num_kv_heads = H;
    r.head_dim = D;
    r.group_size = group_size;
    r.warmup_iters = warmup_iters;
    r.measure_iters = measure_iters;

    r.dense_page_bytes   = dense_layout.page_size_bytes;
    r.uniform_page_bytes = uniform_layout.page_size_bytes;
    r.turbo_page_bytes   = turbo_layout.page_size_bytes;
    r.v6_page_bytes      = v6_layout.page_size_bytes;
    r.qjl_page_bytes     = qjl_layout.page_size_bytes;
    r.polar_page_bytes   = polar_layout.page_size_bytes;

    const double dense_bytes = static_cast<double>(dense_layout.page_size_bytes);
    r.uniform_compression_ratio = dense_bytes / uniform_layout.page_size_bytes;
    r.turbo_compression_ratio   = dense_bytes / turbo_layout.page_size_bytes;
    r.v6_compression_ratio      = dense_bytes / v6_layout.page_size_bytes;
    r.qjl_compression_ratio     = dense_bytes / qjl_layout.page_size_bytes;
    r.polar_compression_ratio   = dense_bytes / polar_layout.page_size_bytes;

    r.dense_k_mse = dense_k_mse;
    r.dense_v_mse = dense_v_mse;

    r.uniform_err = uniform_err;
    r.turbo_err   = turbo_err;
    r.v6_err      = v6_err;
    r.qjl_err     = qjl_err;
    r.polar_err   = polar_err;

    r.v6_fused_attn_mse     = v6_fused_attn_mse;
    r.v6_fused_attn_max_abs = v6_fused_attn_max_abs;

    r.qjl_fused_attn_mse     = qjl_fused_attn_mse;
    r.qjl_fused_attn_max_abs = qjl_fused_attn_max_abs;

    r.turbo_fused_attn_mse     = turbo_fused_attn_mse;
    r.turbo_fused_attn_max_abs = turbo_fused_attn_max_abs;

    r.polar_fused_attn_mse     = polar_fused_attn_mse;
    r.polar_fused_attn_max_abs = polar_fused_attn_max_abs;

    r.dense_store_ms    = dense_store_ms;
    r.dense_load_ms     = dense_load_ms;
    r.uniform_pack_ms   = uniform_pack_ms;
    r.uniform_dequant_ms = uniform_dequant_ms;
    r.turbo_pack_ms     = turbo_pack_ms;
    r.turbo_dequant_ms  = turbo_dequant_ms;
    r.v6_pack_ms        = v6_pack_ms;
    r.v6_dequant_ms     = v6_dequant_ms;
    r.v6_fused_attn_ms  = v6_fused_attn_ms;

    r.qjl_pack_ms       = qjl_pack_ms;
    r.qjl_dequant_ms    = qjl_dequant_ms;
    r.qjl_fused_attn_ms = qjl_fused_attn_ms;

    r.turbo_fused_attn_ms = turbo_fused_attn_ms;

    r.polar_pack_ms       = polar_pack_ms;
    r.polar_dequant_ms    = polar_dequant_ms;
    r.polar_fused_attn_ms = polar_fused_attn_ms;

    CUDA_CHECK(cudaFree(d_key));
    CUDA_CHECK(cudaFree(d_value));
    CUDA_CHECK(cudaFree(d_uniform_out_k));
    CUDA_CHECK(cudaFree(d_uniform_out_v));
    CUDA_CHECK(cudaFree(d_dense_out_k));
    CUDA_CHECK(cudaFree(d_dense_out_v));
    CUDA_CHECK(cudaFree(d_turbo_out_k));
    CUDA_CHECK(cudaFree(d_turbo_out_v));
    CUDA_CHECK(cudaFree(d_v6_out_k));
    CUDA_CHECK(cudaFree(d_v6_out_v));
    CUDA_CHECK(cudaFree(d_query));
    CUDA_CHECK(cudaFree(d_v6_fused_out));
    CUDA_CHECK(cudaFree(d_slots_uniform));
    CUDA_CHECK(cudaFree(d_slots_dense));
    CUDA_CHECK(cudaFree(d_slots_turbo));
    CUDA_CHECK(cudaFree(d_slots_v6));
    CUDA_CHECK(cudaFree(d_qjl_out_k));
    CUDA_CHECK(cudaFree(d_qjl_out_v));
    CUDA_CHECK(cudaFree(d_qjl_fused_out));
    CUDA_CHECK(cudaFree(d_slots_qjl));
    CUDA_CHECK(cudaFree(d_turbo_fused_out));
    CUDA_CHECK(cudaFree(d_polar_out_k));
    CUDA_CHECK(cudaFree(d_polar_out_v));
    CUDA_CHECK(cudaFree(d_polar_fused_out));
    CUDA_CHECK(cudaFree(d_slots_polar));

    return r;
}

static void write_csv(const std::string& path, const std::vector<BenchResult>& results) {
    std::ofstream ofs(path);
    ofs << "num_tokens,block_size,num_kv_heads,head_dim,group_size,warmup_iters,measure_iters,"
           "dense_page_bytes,uniform_page_bytes,turbo_page_bytes,v6_page_bytes,qjl_page_bytes,"
           "uniform_compression_ratio,turbo_compression_ratio,v6_compression_ratio,qjl_compression_ratio,"
           "dense_k_mse,dense_v_mse,"
           "uniform_k_mse,uniform_v_mse,uniform_attn_mse_mean,uniform_attn_mse_max,uniform_attn_max_abs_mean,uniform_attn_max_abs_max,"
           "turbo_k_mse,turbo_v_mse,turbo_attn_mse_mean,turbo_attn_mse_max,turbo_attn_max_abs_mean,turbo_attn_max_abs_max,"
           "v6_k_mse,v6_v_mse,v6_attn_mse_mean,v6_attn_mse_max,v6_attn_max_abs_mean,v6_attn_max_abs_max,"
           "v6_fused_attn_mse,v6_fused_attn_max_abs,"
           "qjl_k_mse,qjl_v_mse,qjl_attn_mse_mean,qjl_attn_mse_max,qjl_attn_max_abs_mean,qjl_attn_max_abs_max,"
           "qjl_fused_attn_mse,qjl_fused_attn_max_abs,"
           "dense_store_mean_ms,dense_store_p50_ms,dense_store_p95_ms,dense_store_min_ms,dense_store_max_ms,"
           "dense_load_mean_ms,dense_load_p50_ms,dense_load_p95_ms,dense_load_min_ms,dense_load_max_ms,"
           "uniform_pack_mean_ms,uniform_pack_p50_ms,uniform_pack_p95_ms,uniform_pack_min_ms,uniform_pack_max_ms,"
           "uniform_dequant_mean_ms,uniform_dequant_p50_ms,uniform_dequant_p95_ms,uniform_dequant_min_ms,uniform_dequant_max_ms,"
           "turbo_pack_mean_ms,turbo_pack_p50_ms,turbo_pack_p95_ms,turbo_pack_min_ms,turbo_pack_max_ms,"
           "turbo_dequant_mean_ms,turbo_dequant_p50_ms,turbo_dequant_p95_ms,turbo_dequant_min_ms,turbo_dequant_max_ms,"
           "v6_pack_mean_ms,v6_pack_p50_ms,v6_pack_p95_ms,v6_pack_min_ms,v6_pack_max_ms,"
           "v6_dequant_mean_ms,v6_dequant_p50_ms,v6_dequant_p95_ms,v6_dequant_min_ms,v6_dequant_max_ms,"
           "v6_fused_attn_mean_ms,v6_fused_attn_p50_ms,v6_fused_attn_p95_ms,v6_fused_attn_min_ms,v6_fused_attn_max_ms,"
           "qjl_pack_mean_ms,qjl_pack_p50_ms,qjl_pack_p95_ms,qjl_pack_min_ms,qjl_pack_max_ms,"
           "qjl_dequant_mean_ms,qjl_dequant_p50_ms,qjl_dequant_p95_ms,qjl_dequant_min_ms,qjl_dequant_max_ms,"
           "qjl_fused_attn_mean_ms,qjl_fused_attn_p50_ms,qjl_fused_attn_p95_ms,qjl_fused_attn_min_ms,qjl_fused_attn_max_ms,"
           "turbo_fused_attn_mse,turbo_fused_attn_max_abs,"
           "turbo_fused_attn_mean_ms,turbo_fused_attn_p50_ms,turbo_fused_attn_p95_ms,turbo_fused_attn_min_ms,turbo_fused_attn_max_ms,"
           "polar_page_bytes,polar_compression_ratio,"
           "polar_k_mse,polar_v_mse,polar_attn_mse_mean,polar_attn_mse_max,polar_attn_max_abs_mean,polar_attn_max_abs_max,"
           "polar_fused_attn_mse,polar_fused_attn_max_abs,"
           "polar_pack_mean_ms,polar_pack_p50_ms,polar_pack_p95_ms,polar_pack_min_ms,polar_pack_max_ms,"
           "polar_dequant_mean_ms,polar_dequant_p50_ms,polar_dequant_p95_ms,polar_dequant_min_ms,polar_dequant_max_ms,"
           "polar_fused_attn_mean_ms,polar_fused_attn_p50_ms,polar_fused_attn_p95_ms,polar_fused_attn_min_ms,polar_fused_attn_max_ms\n";

    for (const auto& r : results) {
        ofs << r.num_tokens << ","
            << r.block_size << ","
            << r.num_kv_heads << ","
            << r.head_dim << ","
            << r.group_size << ","
            << r.warmup_iters << ","
            << r.measure_iters << ","
            << r.dense_page_bytes << ","
            << r.uniform_page_bytes << ","
            << r.turbo_page_bytes << ","
            << r.v6_page_bytes << ","
            << r.qjl_page_bytes << ","
            << r.uniform_compression_ratio << ","
            << r.turbo_compression_ratio << ","
            << r.v6_compression_ratio << ","
            << r.qjl_compression_ratio << ","
            << r.dense_k_mse << ","
            << r.dense_v_mse << ","
            << r.uniform_err.k_mse << ","
            << r.uniform_err.v_mse << ","
            << r.uniform_err.attn_mse_mean << ","
            << r.uniform_err.attn_mse_max << ","
            << r.uniform_err.attn_max_abs_mean << ","
            << r.uniform_err.attn_max_abs_max << ","
            << r.turbo_err.k_mse << ","
            << r.turbo_err.v_mse << ","
            << r.turbo_err.attn_mse_mean << ","
            << r.turbo_err.attn_mse_max << ","
            << r.turbo_err.attn_max_abs_mean << ","
            << r.turbo_err.attn_max_abs_max << ","
            << r.v6_err.k_mse << ","
            << r.v6_err.v_mse << ","
            << r.v6_err.attn_mse_mean << ","
            << r.v6_err.attn_mse_max << ","
            << r.v6_err.attn_max_abs_mean << ","
            << r.v6_err.attn_max_abs_max << ","
            << r.v6_fused_attn_mse << ","
            << r.v6_fused_attn_max_abs << ","
            << r.qjl_err.k_mse << ","
            << r.qjl_err.v_mse << ","
            << r.qjl_err.attn_mse_mean << ","
            << r.qjl_err.attn_mse_max << ","
            << r.qjl_err.attn_max_abs_mean << ","
            << r.qjl_err.attn_max_abs_max << ","
            << r.qjl_fused_attn_mse << ","
            << r.qjl_fused_attn_max_abs << ","
            << r.dense_store_ms.mean << ","
            << r.dense_store_ms.p50 << ","
            << r.dense_store_ms.p95 << ","
            << r.dense_store_ms.min << ","
            << r.dense_store_ms.max << ","
            << r.dense_load_ms.mean << ","
            << r.dense_load_ms.p50 << ","
            << r.dense_load_ms.p95 << ","
            << r.dense_load_ms.min << ","
            << r.dense_load_ms.max << ","
            << r.uniform_pack_ms.mean << ","
            << r.uniform_pack_ms.p50 << ","
            << r.uniform_pack_ms.p95 << ","
            << r.uniform_pack_ms.min << ","
            << r.uniform_pack_ms.max << ","
            << r.uniform_dequant_ms.mean << ","
            << r.uniform_dequant_ms.p50 << ","
            << r.uniform_dequant_ms.p95 << ","
            << r.uniform_dequant_ms.min << ","
            << r.uniform_dequant_ms.max << ","
            << r.turbo_pack_ms.mean << ","
            << r.turbo_pack_ms.p50 << ","
            << r.turbo_pack_ms.p95 << ","
            << r.turbo_pack_ms.min << ","
            << r.turbo_pack_ms.max << ","
            << r.turbo_dequant_ms.mean << ","
            << r.turbo_dequant_ms.p50 << ","
            << r.turbo_dequant_ms.p95 << ","
            << r.turbo_dequant_ms.min << ","
            << r.turbo_dequant_ms.max << ","
            << r.v6_pack_ms.mean << ","
            << r.v6_pack_ms.p50 << ","
            << r.v6_pack_ms.p95 << ","
            << r.v6_pack_ms.min << ","
            << r.v6_pack_ms.max << ","
            << r.v6_dequant_ms.mean << ","
            << r.v6_dequant_ms.p50 << ","
            << r.v6_dequant_ms.p95 << ","
            << r.v6_dequant_ms.min << ","
            << r.v6_dequant_ms.max << ","
            << r.v6_fused_attn_ms.mean << ","
            << r.v6_fused_attn_ms.p50 << ","
            << r.v6_fused_attn_ms.p95 << ","
            << r.v6_fused_attn_ms.min << ","
            << r.v6_fused_attn_ms.max << ","
            << r.qjl_pack_ms.mean << ","
            << r.qjl_pack_ms.p50 << ","
            << r.qjl_pack_ms.p95 << ","
            << r.qjl_pack_ms.min << ","
            << r.qjl_pack_ms.max << ","
            << r.qjl_dequant_ms.mean << ","
            << r.qjl_dequant_ms.p50 << ","
            << r.qjl_dequant_ms.p95 << ","
            << r.qjl_dequant_ms.min << ","
            << r.qjl_dequant_ms.max << ","
            << r.qjl_fused_attn_ms.mean << ","
            << r.qjl_fused_attn_ms.p50 << ","
            << r.qjl_fused_attn_ms.p95 << ","
            << r.qjl_fused_attn_ms.min << ","
            << r.qjl_fused_attn_ms.max << ","
            << r.turbo_fused_attn_mse << ","
            << r.turbo_fused_attn_max_abs << ","
            << r.turbo_fused_attn_ms.mean << ","
            << r.turbo_fused_attn_ms.p50 << ","
            << r.turbo_fused_attn_ms.p95 << ","
            << r.turbo_fused_attn_ms.min << ","
            << r.turbo_fused_attn_ms.max << ","
            << r.polar_page_bytes << ","
            << r.polar_compression_ratio << ","
            << r.polar_err.k_mse << ","
            << r.polar_err.v_mse << ","
            << r.polar_err.attn_mse_mean << ","
            << r.polar_err.attn_mse_max << ","
            << r.polar_err.attn_max_abs_mean << ","
            << r.polar_err.attn_max_abs_max << ","
            << r.polar_fused_attn_mse << ","
            << r.polar_fused_attn_max_abs << ","
            << r.polar_pack_ms.mean << ","
            << r.polar_pack_ms.p50 << ","
            << r.polar_pack_ms.p95 << ","
            << r.polar_pack_ms.min << ","
            << r.polar_pack_ms.max << ","
            << r.polar_dequant_ms.mean << ","
            << r.polar_dequant_ms.p50 << ","
            << r.polar_dequant_ms.p95 << ","
            << r.polar_dequant_ms.min << ","
            << r.polar_dequant_ms.max << ","
            << r.polar_fused_attn_ms.mean << ","
            << r.polar_fused_attn_ms.p50 << ","
            << r.polar_fused_attn_ms.p95 << ","
            << r.polar_fused_attn_ms.min << ","
            << r.polar_fused_attn_ms.max << "\n";
    }
}

int main() {
    try {
        std::vector<int> num_tokens_list = {32, 128, 512};
        std::vector<int> block_sizes = {16, 32};
        std::vector<int> head_dims = {64, 128};
        int num_kv_heads = 8;
        int warmup_iters = 5;
        int measure_iters = 20;

        std::vector<BenchResult> results;

        for (int T : num_tokens_list) {
            for (int B : block_sizes) {
                for (int D : head_dims) {
                    int G = D;
                    std::cout << "Running case: "
                              << "T=" << T
                              << " B=" << B
                              << " H=" << num_kv_heads
                              << " D=" << D
                              << " G=" << G
                              << " warmup=" << warmup_iters
                              << " measure=" << measure_iters << "\n";

                    auto r = run_one_case(T, B, num_kv_heads, D, G, warmup_iters, measure_iters);
                    results.push_back(r);

                    std::cout << "  compression uniform/turbo/v6/qjl/polar="
                              << std::fixed << std::setprecision(3)
                              << r.uniform_compression_ratio << "/"
                              << r.turbo_compression_ratio << "/"
                              << r.v6_compression_ratio << "/"
                              << r.qjl_compression_ratio << "/"
                              << r.polar_compression_ratio << "\n";

                    std::cout << "  uniform k_mse=" << r.uniform_err.k_mse
                              << " v_mse=" << r.uniform_err.v_mse
                              << " attn_mean=" << r.uniform_err.attn_mse_mean
                              << " attn_max=" << r.uniform_err.attn_mse_max
                              << " abs_mean=" << r.uniform_err.attn_max_abs_mean
                              << " abs_max=" << r.uniform_err.attn_max_abs_max << "\n";

                    std::cout << "  turbo   k_mse=" << r.turbo_err.k_mse
                              << " v_mse=" << r.turbo_err.v_mse
                              << " attn_mean=" << r.turbo_err.attn_mse_mean
                              << " attn_max=" << r.turbo_err.attn_mse_max
                              << " abs_mean=" << r.turbo_err.attn_max_abs_mean
                              << " abs_max=" << r.turbo_err.attn_max_abs_max << "\n";

                    std::cout << "  v6      k_mse=" << r.v6_err.k_mse
                              << " v_mse=" << r.v6_err.v_mse
                              << " attn_mean=" << r.v6_err.attn_mse_mean
                              << " attn_max=" << r.v6_err.attn_mse_max
                              << " abs_mean=" << r.v6_err.attn_max_abs_mean
                              << " abs_max=" << r.v6_err.attn_max_abs_max << "\n";

                    std::cout << "  v6 fused attn mse=" << r.v6_fused_attn_mse
                              << " max_abs=" << r.v6_fused_attn_max_abs
                              << " (vs unscaled dense ref)\n";

                    std::cout << "  qjl     k_mse=" << r.qjl_err.k_mse
                              << " v_mse=" << r.qjl_err.v_mse
                              << " attn_mean=" << r.qjl_err.attn_mse_mean
                              << " attn_max=" << r.qjl_err.attn_mse_max
                              << " abs_mean=" << r.qjl_err.attn_max_abs_mean
                              << " abs_max=" << r.qjl_err.attn_max_abs_max << "\n";

                    std::cout << "  qjl fused attn mse=" << r.qjl_fused_attn_mse
                              << " max_abs=" << r.qjl_fused_attn_max_abs
                              << " (vs unscaled dense ref)\n";

                    std::cout << "  turbo fused attn mse=" << r.turbo_fused_attn_mse
                              << " max_abs=" << r.turbo_fused_attn_max_abs
                              << " (vs unscaled dense ref)\n";

                    std::cout << "  uniform pack/dequant mean/p95="
                              << r.uniform_pack_ms.mean << "/" << r.uniform_pack_ms.p95
                              << " , "
                              << r.uniform_dequant_ms.mean << "/" << r.uniform_dequant_ms.p95 << " ms\n";

                    std::cout << "  turbo   pack/dequant mean/p95="
                              << r.turbo_pack_ms.mean << "/" << r.turbo_pack_ms.p95
                              << " , "
                              << r.turbo_dequant_ms.mean << "/" << r.turbo_dequant_ms.p95 << " ms\n";

                    std::cout << "  v6      pack/dequant/fused mean/p95="
                              << r.v6_pack_ms.mean << "/" << r.v6_pack_ms.p95
                              << " , "
                              << r.v6_dequant_ms.mean << "/" << r.v6_dequant_ms.p95
                              << " , "
                              << r.v6_fused_attn_ms.mean << "/" << r.v6_fused_attn_ms.p95 << " ms\n";

                    std::cout << "  qjl     pack/dequant/fused mean/p95="
                              << r.qjl_pack_ms.mean << "/" << r.qjl_pack_ms.p95
                              << " , "
                              << r.qjl_dequant_ms.mean << "/" << r.qjl_dequant_ms.p95
                              << " , "
                              << r.qjl_fused_attn_ms.mean << "/" << r.qjl_fused_attn_ms.p95 << " ms\n";

                    {
                        double unfused = r.turbo_dequant_ms.mean + r.v6_fused_attn_ms.mean;
                        double speedup = unfused / r.turbo_fused_attn_ms.mean;
                        std::cout << "  turbo   fused mean/p95="
                                  << r.turbo_fused_attn_ms.mean << "/" << r.turbo_fused_attn_ms.p95
                                  << " ms  [speedup vs dequant+attn: "
                                  << std::fixed << std::setprecision(2) << speedup << "x]\n";
                    }

                    std::cout << "  polar   k_mse=" << r.polar_err.k_mse
                              << " v_mse=" << r.polar_err.v_mse
                              << " attn_mean=" << r.polar_err.attn_mse_mean
                              << " attn_max=" << r.polar_err.attn_mse_max << "\n";
                    std::cout << "  polar   fused mse=" << r.polar_fused_attn_mse
                              << " max_abs=" << r.polar_fused_attn_max_abs
                              << " (vs unscaled dense ref)\n";

                    {
                        double speedup = r.qjl_fused_attn_ms.mean / r.polar_fused_attn_ms.mean;
                        std::cout << "  polar   pack/dequant/fused mean/p95="
                                  << r.polar_pack_ms.mean << "/" << r.polar_pack_ms.p95
                                  << " , "
                                  << r.polar_dequant_ms.mean << "/" << r.polar_dequant_ms.p95
                                  << " , "
                                  << r.polar_fused_attn_ms.mean << "/" << r.polar_fused_attn_ms.p95
                                  << " ms  [fused vs qjl: "
                                  << std::fixed << std::setprecision(2) << speedup << "x]\n";
                    }

                    std::cout << "  dense   store/load mean/p95="
                              << r.dense_store_ms.mean << "/" << r.dense_store_ms.p95
                              << " , "
                              << r.dense_load_ms.mean << "/" << r.dense_load_ms.p95 << " ms\n";
                }
            }
        }

        write_csv("results_5way.csv", results);
        std::cout << "\nWrote results_5way.csv with " << results.size() << " rows.\n";
    } catch (const std::exception& e) {
        std::cerr << "Host exception: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
