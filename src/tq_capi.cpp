#include "tq_turbo_prod.cuh"
#include "tq_turbo_mse_layout.h"
#include "tq_turbo_mse_kernels.cuh"
#include "tq_cuda_check.h"
#include <cstdio>
#include <cstring>
#include <stdexcept>

// Return codes for every tq_* function
//   TQ_OK       =  0  — success
//   TQ_ERR_ARG  = -1  — null pointer or invalid argument
//   TQ_ERR_CUDA = -2  — CUDA runtime / kernel-launch error; call tq_last_error()

static thread_local char s_tq_last_error[512] = "";

// Returns the CUDA / kernel error string from the last failed call, or "".
extern "C" const char* tq_last_error() {
    return s_tq_last_error;
}

// Wraps any kernel launch: catches std::exception thrown by launch helpers
// (pre-launch validation or TQ_CHECK_LAUNCH), stores the message in the
// thread-local buffer, and returns TQ_ERR_CUDA = -2.
// Prevents C++ exceptions from crossing the extern "C" ABI boundary (UB).
#define TQ_KERNEL_CALL(expr)                                            \
    do {                                                                \
        try { (expr); }                                                 \
        catch (const std::exception& _e) {                              \
            std::snprintf(s_tq_last_error, sizeof(s_tq_last_error),    \
                          "%.511s", _e.what());                         \
            return -2;                                                  \
        }                                                               \
    } while (0)

extern "C" int tq_make_default_config(TQConfig* out_cfg) {
    if (!out_cfg) return -1;
    *out_cfg = TQConfig{};
    return 0;
}

extern "C" int tq_make_turbo_prod_layout(
    const TQConfig* cfg,
    TQTurboProdPageLayout* out_layout) {
    if (!cfg || !out_layout) return -1;
    *out_layout = make_tq_turbo_prod_page_layout(*cfg);
    return 0;
}

extern "C" int tq_launch_turbo_prod_pack_kv(
    const half* key,
    const half* value,
    const int32_t* slot_mapping,
    uint8_t* page_pool,
    const TQTurboProdPageLayout* layout,
    const TQConfig* cfg,
    int num_tokens,
    cudaStream_t stream) {
    if (!key || !value || !slot_mapping || !page_pool || !layout || !cfg) return -1;
    TQ_KERNEL_CALL(launch_tq_turbo_prod_pack_kv(
        key, value, slot_mapping, page_pool,
        *layout, *cfg, num_tokens, stream,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr));
    return 0;
}

extern "C" int tq_launch_turbo_prod_dequant_kv(
    const uint8_t* page_pool,
    const int32_t* slot_mapping,
    half* out_key,
    half* out_value,
    const TQTurboProdPageLayout* layout,
    const TQConfig* cfg,
    int num_tokens,
    cudaStream_t stream) {
    if (!page_pool || !slot_mapping || !out_key || !out_value || !layout || !cfg) return -1;
    TQ_KERNEL_CALL(launch_tq_turbo_prod_dequant_kv(
        page_pool, slot_mapping, out_key, out_value,
        *layout, *cfg, num_tokens, stream));
    return 0;
}

extern "C" int tq_launch_turbo_prod_fused_attention_logits(
    const half* query,
    const uint8_t* page_pool,
    const int32_t* slot_mapping,
    float* logits,
    const TQTurboProdPageLayout* layout,
    const TQConfig* cfg,
    int num_queries,
    int num_kv_tokens,
    cudaStream_t stream) {
    if (!query || !page_pool || !slot_mapping || !logits || !layout || !cfg) return -1;
    TQ_KERNEL_CALL(launch_tq_turbo_prod_fused_attention_logits(
        query, page_pool, slot_mapping, logits,
        *layout, *cfg, num_queries, num_kv_tokens, stream));
    return 0;
}

extern "C" int tq_launch_turbo_prod_fused_attention_output(
    const half* query,
    const uint8_t* page_pool,
    const int32_t* slot_mapping,
    half* output,
    const TQTurboProdPageLayout* layout,
    const TQConfig* cfg,
    int num_queries,
    int num_kv_tokens,
    cudaStream_t stream) {
    if (!query || !page_pool || !slot_mapping || !output || !layout || !cfg) return -1;
    TQ_KERNEL_CALL(launch_tq_turbo_prod_fused_attention_output(
        query, page_pool, slot_mapping, output,
        *layout, *cfg, num_queries, num_kv_tokens, stream));
    return 0;
}

// ── turbo_mse ──────────────────────────────────────────────────────────── //

extern "C" int tq_make_turbo_mse_layout(
    const TQConfig* cfg,
    TQTurbomsePageLayout* out_layout) {
    if (!cfg || !out_layout) return -1;
    *out_layout = make_tq_turbo_mse_page_layout(*cfg);
    return 0;
}

extern "C" int tq_launch_turbo_mse_pack_kv(
    const half* key,
    const half* value,
    const int32_t* slot_mapping,
    uint8_t* page_pool,
    const TQTurbomsePageLayout* layout,
    const TQConfig* cfg,
    int num_tokens,
    cudaStream_t stream) {
    if (!key || !value || !slot_mapping || !page_pool || !layout || !cfg) return -1;
    TQ_KERNEL_CALL(launch_tq_turbo_mse_pack_kv(
        key, value, slot_mapping, page_pool,
        *layout, *cfg, num_tokens, stream));
    return 0;
}

extern "C" int tq_launch_turbo_mse_dequant_kv(
    const uint8_t* page_pool,
    const int32_t* slot_mapping,
    half* out_key,
    half* out_value,
    const TQTurbomsePageLayout* layout,
    const TQConfig* cfg,
    int num_tokens,
    cudaStream_t stream) {
    if (!page_pool || !slot_mapping || !out_key || !out_value || !layout || !cfg) return -1;
    TQ_KERNEL_CALL(launch_tq_turbo_mse_dequant_kv(
        page_pool, slot_mapping, out_key, out_value,
        *layout, *cfg, num_tokens, stream));
    return 0;
}

extern "C" int tq_launch_turbo_mse_fused_attention_output(
    const half* query,
    const uint8_t* page_pool,
    const int32_t* slot_mapping,
    half* output,
    const TQTurbomsePageLayout* layout,
    const TQConfig* cfg,
    int num_queries,
    int num_kv_tokens,
    cudaStream_t stream) {
    if (!query || !page_pool || !slot_mapping || !output || !layout || !cfg) return -1;
    TQ_KERNEL_CALL(launch_tq_turbo_mse_fused_attention_output(
        query, page_pool, slot_mapping, output,
        *layout, *cfg, num_queries, num_kv_tokens, stream));
    return 0;
}

