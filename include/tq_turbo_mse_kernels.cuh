#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "tq_config.h"
#include "tq_turbo_mse_layout.h"

void launch_tq_turbo_mse_pack_kv(
    const half* key,
    const half* value,
    const int32_t* slot_mapping,
    uint8_t* page_pool,
    const TQTurbomsePageLayout& layout,
    const TQConfig& cfg,
    int num_tokens,
    cudaStream_t stream);

void launch_tq_turbo_mse_dequant_kv(
    const uint8_t* page_pool,
    const int32_t* slot_mapping,
    half* out_key,
    half* out_value,
    const TQTurbomsePageLayout& layout,
    const TQConfig& cfg,
    int num_tokens,
    cudaStream_t stream);

// Fused attention: online softmax → weighted V sum → inverse rotation.
// Reads directly from compressed page pool; no intermediate FP16 KV buffer.
// Logit convention: <q, k>  (no 1/sqrt(D)), matching V6 / QJL convention.
void launch_tq_turbo_mse_fused_attention_output(
    const half* query,
    const uint8_t* page_pool,
    const int32_t* slot_mapping,
    half* output,
    const TQTurbomsePageLayout& layout,
    const TQConfig& cfg,
    int num_queries,
    int num_kv_tokens,
    cudaStream_t stream);
