#pragma once
#include <cstdint>

using block_id_t = int32_t;
using seq_id_t = int32_t;
using slot_t = int32_t;

enum class TQScaleType : int {
    FP16 = 0,
    FP32 = 1,
};

enum class TQQuantMode : int {
    MSE_INT4 = 0,
};
