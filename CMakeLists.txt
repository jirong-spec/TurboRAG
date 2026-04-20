cmake_minimum_required(VERSION 3.20)
project(turbo_paged_kv LANGUAGES CXX CUDA)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 80)

add_library(turboquant SHARED
    src/tq_turbo_prod.cpp
    src/tq_block_table.cpp
    src/tq_dense_layout.cpp
    src/tq_layout.cpp
    src/tq_turbo_mse_layout.cpp
    src/tq_qjl.cpp
    src/tq_allocator.cpp
    src/tq_attention_ref.cpp
    src/gaussian_codebook.cpp

    src/cuda/tq_pack_kernels.cu
    src/cuda/tq_dequant_kernels.cu
    src/cuda/tq_qjl_kernels.cu
    src/cuda/tq_dense_kernels.cu
    src/cuda/tq_turbo_mse_kernels.cu
    src/cuda/tq_attention_ref_kernel.cu
    src/cuda/tq_turbo_prod_kernels.cu
    src/tq_capi.cpp
)

target_include_directories(turboquant PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda
)

set_target_properties(turboquant PROPERTIES
    OUTPUT_NAME turboquant
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_link_libraries(turboquant PUBLIC
    CUDA::cudart
)