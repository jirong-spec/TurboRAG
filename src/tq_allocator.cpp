#include "tq_allocator.h"
#include <cuda_runtime.h>
#include <stdexcept>

TQAllocator::TQAllocator(const TQConfig& cfg, const TQPageLayout& layout, int num_pages)
    : cfg_(cfg), layout_(layout), num_pages_(num_pages) {
    size_t total_bytes = static_cast<size_t>(num_pages_) * layout_.page_size_bytes;
    if (cudaMalloc(&page_pool_, total_bytes) != cudaSuccess) {
        throw std::runtime_error("cudaMalloc page_pool failed");
    }
    for (int i = num_pages_ - 1; i >= 0; --i) {
        free_list_.push_back(i);
    }
}

TQAllocator::~TQAllocator() {
    if (page_pool_) {
        cudaFree(page_pool_);
    }
}

block_id_t TQAllocator::alloc_block() {
    if (free_list_.empty()) return -1;
    block_id_t id = free_list_.back();
    free_list_.pop_back();
    return id;
}

void TQAllocator::free_block(block_id_t id) {
    free_list_.push_back(id);
}
