#pragma once
#include <cstdint>
#include <vector>
#include "tq_layout.h"

class TQAllocator {
public:
    TQAllocator(const TQConfig& cfg, const TQPageLayout& layout, int num_pages);
    ~TQAllocator();

    block_id_t alloc_block();
    void free_block(block_id_t id);

    uint8_t* device_page_pool() const { return page_pool_; }
    int num_pages() const { return num_pages_; }
    const TQPageLayout& layout() const { return layout_; }

private:
    TQConfig cfg_;
    TQPageLayout layout_;
    int num_pages_;
    uint8_t* page_pool_ = nullptr;
    std::vector<block_id_t> free_list_;
};
