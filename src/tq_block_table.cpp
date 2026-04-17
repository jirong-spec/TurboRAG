#include "tq_block_table.h"
#include <stdexcept>

TQBlockTable::TQBlockTable(TQAllocator& alloc, const TQConfig& cfg)
    : alloc_(alloc), cfg_(cfg) {}

void TQBlockTable::ensure_slots(seq_id_t seq_id, int append_tokens) {
    auto& s = seqs_[seq_id];
    int old_tokens = s.num_tokens;
    int new_tokens = old_tokens + append_tokens;

    int old_blocks = (old_tokens + cfg_.block_size - 1) / cfg_.block_size;
    int new_blocks = (new_tokens + cfg_.block_size - 1) / cfg_.block_size;

    for (int i = old_blocks; i < new_blocks; ++i) {
        block_id_t b = alloc_.alloc_block();
        if (b < 0) {
            throw std::runtime_error("out of blocks");
        }
        s.blocks.push_back(b);
    }
}

std::vector<int32_t> TQBlockTable::build_slot_map(seq_id_t seq_id, int append_tokens) {
    ensure_slots(seq_id, append_tokens);
    auto& s = seqs_[seq_id];

    std::vector<int32_t> slots(append_tokens);
    for (int i = 0; i < append_tokens; ++i) {
        int token_idx = s.num_tokens + i;
        int logical_block = token_idx / cfg_.block_size;
        int token_in_block = token_idx % cfg_.block_size;
        int physical_block = s.blocks.at(logical_block);
        slots[i] = physical_block * cfg_.block_size + token_in_block;
    }

    s.num_tokens += append_tokens;
    return slots;
}

const SequenceState& TQBlockTable::state(seq_id_t seq_id) const {
    return seqs_.at(seq_id);
}
