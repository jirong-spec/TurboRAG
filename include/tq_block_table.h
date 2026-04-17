#pragma once
#include <unordered_map>
#include <vector>
#include "tq_allocator.h"

struct SequenceState {
    std::vector<block_id_t> blocks;
    int num_tokens = 0;
};

class TQBlockTable {
public:
    TQBlockTable(TQAllocator& alloc, const TQConfig& cfg);
    void ensure_slots(seq_id_t seq_id, int append_tokens);
    std::vector<int32_t> build_slot_map(seq_id_t seq_id, int append_tokens);
    const SequenceState& state(seq_id_t seq_id) const;

private:
    TQAllocator& alloc_;
    TQConfig cfg_;
    std::unordered_map<seq_id_t, SequenceState> seqs_;
};
