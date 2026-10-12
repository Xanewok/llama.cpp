#pragma once
#include "llama.h"
#include <vector>

// 98c695111 removed the common helpers; retain their legacy llama_batch layout.
static void common_batch_clear(llama_batch & batch) { batch.n_tokens = 0; }
static void common_batch_add(llama_batch & batch, llama_token id, llama_pos pos,
                             const std::vector<llama_seq_id> & seqs, bool logits) {
    const int i = batch.n_tokens++;
    batch.token[i] = id;
    batch.pos[i] = pos;
    batch.n_seq_id[i] = int(seqs.size());
    for (size_t j = 0; j < seqs.size(); ++j) batch.seq_id[i][j] = seqs[j];
    batch.logits[i] = logits;
}
