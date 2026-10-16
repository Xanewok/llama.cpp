#pragma once

#include "llama.h"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

struct server_output_record {
    int id_slot;
    int id_task;
    std::vector<llama_token> tokens;
};

inline int32_t server_output_remaining(int32_t maximum, int32_t emitted, int32_t pending) {
    return maximum == -1 ? -1 : maximum - emitted - pending;
}

inline bool server_output_can_defer(bool eligible, bool terminal, int64_t remaining, size_t produced) {
    return eligible && !terminal && produced > 0 &&
        (remaining == -1 || (remaining > 0 && uint64_t(remaining) > produced));
}

class server_output_queue {
    std::vector<server_output_record> pending;

public:
    bool empty() const { return pending.empty(); }
    size_t size() const { return pending.size(); }

    bool submit(std::vector<server_output_record> round) {
        if (!pending.empty()) {
            return false;
        }
        pending = std::move(round);
        return true;
    }

    std::vector<server_output_record> take() {
        std::vector<server_output_record> round;
        round.swap(pending);
        return round;
    }
};
