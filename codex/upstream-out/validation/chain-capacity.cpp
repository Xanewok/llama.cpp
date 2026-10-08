#include "common.h"
#include "speculative.h"

#include <cstdio>
#include <stdexcept>

static void check(common_params params, bool chain) {
    const auto draft = common_base_params_to_speculative(params);
    if (draft.n_outputs_max_per_seq != (chain ? 3 : 1) ||
            draft.n_outputs_max != (chain ? 3 : params.n_parallel)) {
        throw std::runtime_error("draft output capacity changed outside the chain contract");
    }
}

int main() {
    common_params p;
    p.n_parallel = 1;
    p.speculative.types = {COMMON_SPECULATIVE_TYPE_DRAFT_MTP};
    p.speculative.draft.n_max = 3;
    p.speculative.draft.probabilistic = true;
    p.speculative.draft.backend_sampling = true;
    p.speculative.draft.p_min = 0.0f;
    check(p, true);
    auto q = p;
    q.n_parallel = 2;
    check(q, false);
    q = p;
    q.speculative.types = {COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE};
    check(q, false);
    q = p;
    q.speculative.draft.probabilistic = false;
    check(q, false);
    q = p;
    q.speculative.draft.p_min = 0.1f;
    check(q, false);
    q = p;
    q.speculative.draft.n_max = 1;
    check(q, false);
    q = p;
    q.speculative.draft.n_max = 1024;
    check(q, false);
    std::puts("M: chain capacity contract PASS (7 configurations)");
}
