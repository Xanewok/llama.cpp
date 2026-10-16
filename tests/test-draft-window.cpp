#include "arg.h"
#include "common.h"
#include "speculative.h"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

static void require(bool ok) {
    if (!ok) {
        throw std::runtime_error("draft window parameter check failed");
    }
}

static bool parse(common_params & params, std::vector<const char *> args) {
    return common_params_parse(args.size(), const_cast<char **>(args.data()), params, LLAMA_EXAMPLE_SERVER);
}

int main() {
    try {
        common_params base;
        require(base.speculative.draft.swa_window == 0);
        require(base.speculative.draft.swa_sinks == 16);
        require(common_context_params_to_llama(base).swa_window == 0);
        require(common_context_params_to_llama(common_base_params_to_speculative(base)).swa_window == 0);
        base.n_parallel = 2;
        base.n_batch = base.n_ubatch = 1;
        base.speculative.draft.n_max = 5;
        base.speculative.draft.swa_window = 8192;
        const auto draft = common_base_params_to_speculative(base);
        require(draft.n_batch == 10 && draft.n_ubatch == 10);
        require(common_context_params_to_llama(base).swa_window == 0);
        require(common_context_params_to_llama(draft).swa_window == 8192);
        require(common_context_params_to_llama(draft).swa_sinks == 16);
        base.speculative.draft.swa_window = 0;
        const auto off = common_base_params_to_speculative(base);
        require(off.n_batch == 1 && off.n_ubatch == 1);

        for (const char * option : {"--spec-draft-swa", "--spec-draft-swa-sinks"}) {
            common_params params;
            require(!parse(params, {"test", option, "-1"}));
        }
        common_params params;
        require(parse(params, {"test", "--spec-draft-swa", "8192", "--spec-draft-swa-sinks", "3"}));
        require(params.speculative.draft.swa_window == 8192 && params.speculative.draft.swa_sinks == 3);
        require(params.swa_window == 0);
#ifndef _WIN32
        setenv("LLAMA_ARG_SPEC_DRAFT_SWA", "4096", 1);
        setenv("LLAMA_ARG_SPEC_DRAFT_SWA_SINKS", "16", 1);
        require(parse(params, {"test"}));
        require(params.speculative.draft.swa_window == 4096 && params.speculative.draft.swa_sinks == 16);
        require(parse(params, {"test", "--spec-draft-swa", "0"}));
        require(params.speculative.draft.swa_window == 0);
        unsetenv("LLAMA_ARG_SPEC_DRAFT_SWA");
        unsetenv("LLAMA_ARG_SPEC_DRAFT_SWA_SINKS");
#endif
        printf("test-draft-window: CLI, environment, draft-only propagation and rollback slack OK\n");
        return 0;
    } catch (const std::exception & e) {
        fprintf(stderr, "test-draft-window: %s\n", e.what());
        return 1;
    }
}
