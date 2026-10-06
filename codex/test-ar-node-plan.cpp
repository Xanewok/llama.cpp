// Unit test of ggml_meta_ar_plan_build (ggml/src/ggml-meta-ar-plan.h): header-only, no ggml link.
//   g++ -std=c++17 -O1 -I../ggml/src test-ar-node-plan.cpp -o test-ar-node-plan && ./test-ar-node-plan
// With AR_PLAN_WRONG_EXPECTATION=1 one expectation is deliberately wrong and the binary must fail (negative control).
#include "ggml-meta-ar-plan.h"

#include <cstdio>
#include <cstdlib>

static int failures = 0;

static void check(const char * name, bool ok) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", name);
    failures += !ok;
}

int main() {
    const bool wrong = std::getenv("AR_PLAN_WRONG_EXPECTATION") != nullptr;
    using B = ggml_meta_ar_boundary;

    // 3 boundaries (nodes 9, 19, 29 end subgraphs) + the final one; kinds: qualified PARTIAL, gather, non-qualified.
    {
        const std::vector<B> b = { {9, true}, {19, false /* gather */}, {29, false /* too large */}, {39, false} };
        const auto split = ggml_meta_ar_plan_build(40, b, false);
        check("split: 4 subgraphs", split.n_split == 4 && split.offsets == std::vector<int>({0, 10, 20, 30}) && split.site_after.empty());
        const auto node = ggml_meta_ar_plan_build(40, b, true);
        check("node: qualified boundary fused, gather and large stay splits",
              node.n_split == 4 && node.offsets == std::vector<int>({0, 20, 30}) && node.site_after == std::vector<int>({9}));
        check("node: subgraph count 4 -> 3", node.offsets.size() == (wrong ? 4u : 3u));
    }
    // All qualified: one subgraph, one site per non-final boundary, final boundary never gets a node.
    {
        const std::vector<B> b = { {4, true}, {9, true}, {14, true}, {19, true} };
        const auto node = ggml_meta_ar_plan_build(20, b, true);
        check("node: all qualified -> 1 subgraph, 3 sites", node.offsets == std::vector<int>({0}) && node.site_after == std::vector<int>({4, 9, 14}) && node.n_split == 4);
        const auto split = ggml_meta_ar_plan_build(20, b, false);
        check("split: qualified ignored without node mode", split.offsets == std::vector<int>({0, 5, 10, 15}) && split.site_after.empty());
    }
    // Qualified flag of the final boundary is ignored.
    {
        const std::vector<B> b = { {7, true}, {15, true} };
        const auto node = ggml_meta_ar_plan_build(16, b, true);
        check("node: final boundary flag ignored", node.offsets == std::vector<int>({0}) && node.site_after == std::vector<int>({7}));
    }
    // Single subgraph (no AllReduce at all).
    {
        const std::vector<B> b = { {11, true} };
        const auto node = ggml_meta_ar_plan_build(12, b, true);
        check("node: single boundary -> 1 subgraph, 0 sites", node.offsets == std::vector<int>({0}) && node.site_after.empty() && node.n_split == 1);
    }
    // Adjacent boundaries (single-node subgraphs) mixed.
    {
        const std::vector<B> b = { {0, false}, {1, true}, {2, true}, {3, false}, {4, true}, {5, false} };
        const auto node = ggml_meta_ar_plan_build(6, b, true);
        check("node: adjacent boundaries", node.offsets == std::vector<int>({0, 1, 4}) && node.site_after == std::vector<int>({1, 2, 4}) && node.n_split == 6);
    }
    std::printf("%s (%d failures)\n", failures ? "FAILED" : "ALL PASSED", failures);
    return failures ? 1 : 0;
}
