#pragma once

#include <cassert>
#include <vector>

// Pure planning of the per-device subgraphs of a meta-backend cgraph. A boundary is the end of a subgraph that is followed
// by an AllReduce. In node mode a boundary whose AllReduce qualifies does not split the subgraph: the backend executes the
// AllReduce as a node appended after the boundary node. Kept free of ggml types so that it can be unit-tested.

struct ggml_meta_ar_boundary {
    int  end;        // index of the last cgraph node of the subgraph ending here
    bool qualified;  // the AllReduce after `end` qualifies as a node on every device; ignored for the final boundary
};

struct ggml_meta_ar_plan {
    int              n_split = 0;  // subgraph count without node mode
    std::vector<int> offsets;      // first cgraph node of each subgraph
    std::vector<int> site_after;   // site s: its AllReduce node is appended after cgraph node site_after[s]
};

// `boundaries` must be ordered, with the last one ending at n_nodes - 1 (that one is followed by no AllReduce).
static inline ggml_meta_ar_plan ggml_meta_ar_plan_build(
        int n_nodes, const std::vector<ggml_meta_ar_boundary> & boundaries, bool node_mode) {
    (void) n_nodes; // only read by the asserts
    ggml_meta_ar_plan plan;
    plan.n_split = (int) boundaries.size();
    int start = 0;
    for (size_t k = 0; k < boundaries.size(); ++k) {
        const ggml_meta_ar_boundary & b = boundaries[k];
        assert(b.end >= start && b.end < n_nodes);
        const bool last = k + 1 == boundaries.size();
        assert(last == (b.end == n_nodes - 1));
        if (!last && node_mode && b.qualified) {
            plan.site_after.push_back(b.end);
            continue;
        }
        plan.offsets.push_back(start);
        start = b.end + 1;
    }
    assert(start == n_nodes);
    return plan;
}
