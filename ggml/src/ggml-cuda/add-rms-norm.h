#pragma once

#include "ggml.h"
#include "ggml-impl.h"

#include <climits>
#include <cstdint>

// Rows are independent, so this bounds only where the fusion has been measured, not its exactness.
static constexpr int64_t GGML_CUDA_ADD_RMS_NORM_MAX_ROWS = 16;

// Nodes i..i+2 are ADD, RMS_NORM, MUL and the fused kernel may stand in for them: it writes the ADD result and the
// MUL result and never materializes the RMS_NORM result, so every buffer the unfused chain would write must be
// private to it or exactly aliased as the kernel's per-row, per-thread access order allows.
static inline bool ggml_cuda_can_fuse_add_rms_norm(const ggml_cgraph * cgraph, int i) {
    if (!ggml_can_fuse_subgraph(cgraph, i, {GGML_OP_ADD, GGML_OP_RMS_NORM, GGML_OP_MUL}, {i, i + 2})) {
        return false;
    }
    const ggml_tensor * add = cgraph->nodes[i];
    const ggml_tensor * norm = cgraph->nodes[i + 1];
    const ggml_tensor * mul = cgraph->nodes[i + 2];
    const ggml_tensor * x = add->src[0];
    const ggml_tensor * y = add->src[1];
    const ggml_tensor * weight = mul->src[1];
    bool valid = norm->src[0] == add && mul->src[0] == norm &&
        ggml_are_same_shape(x, y) && ggml_are_same_shape(add, norm) && ggml_are_same_shape(add, mul) &&
        ggml_is_matrix(add) && add->ne[0] <= INT_MAX - 1024 && add->ne[1] <= GGML_CUDA_ADD_RMS_NORM_MAX_ROWS &&
        ggml_is_vector(weight) && weight->ne[0] == add->ne[0];
    const ggml_tensor * tensors[] = {x, y, weight, add, norm, mul};
    for (const ggml_tensor * t : tensors) {
        valid = valid && t->type == GGML_TYPE_F32 && ggml_is_contiguous(t);
    }
    const auto overlaps = [](const ggml_tensor * a, const ggml_tensor * b) {
        const uintptr_t ap = (uintptr_t) a->data;
        const uintptr_t bp = (uintptr_t) b->data;
        return ap <= bp ? bp - ap < ggml_nbytes(a) : ap - bp < ggml_nbytes(b);
    };
    // Each row owns its input elements; an exact in-place residual is safe.
    for (const ggml_tensor * t : {x, y}) {
        valid = valid && (!overlaps(add, t) || add->data == t->data) && (!overlaps(mul, t) || mul->data == t->data);
    }
    valid = valid && !overlaps(add, weight) && !overlaps(mul, weight) && !overlaps(mul, add);
    // The unfused RMS_NORM writes norm's buffer and the fused kernel does not, so no other operand may live there.
    for (const ggml_tensor * t : {x, y, weight, add}) {
        valid = valid && !overlaps(norm, t);
    }
    return valid && (!overlaps(norm, mul) || norm->data == mul->data);
}
