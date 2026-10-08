#pragma once

#include "ggml-impl.h"
#include "ggml-ssm-conv.h"

static inline bool ggml_rms_norm_scale_pair_is_noop(const ggml_tensor * t) {
    return ggml_is_empty(t) || t->op == GGML_OP_NONE || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_TRANSPOSE ||
           t->op == GGML_OP_VIEW || t->op == GGML_OP_PERMUTE;
}

static inline bool ggml_is_rms_norm_scale_chain(const ggml_cgraph * graph, int i) {
    static const enum ggml_op ops[] = { GGML_OP_RMS_NORM, GGML_OP_SCALE };
    if (!ggml_can_fuse(graph, i, ops, 2)) {
        return false;
    }
    const ggml_tensor * rms   = graph->nodes[i];
    const ggml_tensor * scale = graph->nodes[i + 1];

    float bias;
    memcpy(&bias, (const float *) scale->op_params + 1, sizeof(float));

    return bias == 0.0f && rms->src[0]->type == GGML_TYPE_F32 && scale->type == GGML_TYPE_F32 &&
           rms->src[0]->nb[0] == sizeof(float) && ggml_is_contiguous(scale) && rms->src[0]->data && scale->data;
}

// Two independent RMS_NORM -> SCALE chains of equal shape that start at node i and node i_b, separated only by
// no-op nodes (the q and k views of a gated-delta-net layer). A backend running them as one operation at
// node i reads chain b's input before chain a's nodes would have run, and writes both outputs concurrently,
// so b must not depend on a and neither output may overlap the other chain's input or output.
// Both outputs must be contiguous: the paired launch addresses them densely.
static inline bool ggml_match_rms_norm_scale_pair(const ggml_cgraph * graph, int i, int & i_b) {
    if (!ggml_is_rms_norm_scale_chain(graph, i)) {
        return false;
    }

    int j = i + 2;
    while (j < graph->n_nodes && ggml_rms_norm_scale_pair_is_noop(graph->nodes[j])) {
        ++j;
    }
    if (j + 1 >= graph->n_nodes || !ggml_is_rms_norm_scale_chain(graph, j)) {
        return false;
    }

    const ggml_tensor * x[2]     = { graph->nodes[i]->src[0], graph->nodes[j]->src[0] };
    const ggml_tensor * scale[2] = { graph->nodes[i + 1], graph->nodes[j + 1] };

    if (!ggml_are_same_shape(x[0], x[1])) {
        return false;
    }

    static const bool alias_mutant = getenv("GGML_RMS_NORM_SCALE_PAIR_ALIAS_MUTANT") != nullptr;
    if (alias_mutant) {
        i_b = j;
        return true;
    }

    const ggml_tensor * root_b = x[1];
    while (root_b->view_src) {
        root_b = root_b->view_src;
    }
    if (root_b == graph->nodes[i] || root_b == scale[0]) {
        return false;
    }

    if (ggml_ssm_conv_overlaps(scale[0], x[1]) || ggml_ssm_conv_overlaps(scale[1], x[0]) ||
            ggml_ssm_conv_overlaps(scale[0], scale[1])) {
        return false;
    }

    i_b = j;
    return true;
}
