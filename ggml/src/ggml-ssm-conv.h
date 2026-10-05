#pragma once

#include "ggml-impl.h"

#include <algorithm>
#include <cstdint>
#include <vector>

struct ggml_ssm_conv_state_fusion {
    ggml_tensor * conv = nullptr;
    ggml_tensor * out = nullptr;
    ggml_tensor * cache = nullptr; // newest slot
    int64_t slots = 0;
    size_t slot_stride = 0;
    int node_count = 0;
};

static inline bool ggml_ssm_conv_overlaps(const ggml_tensor * a, const ggml_tensor * b) {
    const uintptr_t pa = reinterpret_cast<uintptr_t>(a->data);
    const uintptr_t pb = reinterpret_cast<uintptr_t>(b->data);
    return pa <= pb ? pb - pa < ggml_nbytes(a) : pa - pb < ggml_nbytes(b);
}

// Keep every cache write and reject any reader of the elided concat or snapshot views.
static inline bool ggml_match_ssm_conv_state_fusion(
        const ggml_cgraph * graph, int start, ggml_ssm_conv_state_fusion & match) {
    ggml_tensor * concat = graph->nodes[start];
    if (concat->op != GGML_OP_CONCAT || concat->type != GGML_TYPE_F32 ||
        ggml_get_op_params_i32(concat, 0) != 0 || !ggml_is_contiguous(concat) || concat->ne[3] != 1) {
        return false;
    }
    const ggml_tensor * state = concat->src[0];
    const ggml_tensor * tokens = concat->src[1];
    const int64_t history = state->ne[0];
    const int64_t channels = concat->ne[1];
    const int64_t seqs = concat->ne[2];
    if (history < 1 || state->type != GGML_TYPE_F32 || tokens->type != GGML_TYPE_F32 ||
        state->ne[1] != channels || state->ne[2] != seqs || state->ne[3] != 1 ||
        tokens->ne[1] != channels || tokens->ne[2] != seqs || tokens->ne[3] != 1 || tokens->ne[0] < 1) {
        return false;
    }

    std::vector<ggml_tensor *> copies;
    std::vector<ggml_op> ops = { GGML_OP_CONCAT };
    std::vector<int> outputs;
    int end = start + 1;
    for (; end < graph->n_nodes && end - start < 30; ++end) {
        ggml_tensor * node = graph->nodes[end];
        if (node->op == GGML_OP_SSM_CONV) {
            break;
        }
        if (node->op == GGML_OP_VIEW) {
            if (node->view_src != concat) {
                outputs.push_back(end); // cache destination views must remain visible
            }
        } else if (node->op == GGML_OP_CPY) {
            const ggml_tensor * src = node->src[0];
            const ggml_tensor * dst = node->src[1];
            if (src->op != GGML_OP_VIEW || src->view_src != concat || src->type != GGML_TYPE_F32 ||
                src->ne[0] != history || src->ne[1] != channels || src->ne[2] != seqs || src->ne[3] != 1 ||
                src->nb[0] != sizeof(float) || src->nb[1] != concat->nb[1] || src->nb[2] != concat->nb[2] ||
                dst->op != GGML_OP_VIEW || dst->type != GGML_TYPE_F32 ||
                dst->ne[0] != history * channels || dst->ne[1] != seqs || dst->ne[2] != 1 || dst->ne[3] != 1 ||
                dst->nb[0] != sizeof(float) || dst->nb[1] < (size_t) (history * channels) * sizeof(float)) {
                return false;
            }
            copies.push_back(node);
            outputs.push_back(end);
        } else {
            return false;
        }
        ops.push_back(node->op);
    }
    if (end >= graph->n_nodes || end - start >= 30 || copies.empty()) {
        return false;
    }
    ggml_tensor * conv = graph->nodes[end];
    const ggml_tensor * weights = conv->src[1];
    if (conv->op != GGML_OP_SSM_CONV || conv->src[0] != concat || conv->type != GGML_TYPE_F32 ||
        weights->type != GGML_TYPE_F32 || weights->ne[0] != history + 1 || weights->nb[0] != sizeof(float) ||
        weights->ne[1] != channels || !ggml_is_contiguous(conv)) {
        return false;
    }
    ops.push_back(GGML_OP_SSM_CONV);
    ggml_tensor * out = conv;
    if (end + 1 < graph->n_nodes && graph->nodes[end + 1]->op == GGML_OP_UNARY &&
        graph->nodes[end + 1]->src[0] == conv && ggml_get_unary_op(graph->nodes[end + 1]) == GGML_UNARY_OP_SILU) {
        out = graph->nodes[++end];
        ops.push_back(GGML_OP_UNARY);
    }
    outputs.push_back(end);
    if (!ggml_can_fuse_subgraph(graph, start, ops.size(), ops.data(), outputs.data(), outputs.size())) {
        return false;
    }

    const int64_t slots = copies.size();
    ggml_tensor * newest = copies.back()->src[1];
    const size_t oldest_off = copies.front()->src[1]->view_offs;
    if (oldest_off < newest->view_offs || (slots > 1 && (oldest_off - newest->view_offs) % (slots - 1) != 0)) {
        return false;
    }
    const size_t slot_stride = slots > 1 ? (oldest_off - newest->view_offs) / (slots - 1) : 0;
    if (slots > 1 && slot_stride < ggml_nbytes(newest)) {
        return false;
    }
    for (int64_t i = 0; i < slots; ++i) {
        const ggml_tensor * src = copies[i]->src[0];
        const ggml_tensor * dst = copies[i]->src[1];
        const int64_t slot = slots - 1 - i;
        if (src->view_offs != (size_t) std::max<int64_t>(0, tokens->ne[0] - slot) * sizeof(float) ||
            dst->view_src != newest->view_src || dst->nb[1] != newest->nb[1] ||
            dst->view_offs != newest->view_offs + slot * slot_stride) {
            return false;
        }
        for (const ggml_tensor * read : {state, tokens, weights}) {
            if (!read->data || !dst->data || ggml_ssm_conv_overlaps(dst, read)) {
                return false;
            }
        }
        if (ggml_ssm_conv_overlaps(dst, out)) {
            return false;
        }
    }
    for (const ggml_tensor * read : {state, tokens, weights}) {
        if (!out->data || ggml_ssm_conv_overlaps(out, read)) {
            return false;
        }
    }
    match = { conv, out, newest, slots, slot_stride, end - start + 1 };
    return true;
}
