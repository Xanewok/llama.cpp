#pragma once

#include <cstdint>
#include <cmath>

#ifdef __CUDACC__
#define GGML_SYNCFREE_HD __host__ __device__
#else
#define GGML_SYNCFREE_HD
#endif

// Each sequence has one contiguous packet and result.
// packet: [base_pos, greedy, k draft ids, k*10 q ids, k*10 q probabilities, 2*k uniforms].
// result: [n_acc, bonus, next_pos, k+1 accepted ids, n_pos*k draft positions].
GGML_SYNCFREE_HD inline void ggml_syncfree_accept(
        const float * p, const int32_t * ids, const int32_t * sampled,
        const float * packet, int32_t * out, int k, int n_pos, int nc) {
    const float * draft = packet + 2;
    const float * q_ids = draft + k;
    const float * q     = q_ids + 10*k;
    const float * u     = q + 10*k;
    int a = 0;
    int32_t bonus = sampled[k];
    for (; a < k; ++a) {
        const int32_t x = (int32_t) draft[a];
        if (packet[1] != 0.0f) {
            if (sampled[a] == x) {
                continue;
            }
            bonus = sampled[a];
            break;
        }
        float px = 0.0f;
        float qx = 0.0f;
        for (int j = 0; j < nc; ++j) {
            if (ids[a*nc + j] == x) {
                px = p[a*nc + j];
            }
        }
        for (int j = 0; j < 10; ++j) {
            if ((int32_t) q_ids[a*10 + j] == x) {
                qx = q[a*10 + j];
            }
        }
        if (qx > 0.0f && (px >= qx || u[2*a] < px/qx)) {
            continue;
        }
        float mass = 0.0f;
        for (int j = 0; j < nc; ++j) {
            float qj = 0.0f;
            for (int z = 0; z < 10; ++z) {
                if ((int32_t) q_ids[a*10 + z] == ids[a*nc + j]) {
                    qj = q[a*10 + z];
                }
            }
            const float r = p[a*nc + j] - qj;
            mass += r > 0.0f ? r : 0.0f;
        }
        bonus = sampled[a];
        if (mass > 0.0f) {
            float draw = u[2*a + 1]*mass;
            for (int j = 0; j < nc; ++j) {
                float qj = 0.0f;
                for (int z = 0; z < 10; ++z) {
                    if ((int32_t) q_ids[a*10 + z] == ids[a*nc + j]) {
                        qj = q[a*10 + z];
                    }
                }
                const float r = p[a*nc + j] - qj;
                if (r > 0.0f) {
                    bonus = ids[a*nc + j];
                    draw -= r;
                    if (draw <= 0.0f) {
                        break;
                    }
                }
            }
        }
        break;
    }
    out[0] = a;
    out[1] = bonus;
    out[2] = (int32_t) packet[0] + a + 1;
    for (int j = 0; j <= k; ++j) {
        out[3 + j] = j < a ? (int32_t) draft[j] : bonus;
    }
    for (int axis = 0; axis < n_pos; ++axis) {
        for (int s = 0; s < k; ++s) {
            out[4 + k + axis*k + s] = axis < 3 ? out[2] + s : 0;
        }
    }
}

GGML_SYNCFREE_HD inline void ggml_syncfree_accept_batch(
        const float * p, const int32_t * ids, const int32_t * sampled,
        const float * packet, int32_t * out, int k, int n_pos, int nc, int n_seq) {
    for (int s = 0; s < n_seq; ++s) {
        ggml_syncfree_accept(p ? p + s*nc*(k + 1) : nullptr, ids ? ids + s*nc*(k + 1) : nullptr,
            sampled + s*(k + 1), packet + s*(2 + 23*k), out + s*(4 + k + n_pos*k), k, n_pos, nc);
    }
}

GGML_SYNCFREE_HD inline float ggml_syncfree_mask_value(
        float mask, int32_t pos, const int32_t * accept, const float * packet, int k) {
    return pos >= accept[2] && pos < (int32_t) packet[0] + k + 1 ? -INFINITY : mask;
}

#undef GGML_SYNCFREE_HD
