#pragma once

#include <cstddef>
#include <cstdint>

#define FATTN_KQ_STRIDE 256

#if defined(__CUDACC__) || defined(__HIPCC__) || defined(__MUSACC__)
#define FATTN_HOST_DEVICE __host__ __device__
#else
#define FATTN_HOST_DEVICE
#endif

static FATTN_HOST_DEVICE constexpr bool flash_attn_decode_mask_packed(uintptr_t mask, int nk, size_t nbq, int warp_size) {
    const size_t nbytes = FATTN_KQ_STRIDE / (2*warp_size) * sizeof(uint32_t);
    return nk % FATTN_KQ_STRIDE == 0 && nbq % nbytes == 0 && mask % nbytes == 0;
}

static FATTN_HOST_DEVICE constexpr size_t flash_attn_decode_mask_offset(int q, size_t nbq, int start, int lane, int warp_size) {
    const int nwords = FATTN_KQ_STRIDE / (2*warp_size);
    return size_t(q)*nbq + size_t(start + lane*2*nwords)*2;
}

#undef FATTN_HOST_DEVICE
