#pragma once

#include "common.cuh"
#include "ggml-backend-impl.h"

#include <cstddef>

// Opaque pipeline context -- owns all pinned buffers, streams, and events.
struct ggml_cuda_ar_pipeline;
struct ggml_cuda_ar_pending;

// Allocate a pipeline for n_devices GPUs.
// devices[] holds the GPU device IDs in rank order.
// Returns nullptr on allocation failure.
ggml_cuda_ar_pipeline * ggml_cuda_ar_pipeline_init(
    const int * devices, size_t n_devices);

// Release all resources owned by the pipeline.
void ggml_cuda_ar_pipeline_free(ggml_cuda_ar_pipeline * pipeline);

// Execute an in-place AllReduce (sum) across tensors[0..n_devices-1].
// tensors[i] must live on the device managed by backends[i] and be
// contiguous F32, F16, or BF16.
// Preconditions are checked by the CUDA comm dispatcher before calling this.
// Returns true once the reduction work has been enqueued successfully.
bool ggml_cuda_ar_allreduce(
    ggml_cuda_ar_pipeline * pipeline,
    ggml_backend_t        * backends,
    ggml_tensor           ** tensors);

// NULL means unsupported with no submitted work. Finish releases a successful start's handle.
ggml_cuda_ar_pending * ggml_cuda_ar_start(ggml_cuda_ar_pipeline * pipeline, ggml_backend_t * backends, ggml_tensor ** tensors);
void ggml_cuda_ar_finish(ggml_cuda_ar_pending * pending);
