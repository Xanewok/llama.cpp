#pragma once

#include "common.cuh"
#include "ggml-backend-impl.h"

#include <cstddef>

// Opaque pipeline context -- owns all pinned buffers, streams, and events.
struct ggml_cuda_ar_pipeline;

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


// Node mode: the AllReduce is a stream-ordered node of the per-device graph
// (and thus capturable), instead of a host-driven call between subgraphs.

// Immutable launch descriptor, owned by the pipeline and deduplicated by
// (rank, site, data, count, type, compute) so its address is stable across plans.
struct ggml_cuda_ar_node_desc;

// Plan time: returns the descriptor for `tensor` (the rank's shard) at static
// `site`, or nullptr when the chunked-kernel rules do not hold (type, layout,
// alignment, wire bytes >= copy threshold or > staging, tail elements).
// Allocates per-site staging on first use; sites are shared by all graphs.
// Never call during stream capture.
ggml_cuda_ar_node_desc * ggml_cuda_ar_node_plan(
    ggml_cuda_ar_pipeline * pipeline, int rank, const ggml_tensor * tensor, int site, bool compute);

int ggml_cuda_ar_node_rank(const ggml_cuda_ar_node_desc * desc);

// Stream work only: no event, no host sync, no allocation.  Capturable.
void ggml_cuda_ar_node_launch(const ggml_cuda_ar_node_desc * desc, cudaStream_t stream);
