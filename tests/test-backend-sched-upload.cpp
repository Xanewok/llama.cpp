// The scheduler must not stall the host on device completion to upload host inputs when the split backend takes
// ownership of the bytes (set_tensor_batch_owned), and the uploads must still reach each compute in order.
//
// The device below queues its work and runs it only on synchronize. Its buffers are plain memory that it reports
// as device memory, and its synchronous buffer writes are not ordered with the queue, as with a separate copy stream.

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cpp.h"
#include "../ggml/src/ggml-backend-impl.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <vector>

enum class owned_mode {
    none,        // no set_tensor_batch_owned: the scheduler falls back to synchronize + set
    refuse,      // reports the capability but refuses every transfer
    copy,        // copies the host bytes before returning, as the contract requires
    alias,       // negative control: keeps pointers to the host bytes instead of copying them
};

struct fake_device {
    owned_mode mode = owned_mode::copy;

    std::deque<std::function<void()>> queue;
    size_t n_sync = 0;

    ggml_backend_t cpu = nullptr;

    ggml_backend_buffer_type buft;
    ggml_backend_device      dev;
    ggml_backend_i           iface;
    ggml_backend             backend;
};

static void fake_drain(fake_device * fd) {
    while (!fd->queue.empty()) {
        auto job = std::move(fd->queue.front());
        fd->queue.pop_front();
        job();
    }
}

// buffer

static void fake_buffer_free(ggml_backend_buffer_t buffer) {
    free(buffer->context);
}

static void * fake_buffer_get_base(ggml_backend_buffer_t buffer) {
    return buffer->context;
}

static void fake_buffer_set_tensor(ggml_backend_buffer_t, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    memcpy((char *) tensor->data + offset, data, size);
}

static void fake_buffer_get_tensor(ggml_backend_buffer_t, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    memcpy(data, (const char *) tensor->data + offset, size);
}

static void fake_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    memset(buffer->context, value, buffer->size);
}

static ggml_backend_buffer_i fake_buffer_iface() {
    ggml_backend_buffer_i i = {};
    i.free_buffer = fake_buffer_free;
    i.get_base    = fake_buffer_get_base;
    i.set_tensor  = fake_buffer_set_tensor;
    i.get_tensor  = fake_buffer_get_tensor;
    i.clear       = fake_buffer_clear;
    return i;
}

// buffer type

static const char * fake_buft_get_name(ggml_backend_buffer_type_t) {
    return "fake_device";
}

static ggml_backend_buffer_t fake_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * data = calloc(1, size + 64);
    return ggml_backend_buffer_init(buft, fake_buffer_iface(), data, size);
}

static size_t fake_buft_get_alignment(ggml_backend_buffer_type_t) {
    return 32;
}

static bool fake_buft_is_host(ggml_backend_buffer_type_t) {
    return false;
}

// device

static const char * fake_dev_get_name(ggml_backend_dev_t) {
    return "fake_device";
}

static enum ggml_backend_dev_type fake_dev_get_type(ggml_backend_dev_t) {
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static ggml_backend_buffer_type_t fake_dev_get_buffer_type(ggml_backend_dev_t dev) {
    return &((fake_device *) dev->context)->buft;
}

static bool fake_dev_supports_op(ggml_backend_dev_t, const ggml_tensor * op) {
    return op->op == GGML_OP_SCALE || op->op == GGML_OP_NONE;
}

static bool fake_dev_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return buft == &((fake_device *) dev->context)->buft;
}

// backend

static const char * fake_get_name(ggml_backend_t) {
    return "fake_device";
}

static void fake_free(ggml_backend_t) {
}

static void fake_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    auto * fd = (fake_device *) backend->context;
    fd->queue.push_back([=] { memcpy(data, (const char *) tensor->data + offset, size); });
}

static void fake_synchronize(ggml_backend_t backend) {
    auto * fd = (fake_device *) backend->context;
    fd->n_sync++;
    fake_drain(fd);
}

static enum ggml_status fake_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    auto * fd = (fake_device *) backend->context;
    fd->queue.push_back([fd, cgraph] { ggml_backend_graph_compute(fd->cpu, cgraph); });
    return GGML_STATUS_SUCCESS;
}

static bool fake_set_tensor_batch_owned(ggml_backend_t backend, const ggml_backend_tensor_transfer * transfers, size_t n_transfers) {
    auto * fd = (fake_device *) backend->context;
    if (fd->mode == owned_mode::refuse && n_transfers > 0) {
        return false;
    }
    for (size_t i = 0; i < n_transfers; ++i) {
        void * dst = (char *) transfers[i].tensor->data + transfers[i].offset;
        const size_t size = transfers[i].size;
        if (fd->mode == owned_mode::alias) {
            const void * src = transfers[i].data;
            fd->queue.push_back([=] { memcpy(dst, src, size); });
        } else {
            std::vector<char> owned((const char *) transfers[i].data, (const char *) transfers[i].data + size);
            fd->queue.push_back([=] { memcpy(dst, owned.data(), size); });
        }
    }
    return true;
}

static void fake_init(fake_device & fd, owned_mode mode, ggml_backend_t cpu) {
    fd.mode = mode;
    fd.cpu  = cpu;

    fd.buft = {};
    fd.buft.iface.get_name       = fake_buft_get_name;
    fd.buft.iface.alloc_buffer   = fake_buft_alloc_buffer;
    fd.buft.iface.get_alignment  = fake_buft_get_alignment;
    fd.buft.iface.is_host        = fake_buft_is_host;
    fd.buft.device               = &fd.dev;
    fd.buft.context              = &fd;

    fd.dev = {};
    fd.dev.iface.get_name        = fake_dev_get_name;
    fd.dev.iface.get_description = fake_dev_get_name;
    fd.dev.iface.get_type        = fake_dev_get_type;
    fd.dev.iface.get_buffer_type = fake_dev_get_buffer_type;
    fd.dev.iface.supports_op     = fake_dev_supports_op;
    fd.dev.iface.supports_buft   = fake_dev_supports_buft;
    fd.dev.context               = &fd;

    fd.iface = {};
    fd.iface.get_name               = fake_get_name;
    fd.iface.free                   = fake_free;
    fd.iface.get_tensor_async       = fake_get_tensor_async;
    fd.iface.synchronize            = fake_synchronize;
    fd.iface.graph_compute          = fake_graph_compute;
    fd.iface.set_tensor_batch_owned = mode == owned_mode::none ? nullptr : fake_set_tensor_batch_owned;

    fd.backend = {};
    fd.backend.iface   = fd.iface;
    fd.backend.device  = &fd.dev;
    fd.backend.context = &fd;
}

// n_rounds graphs are queued back to back, each with a different host input, before one final synchronize
static bool run(owned_mode mode, size_t & n_sync_queued, bool & values_ok) {
    const int n_rounds = 4;
    const int n        = 8;

    ggml_backend_ptr cpu(ggml_backend_cpu_init());
    fake_device fd;
    fake_init(fd, mode, cpu.get());

    ggml_backend_t backends[] = { &fd.backend, cpu.get() };
    ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends, nullptr, 2, 64, false, true));

    ggml_init_params params = { 16*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context_ptr ctx(ggml_init(params));
    ggml_tensor * x = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, n);
    ggml_set_input(x);
    ggml_tensor * y = ggml_scale(ctx.get(), x, 2.0f);
    ggml_set_output(y);
    ggml_cgraph * gf = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(gf, y);

    ggml_backend_sched_set_tensor_backend(sched.get(), y, &fd.backend);
    if (!ggml_backend_sched_alloc_graph(sched.get(), gf)) {
        fprintf(stderr, "alloc failed\n");
        return false;
    }
    if (ggml_backend_sched_get_tensor_backend(sched.get(), y) != &fd.backend || !ggml_backend_buffer_is_host(x->buffer)) {
        fprintf(stderr, "unexpected placement\n");
        return false;
    }

    std::vector<std::vector<float>> out(n_rounds, std::vector<float>(n));
    std::vector<float> host(n);
    const size_t n_sync_setup = fd.n_sync;
    for (int r = 0; r < n_rounds; ++r) {
        for (int i = 0; i < n; ++i) {
            host[i] = float(100*r + i);
        }
        ggml_backend_tensor_set(x, host.data(), 0, ggml_nbytes(x));
        ggml_backend_sched_graph_compute_async(sched.get(), gf);
        ggml_backend_tensor_get_async(&fd.backend, y, out[r].data(), 0, ggml_nbytes(y));
        // the host reuses its input buffer as soon as compute_async returns
        std::fill(host.begin(), host.end(), -1.0f);
        ggml_backend_tensor_set(x, host.data(), 0, ggml_nbytes(x));
    }
    n_sync_queued = fd.n_sync - n_sync_setup;
    ggml_backend_sched_synchronize(sched.get());

    values_ok = true;
    for (int r = 0; r < n_rounds; ++r) {
        for (int i = 0; i < n; ++i) {
            values_ok = values_ok && out[r][i] == 2.0f*float(100*r + i);
        }
    }
    return true;
}

int main() {
    struct test_case {
        const char * name;
        owned_mode   mode;
        bool         expect_no_stall;
        bool         expect_values;
    };
    const test_case cases[] = {
        { "no owned uploads",       owned_mode::none,   false, true  },
        { "owned uploads refused",  owned_mode::refuse, false, true  },
        { "owned uploads",          owned_mode::copy,   true,  true  },
        { "aliased host bytes",     owned_mode::alias,  true,  false }, // negative control: the check must see it
    };

    bool ok = true;
    for (const auto & tc : cases) {
        size_t n_sync = 0;
        bool values_ok = false;
        if (!run(tc.mode, n_sync, values_ok)) {
            return 1;
        }
        const bool case_ok = (!tc.expect_no_stall || n_sync == 0) && values_ok == tc.expect_values;
        printf("%-24s host stalls %zu, values %s: %s\n", tc.name, n_sync, values_ok ? "ok" : "wrong", case_ok ? "OK" : "FAIL");
        ok = ok && case_ok;
    }
    return ok ? 0 : 1;
}
