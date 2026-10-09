// local: fable-bcd probe API (levers B, C, D of the Fable 10-09 review). Compiled into libggml-cuda; the driver
// tests/test-fable-bcd.cpp reaches it with dlsym. Probe code only: nothing here is served.
#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// One timed arm: [flush] [spin + prefetch] <event> graph <event>, repeated. Everything but `gf` is optional.
struct fable_arm {
    struct ggml_cgraph * gf;
    int                  flush;       // sweep flush_buf (read + write) before each rep, so weights start cold in L2
    int                  mode;        // 0 none, 1 prefetch.global.L2 per 128 B line, 2 ld.global.cg into a sink, 3 cp.async.bulk.prefetch.L2
    const void *         pf_ptr;
    size_t               pf_bytes;    // multiple of 128
    unsigned long long   spin_ns;     // stands in for the AllReduce wait
    int                  pf_flag;     // g_fable_probe_pf while this arm's graph is captured and replayed
    void *               flush_buf;
    size_t               flush_bytes; // multiple of 16
    unsigned *           sink;
};

struct fable_result {
    double trim_us;   // mean of the per-rep graph interval, 10 % trimmed each side
    double median_us;
    double total_us;  // whole loop per rep, flush and spin included
    double host_us;   // host enqueue time per rep (CUDA graph engaged when far below trim_us)
    int    rc;
};

struct fable_api {
    size_t (*l2_bytes)(ggml_backend_t);
    int    (*nsm)(ggml_backend_t);
    int    (*route_mmvq)(ggml_backend_t, int type, int n);
    void * (*alloc)(ggml_backend_t, size_t);
    void   (*free_)(ggml_backend_t, void *);
    void   (*fill_pattern)(ggml_backend_t, void * dst, size_t n, const void * host_pat, size_t patn);
    void   (*run)(ggml_backend_t, const struct fable_arm *, int reps, struct fable_result *);
};

const struct fable_api * fable_probe_api(void);

#ifdef __cplusplus
}
#endif
