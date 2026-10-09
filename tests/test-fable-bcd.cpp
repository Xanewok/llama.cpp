// local: fable-bcd existence probes for levers B (L2 prefetch during the AllReduce wait), C (weight prefetch above the PDL
// wait) and D (GDN verify snapshot writes). Probe only: nothing here is served. Design, controls and verdict rules:
// codex/REPORT-fable-bcd.md; the verdicts come from codex/fable_bcd_summary.py, this driver only prints rows.
//
// Row format (stdout, tab separated): ROW dev proc lever config arm round trim_us median_us total_us host_us
// All arms of a config are built first and measured interleaved, the order reversed on odd rounds.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "../ggml/src/ggml-cuda/fable-probe.h"

#include <dlfcn.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

struct env_t {
    ggml_backend_t           be   = nullptr;
    const fable_api *        api  = nullptr;
    std::string              dev;
    std::string              proc;
    int                      rounds = 7;
    size_t                   l2 = 0;
    void *                   flush_buf = nullptr;
    size_t                   flush_bytes = 0;
    unsigned *               sink = nullptr;
};

[[noreturn]] static void die(const char * msg) {
    fprintf(stderr, "test-fable-bcd: FAIL %s\n", msg);
    fflush(stderr);
    exit(8);
}

// ---------------------------------------------------------------------------------------------------------------------

static std::vector<uint8_t> quant_row(ggml_type type, int64_t k, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> f(k);
    for (auto & v : f) {
        v = nd(rng);
    }
    std::vector<uint8_t> out(ggml_row_size(type, k));
    ggml_quantize_init(type);
    const size_t n = ggml_quantize_chunk(type, f.data(), out.data(), 0, 1, k, nullptr);
    if (n != out.size()) {
        die("quantize_chunk size");
    }
    return out;
}

struct pool_t {
    ggml_context *              ctx = nullptr;
    ggml_backend_buffer_t       buf = nullptr;
    std::vector<ggml_tensor *>  w;
    int64_t                     k = 0;
    int64_t                     m = 0;
};

static std::unique_ptr<pool_t> make_pool(env_t & env, ggml_type type, int64_t k, int64_t m, int count) {
    auto p = std::make_unique<pool_t>();
    p->k = k;
    p->m = m;
    ggml_init_params ip = { ggml_tensor_overhead() * (size_t) (count + 4), nullptr, true };
    p->ctx = ggml_init(ip);
    for (int i = 0; i < count; ++i) {
        p->w.push_back(ggml_new_tensor_2d(p->ctx, type, k, m));
    }
    p->buf = ggml_backend_alloc_ctx_tensors(p->ctx, env.be);
    if (!p->buf) {
        die("weight pool allocation");
    }
    ggml_backend_buffer_set_usage(p->buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    const std::vector<uint8_t> row = quant_row(type, k, 1234);
    for (ggml_tensor * t : p->w) {
        env.api->fill_pattern(env.be, t->data, ggml_nbytes(t), row.data(), row.size());
    }
    return p;
}

static void free_pool(std::unique_ptr<pool_t> & p) {
    ggml_backend_buffer_free(p->buf);
    ggml_free(p->ctx);
    p.reset();
}

struct arm_t {
    std::string           name;
    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_cgraph *         gf  = nullptr;
    fable_arm             cfg = {};
};

static ggml_context * new_arm_ctx(size_t n_tensors) {
    ggml_init_params ip = { ggml_tensor_overhead() * (n_tensors + 16) + ggml_graph_overhead_custom(n_tensors + 16, false) + (1u << 20),
                            nullptr, true };
    return ggml_init(ip);
}

static void upload_random(ggml_tensor * t, float lo, float hi, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> ud(lo, hi);
    std::vector<float> v(ggml_nelements(t));
    for (auto & x : v) {
        x = ud(rng);
    }
    ggml_backend_tensor_set(t, v.data(), 0, v.size() * sizeof(float));
}

static void finish_arm(env_t & env, arm_t & a) {
    a.buf = ggml_backend_alloc_ctx_tensors(a.ctx, env.be);
    if (!a.buf) {
        die("arm tensor allocation");
    }
}

static void free_arm(arm_t & a) {
    ggml_backend_buffer_free(a.buf);
    ggml_free(a.ctx);
    a.ctx = nullptr;
    a.buf = nullptr;
}

static void measure(env_t & env, const char * lever, const std::string & config, std::vector<arm_t> & arms) {
    for (int r = 0; r < env.rounds; ++r) {
        for (size_t j = 0; j < arms.size(); ++j) {
            arm_t & a = arms[(r % 2 == 0) ? j : arms.size() - 1 - j];
            fable_result res = {};
            // 400 reps for B (one GEMM per rep), fewer for whole-graph arms
            const int reps = a.cfg.flush || a.cfg.spin_ns > 0 ? 400 : (ggml_graph_n_nodes(a.gf) > 100 ? 60 : 120);
            env.api->run(env.be, &a.cfg, reps, &res);
            if (res.rc != 0) {
                die("graph compute failed");
            }
            printf("ROW\t%s\t%s\t%s\t%s\t%s\t%d\t%.4f\t%.4f\t%.4f\t%.4f\n", env.dev.c_str(), env.proc.c_str(), lever, config.c_str(),
                   a.name.c_str(), r, res.trim_us, res.median_us, res.total_us, res.host_us);
            fflush(stdout);
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// B: one GEMM, cold L2 vs after a spin kernel that prefetches the first MB of its weights.

static arm_t build_single(env_t & env, const std::string & name, pool_t & pool, int64_t n) {
    arm_t a;
    a.name = name;
    a.ctx  = new_arm_ctx(8);
    ggml_tensor * x   = ggml_new_tensor_2d(a.ctx, GGML_TYPE_F32, pool.k, n);
    ggml_tensor * out = ggml_mul_mat(a.ctx, pool.w[0], x);
    a.gf = ggml_new_graph_custom(a.ctx, 16, false);
    ggml_build_forward_expand(a.gf, out);
    finish_arm(env, a);
    upload_random(x, -1.0f, 1.0f, 77);
    a.cfg.gf = a.gf;
    a.cfg.sink = env.sink;
    return a;
}

static void run_b(env_t & env) {
    struct shape_t { int64_t k, m; };
    const shape_t shapes[] = { { 5120, 8704 }, { 5120, 5120 } };
    const ggml_type types[] = { GGML_TYPE_IQ4_XS, GGML_TYPE_Q6_K };
    const int ns[] = { 6, 12 };
    for (ggml_type type : types) {
        for (const shape_t & sh : shapes) {
            std::unique_ptr<pool_t> pool = make_pool(env, type, sh.k, sh.m, 2);
            const size_t wbytes = ggml_nbytes(pool->w[0]);
            for (int n : ns) {
                printf("ROUTE\t%s\t%s\t%s\t%d\t%d\n", env.dev.c_str(), env.proc.c_str(), ggml_type_name(type), n,
                       env.api->route_mmvq(env.be, (int) type, n));
                char cfg[96];
                snprintf(cfg, sizeof(cfg), "%s_%ldx%ld_n%d", ggml_type_name(type), (long) sh.m, (long) sh.k, n);
                const unsigned long long spin = n == 6 ? 23000ULL : 42000ULL; // measured AR wait at 6 / 12 rows (Fable report, M)
                const void * target = pool->w[0]->data;
                const void * decoy  = pool->w[1]->data;
                auto mb = [&](int m) { return std::min((size_t) m << 20, wbytes) & ~(size_t) 127; };

                std::vector<arm_t> arms;
                auto add = [&](const char * name, int flush, int mode, const void * ptr, size_t bytes) {
                    arms.push_back(build_single(env, name, *pool, n));
                    fable_arm & c = arms.back().cfg;
                    c.flush = flush;
                    c.mode = mode;
                    c.pf_ptr = ptr;
                    c.pf_bytes = bytes;
                    c.spin_ns = spin;
                    c.flush_buf = env.flush_buf;
                    c.flush_bytes = env.flush_bytes;
                };
                add("cold_a",   1, 0, nullptr, 0);
                add("cold_b",   1, 0, nullptr, 0);
                add("warm",     0, 0, nullptr, 0);
                add("pf5",      1, 1, target, mb(5));
                add("pf10",     1, 1, target, mb(10));
                add("pf19",     1, 1, target, mb(19));
                add("wrong10",  1, 1, decoy,  mb(10));
                add("bulk10",   1, 3, target, mb(10));
                add("ldcg10",   1, 2, target, mb(10));
                printf("CFG\t%s\t%s\tB\t%s\tweight_bytes=%zu\tspin_ns=%llu\tpf5=%zu\tpf10=%zu\tpf19=%zu\n", env.dev.c_str(), env.proc.c_str(),
                       cfg, wbytes, spin, mb(5), mb(10), mb(19));
                measure(env, "B", cfg, arms);
                for (arm_t & a : arms) {
                    free_arm(a);
                }
            }
            free_pool(pool);
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// C: chain add -> rms_norm * g -> mul_mat, weights rotating through more bytes than L2 holds, flag-controlled prefetch above the wait.

struct chain_arm_args {
    std::vector<pool_t *> pools;
    int64_t               n;
    int                   links;
};

static arm_t build_chain(env_t & env, const std::string & name, const chain_arm_args & ca, int pf_flag) {
    arm_t a;
    a.name = name;
    a.ctx  = new_arm_ctx((size_t) ca.links * 6 + 16);
    const int64_t k0 = ca.pools[0]->k;
    ggml_tensor * cur = ggml_new_tensor_2d(a.ctx, GGML_TYPE_F32, k0, ca.n);
    ggml_tensor * x0  = cur;
    std::vector<ggml_tensor *> inputs;
    std::vector<ggml_tensor *> r_by_dim;
    std::vector<ggml_tensor *> g_by_dim;
    std::vector<int64_t>       dims;
    auto dim_slot = [&](int64_t d) {
        for (size_t i = 0; i < dims.size(); ++i) {
            if (dims[i] == d) {
                return i;
            }
        }
        dims.push_back(d);
        r_by_dim.push_back(ggml_new_tensor_2d(a.ctx, GGML_TYPE_F32, d, ca.n));
        g_by_dim.push_back(ggml_new_tensor_2d(a.ctx, GGML_TYPE_F32, d, 1));
        return dims.size() - 1;
    };
    const size_t np = ca.pools.size();
    for (int l = 0; l < ca.links; ++l) {
        pool_t * p = ca.pools[l % np];
        if (cur->ne[0] != p->k) {
            die("chain dims");
        }
        const size_t slot = dim_slot(p->k);
        cur = ggml_add(a.ctx, cur, r_by_dim[slot]);
        cur = ggml_rms_norm(a.ctx, cur, 1e-6f);
        cur = ggml_mul(a.ctx, cur, g_by_dim[slot]);
        cur = ggml_mul_mat(a.ctx, p->w[(l / np) % p->w.size()], cur);
    }
    a.gf = ggml_new_graph_custom(a.ctx, (size_t) ca.links * 6 + 16, false);
    ggml_build_forward_expand(a.gf, cur);
    finish_arm(env, a);
    upload_random(x0, -1.0f, 1.0f, 5);
    for (size_t i = 0; i < dims.size(); ++i) {
        upload_random(r_by_dim[i], -0.1f, 0.1f, 11 + (uint32_t) i);
        std::vector<float> ones(dims[i], 1.0f);
        ggml_backend_tensor_set(g_by_dim[i], ones.data(), 0, ones.size() * sizeof(float));
    }
    a.cfg.gf = a.gf;
    a.cfg.pf_flag = pf_flag;
    a.cfg.sink = env.sink;
    return a;
}

static void run_c(env_t & env, bool stock_only) {
    struct chain_t { const char * name; int64_t k1, m1, k2, m2; }; // second pool optional (k2 == 0)
    const chain_t chains[] = {
        { "updown", 5120, 8704, 8704, 5120 },
        { "sq5120", 5120, 5120, 0, 0 },
        { "small",  5120, 3072, 3072, 5120 },
    };
    struct cfg_t { ggml_type type; int n; };
    const cfg_t cfgs[] = {
        { GGML_TYPE_IQ4_XS, 6 }, { GGML_TYPE_Q6_K, 6 }, { GGML_TYPE_IQ4_XS, 1 }, { GGML_TYPE_IQ4_XS, 12 },
    };
    const int links = 48;
    for (const cfg_t & c : cfgs) {
        for (const chain_t & ch : chains) {
            const int np = ch.k2 ? 2 : 1;
            const size_t bytes_each = ggml_row_size(c.type, ch.k1) * (size_t) ch.m1;
            // reuse distance of one weight >= 128 MB, 4x the 32 MB L2 (G)
            const int count = std::max(2, (int) ((128u << 20) / (np * bytes_each)) + 1);
            std::vector<std::unique_ptr<pool_t>> pools;
            pools.push_back(make_pool(env, c.type, ch.k1, ch.m1, count));
            if (ch.k2) {
                pools.push_back(make_pool(env, c.type, ch.k2, ch.m2, count));
            }
            chain_arm_args ca;
            for (auto & p : pools) {
                ca.pools.push_back(p.get());
            }
            ca.n = c.n;
            ca.links = links;
            printf("ROUTE\t%s\t%s\t%s\t%d\t%d\n", env.dev.c_str(), env.proc.c_str(), ggml_type_name(c.type), c.n,
                   env.api->route_mmvq(env.be, (int) c.type, c.n));
            char cfg[96];
            snprintf(cfg, sizeof(cfg), "%s_%s_n%d", ggml_type_name(c.type), ch.name, c.n);
            std::vector<arm_t> arms;
            arms.push_back(build_chain(env, "stock_a", ca, 0));
            arms.push_back(build_chain(env, "stock_b", ca, 0));
            if (!stock_only) {
                arms.push_back(build_chain(env, "pf1", ca, 1));
                arms.push_back(build_chain(env, "pf2", ca, 2));
                arms.push_back(build_chain(env, "pf4", ca, 4));
                arms.push_back(build_chain(env, "pf16", ca, 16));
            }
            printf("CFG\t%s\t%s\tC\t%s\tlinks=%d\tweights_per_pool=%d\tbytes_each=%zu\n", env.dev.c_str(), env.proc.c_str(), cfg, links, count, bytes_each);
            measure(env, "C", cfg, arms);
            for (arm_t & a : arms) {
                free_arm(a);
            }
            for (auto & p : pools) {
                free_pool(p);
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// D: 48 GATED_DELTA_NET ops (one per layer, own state and output so snapshot writes spill to DRAM like in the served round),
// K snapshots vs fewer; alone ("iso") or each followed by two weight GEMMs ("mixed") so DRAM is busy between the scans.

static arm_t build_gdn(env_t & env, const std::string & name, int n_tok, int n_seqs, int K, int layers, bool mixed,
                       pool_t * up, pool_t * down) {
    arm_t a;
    a.name = name;
    a.ctx  = new_arm_ctx((size_t) layers * 6 + 16);
    const int64_t S = 128;
    const int64_t Hk = 8;
    const int64_t Hv = 24;
    ggml_tensor * q    = ggml_new_tensor_4d(a.ctx, GGML_TYPE_F32, S, Hk, n_tok, n_seqs);
    ggml_tensor * k    = ggml_new_tensor_4d(a.ctx, GGML_TYPE_F32, S, Hk, n_tok, n_seqs);
    ggml_tensor * v    = ggml_new_tensor_4d(a.ctx, GGML_TYPE_F32, S, Hv, n_tok, n_seqs);
    ggml_tensor * g    = ggml_new_tensor_4d(a.ctx, GGML_TYPE_F32, 1, Hv, n_tok, n_seqs);
    ggml_tensor * beta = ggml_new_tensor_4d(a.ctx, GGML_TYPE_F32, 1, Hv, n_tok, n_seqs);
    ggml_tensor * x_up = nullptr;
    ggml_tensor * x_dn = nullptr;
    const int64_t ncol = (int64_t) n_tok * n_seqs;
    if (mixed) {
        x_up = ggml_new_tensor_2d(a.ctx, GGML_TYPE_F32, up->k, ncol);
        x_dn = ggml_new_tensor_2d(a.ctx, GGML_TYPE_F32, down->k, ncol);
    }
    std::vector<ggml_tensor *> states(layers);
    for (int l = 0; l < layers; ++l) {
        states[l] = ggml_new_tensor_4d(a.ctx, GGML_TYPE_F32, S, S, Hv, n_seqs);
    }
    a.gf = ggml_new_graph_custom(a.ctx, (size_t) layers * 6 + 16, false);
    for (int l = 0; l < layers; ++l) {
        ggml_tensor * out = ggml_gated_delta_net(a.ctx, q, k, v, g, beta, states[l], nullptr, K);
        ggml_build_forward_expand(a.gf, out);
        if (mixed) {
            ggml_build_forward_expand(a.gf, ggml_mul_mat(a.ctx, up->w[l % up->w.size()], x_up));
            ggml_build_forward_expand(a.gf, ggml_mul_mat(a.ctx, down->w[l % down->w.size()], x_dn));
        }
    }
    finish_arm(env, a);
    upload_random(q, -0.1f, 0.1f, 21);
    upload_random(k, -0.1f, 0.1f, 22);
    upload_random(v, -0.1f, 0.1f, 23);
    upload_random(g, -1.0f, -0.01f, 24);
    upload_random(beta, 0.0f, 1.0f, 25);
    const float small = 0.01f;
    for (int l = 0; l < layers; ++l) {
        env.api->fill_pattern(env.be, states[l]->data, ggml_nbytes(states[l]), &small, sizeof(small));
    }
    if (mixed) {
        upload_random(x_up, -1.0f, 1.0f, 31);
        upload_random(x_dn, -1.0f, 1.0f, 32);
    }
    a.cfg.gf = a.gf;
    a.cfg.sink = env.sink;
    return a;
}

static void run_d(env_t & env) {
    struct shape_t { int n_tok, n_seqs; };
    const shape_t shapes[] = { { 6, 1 }, { 6, 2 } };
    const int layers = 48;
    for (int mixed = 0; mixed <= 1; ++mixed) {
        for (const shape_t & sh : shapes) {
            std::unique_ptr<pool_t> up;
            std::unique_ptr<pool_t> down;
            if (mixed) {
                // 6 distinct matrices per pool: 142 MB / 142 MB of IQ4_XS between reuses
                up   = make_pool(env, GGML_TYPE_IQ4_XS, 5120, 8704, 6);
                down = make_pool(env, GGML_TYPE_IQ4_XS, 8704, 5120, 6);
            }
            char cfg[96];
            snprintf(cfg, sizeof(cfg), "gdn_%s_t%d_s%d", mixed ? "mixed" : "iso", sh.n_tok, sh.n_seqs);
            std::vector<arm_t> arms;
            arms.push_back(build_gdn(env, "K6_a", sh.n_tok, sh.n_seqs, 6, layers, mixed, up.get(), down.get()));
            arms.push_back(build_gdn(env, "K6_b", sh.n_tok, sh.n_seqs, 6, layers, mixed, up.get(), down.get()));
            arms.push_back(build_gdn(env, "K3",   sh.n_tok, sh.n_seqs, 3, layers, mixed, up.get(), down.get()));
            arms.push_back(build_gdn(env, "K1",   sh.n_tok, sh.n_seqs, 1, layers, mixed, up.get(), down.get()));
            printf("CFG\t%s\t%s\tD\t%s\tlayers=%d\tstate_bytes_per_slot_per_layer=%zu\n", env.dev.c_str(), env.proc.c_str(), cfg, layers,
                   (size_t) 128 * 128 * 24 * 4 * sh.n_seqs);
            measure(env, "D", cfg, arms);
            for (arm_t & a : arms) {
                free_arm(a);
            }
            if (mixed) {
                free_pool(up);
                free_pool(down);
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------

int main(int argc, char ** argv) {
    env_t env;
    env.dev = "CUDA1";
    std::string lever = "all";
    bool stock_only = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--dev" && i + 1 < argc) {
            env.dev = argv[++i];
        } else if (a == "--lever" && i + 1 < argc) {
            lever = argv[++i];
        } else if (a == "--rounds" && i + 1 < argc) {
            env.rounds = atoi(argv[++i]);
        } else if (a == "--proc" && i + 1 < argc) {
            env.proc = argv[++i];
        } else if (a == "--stock-only") {
            stock_only = true;
        } else {
            fprintf(stderr, "usage: %s [--dev CUDA1] [--lever B|C|D|all] [--rounds 7] [--proc TAG] [--stock-only]\n", argv[0]);
            return 2;
        }
    }
    if (env.proc.empty()) {
        env.proc = "main";
    }

    const auto api_fn = (const fable_api * (*)(void)) dlsym(RTLD_DEFAULT, "fable_probe_api");
    if (!api_fn) {
        die("fable_probe_api not found (libggml-cuda not loaded?)");
    }
    env.api = api_fn();

    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_name(env.dev.c_str());
    if (!dev) {
        die("device not found");
    }
    env.be = ggml_backend_dev_init(dev, nullptr);
    if (!env.be) {
        die("backend init");
    }

    env.l2 = env.api->l2_bytes(env.be);
    env.flush_bytes = std::max<size_t>(2 * env.l2, (size_t) 64 << 20) & ~(size_t) 15;
    env.flush_buf = env.api->alloc(env.be, env.flush_bytes);
    env.sink = (unsigned *) env.api->alloc(env.be, 64);
    printf("META\t%s\t%s\tl2_bytes=%zu\tflush_bytes=%zu\tnsm=%d\tlever=%s\trounds=%d\tstock_only=%d\n", env.dev.c_str(), env.proc.c_str(), env.l2,
           env.flush_bytes, env.api->nsm(env.be), lever.c_str(), env.rounds, stock_only ? 1 : 0);
    fflush(stdout);

    if (lever == "B" || lever == "all") {
        run_b(env);
    }
    if (lever == "C" || lever == "all") {
        run_c(env, stock_only);
    }
    if (lever == "D" || lever == "all") {
        run_d(env);
    }
    env.api->free_(env.be, env.flush_buf);
    env.api->free_(env.be, env.sink);
    ggml_backend_free(env.be);
    printf("DONE\t%s\t%s\n", env.dev.c_str(), env.proc.c_str());
    return 0;
}
