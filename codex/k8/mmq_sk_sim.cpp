// Host model of the mul_mat_q stream-k partition and fixup (ggml/src/ggml-cuda/mmq.cuh: mul_mat_q, mul_mat_q_stream_k_fixup,
// mmq_get_stream_k_plan). It transcribes the block/part arithmetic and the order of the float additions, nothing else, so it proves
// the design (a narrow tile that partitions in default-tile units accumulates every output element in the default tile's order);
// the CUDA kernels themselves are checked by the window's bitwise hash gates.
// One simulated element stands for the 64 rows x J columns of one row group: the partition does not depend on the position inside a tile.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

struct shape {
    int rows, J, ncols, nch, ns, kb, bpi, nsm;
    int ntx() const { return (ncols + J - 1) / J; }
    int nz() const { return nch * ns; }
};

struct launch {
    int I;      // rows per block tile (64 or 128)
    int group;  // blocks per partition tile (default I / I)
    int nparts;
    bool bug_sub = false; // negative control: the fixup reads the partials of sub-block 0 only
};

static bool tiling_efficient(int ntiles, int nsm) {
    const int nwaves = (ntiles + nsm - 1) / nsm;
    return 100 * ntiles / (nsm * nwaves) >= 90;
}

// mmq_get_stream_k_plan: partition tiles are default (128-row) tiles.
static int plan_nparts(const shape & s, int I_default, int I, bool NVIDIA = true) {
    const int group = I_default / I;
    const int nty   = (s.rows + group * I - 1) / (group * I);
    const int ntiles = nty * s.ntx() * s.nz();
    return NVIDIA && tiling_efficient(ntiles, s.nsm) ? ntiles : s.nsm;
}

static std::vector<float> run(const shape & s, const launch & L, const std::vector<float> & terms) {
    const int rg_per_tile = L.I / 64;
    const int nrg  = s.rows / 64;
    const int ntx  = s.ntx();
    const int nz   = s.nz();
    const int kb   = s.kb;
    const int bpi  = s.bpi;
    const int nty  = (s.rows + L.group * L.I - 1) / (L.group * L.I);
    const int64_t total = (int64_t) s.ns * s.nch * ntx * nty * kb;
    const int blocks = L.nparts * L.group;
    const int niter = kb / bpi;

    auto elem = [&](int rg, int jt, int z) { return ((size_t) rg * ntx + jt) * nz + z; };
    std::vector<float> dst((size_t) nrg * ntx * nz, NAN);
    std::vector<float> tmp((size_t) blocks * rg_per_tile, NAN);

    auto process = [&](int block, int it, int jt, int z, int kb0_start, int kb0_stop, bool fixup) {
        for (int gi = 0; gi < rg_per_tile; ++gi) {
            const size_t e = elem(it * rg_per_tile + gi, jt, z);
            float sum = 0.0f;
            for (int kb0 = kb0_start; kb0 < kb0_stop; kb0 += bpi) {
                sum += terms[(e * niter + kb0 / bpi) * 2 + 0];
                sum += terms[(e * niter + kb0 / bpi) * 2 + 1];
            }
            if (fixup) { tmp[(size_t) block * rg_per_tile + gi] = sum; } else { dst[e] = sum; }
        }
    };
    auto decode = [&](int64_t kbc, int sub, int & it, int & jt, int & z) {
        int64_t t = kbc / kb;
        jt = t % ntx; t /= ntx;
        const int zt = t % s.nch; t /= s.nch;
        const int wt = t % s.ns;  t /= s.ns;
        it = (int) t * L.group + sub;
        z  = wt * s.nch + zt;
    };

    for (int block = 0; block < blocks; ++block) {
        const int part = block / L.group, sub = block % L.group;
        int64_t kbc      = (int64_t) part       * total / L.nparts;
        int64_t kbc_stop = (int64_t) (part + 1) * total / L.nparts;
        kbc      -= (kbc % kb)      % bpi;
        kbc_stop -= (kbc_stop % kb) % bpi;
        int kb0_start = kbc % kb;
        int kb0_stop  = std::min<int64_t>(kb, kb0_start + kbc_stop - kbc);
        int it, jt, z;
        while (kbc < kbc_stop && kb0_stop == kb) {
            decode(kbc, sub, it, jt, z);
            process(block, it, jt, z, kb0_start, kb0_stop, false);
            kbc += kb;
            kbc -= kbc % kb;
            kb0_start = 0;
            kb0_stop  = std::min<int64_t>(kb, kbc_stop - kbc);
        }
        if (kbc >= kbc_stop) {
            continue;
        }
        decode(kbc, sub, it, jt, z);
        process(block, it, jt, z, kb0_start, kb0_stop, true);
    }

    for (int block = 0; block < blocks; ++block) {
        const int bidx0 = block / L.group, sub = block % L.group;
        int64_t kbc0      = (int64_t) bidx0       * total / L.nparts;
        int64_t kbc0_stop = (int64_t) (bidx0 + 1) * total / L.nparts;
        kbc0      -= (kbc0 % kb)      % bpi;
        kbc0_stop -= (kbc0_stop % kb) % bpi;
        if (kbc0 == kbc0_stop || kbc0 % kb == 0 || (kbc0 / kb == kbc0_stop / kb && kbc0_stop % kb != 0)) {
            continue;
        }
        std::vector<float> sum(rg_per_tile, 0.0f);
        bool any = false;
        int bidx = bidx0 - 1;
        int64_t kbc_stop = kbc0;
        while (true) {
            if (bidx < 0) { fprintf(stderr, "fixup walked below block 0\n"); exit(2); }
            int64_t kbc = (int64_t) bidx * total / L.nparts;
            kbc -= (kbc % kb) % bpi;
            if (kbc == kbc_stop) { bidx--; kbc_stop = kbc; continue; }
            any = true;
            for (int gi = 0; gi < rg_per_tile; ++gi) {
                sum[gi] += tmp[(size_t) (bidx * L.group + (L.bug_sub ? 0 : sub)) * rg_per_tile + gi];
            }
            if (kbc % kb == 0 || kbc / kb < kbc0 / kb) { break; }
            bidx--;
            kbc_stop = kbc;
        }
        if (!any) { continue; }
        int it, jt, z;
        decode(kbc0, sub, it, jt, z);
        for (int gi = 0; gi < rg_per_tile; ++gi) {
            dst[elem(it * rg_per_tile + gi, jt, z)] += sum[gi];
        }
    }
    return dst;
}

static size_t ndiff(const std::vector<float> & a, const std::vector<float> & b) {
    size_t n = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        n += memcmp(&a[i], &b[i], sizeof(float)) != 0;
    }
    return n;
}

int main() {
    const int rows_list[] = { 128, 256, 1024, 2688, 3072, 5120, 5248, 6144, 8704, 9216, 11520, 17408, 124160 };
    const struct { int J, ncols; } cols_list[] = { { 16, 9 }, { 16, 12 }, { 16, 24 }, { 24, 24 }, { 24, 25 }, { 16, 40 } };
    const struct { int nch, ns; } z_list[] = { { 1, 1 }, { 4, 1 }, { 2, 3 } };
    const struct { int kb, bpi; } k_list[] = { { 2, 1 }, { 12, 1 }, { 20, 1 }, { 34, 1 }, { 40, 8 }, { 160, 8 } };
    const int nsm_list[] = { 28, 36, 48, 84 };

    std::mt19937 rng(7);
    size_t cases = 0, plain = 0, split = 0;
    size_t aligned_bad = 0, exact_bad = 0;
    size_t d_l1 = 0, d_force = 0, d_mut = 0, n_l1 = 0, n_force = 0, n_mut = 0;

    for (int rows : rows_list) for (auto c : cols_list) for (auto zz : z_list) for (auto kk : k_list) for (int nsm : nsm_list) {
        shape s = { rows, c.J, c.ncols, zz.nch, zz.ns, kk.kb, kk.bpi, nsm };
        if (rows == 124160 && (zz.nch * zz.ns > 1 || kk.kb > 20)) {
            continue; // keep the lm_head row count with the shapes it has in the model
        }
        const int np = plan_nparts(s, 128, 128);
        const int ntiles128 = ((rows + 127) / 128) * s.ntx() * s.nz();
        const launch D  = { 128, 1, np };
        const launch K8 = { 64, 2, np };                                         // partition in default units
        const launch L1 = { 64, 1, plan_nparts(s, 64, 64) };                     // L1 as it was: own 64-row tiles, one block per part
        const launch FO = { 64, 1, nsm };                                        // forced small tiles, grid = nsm
        const launch MU = { 64, 2, nsm + 1 };                                    // misaligned partition

        const size_t nelem = (size_t) (rows / 64) * s.ntx() * s.nz();
        const int niter = kk.kb / kk.bpi;
        std::vector<float> terms(nelem * niter * 2), iterms(terms.size());
        std::uniform_real_distribution<float> mant(-1.0f, 1.0f);
        std::uniform_int_distribution<int> expo(-6, 8);
        for (size_t i = 0; i < terms.size(); ++i) {
            terms[i]  = std::ldexp(mant(rng), expo(rng));
            iterms[i] = (float) (int) (rng() % 7) - 3.0f;
        }

        // Completeness: integer terms sum exactly in any order, so every variant must equal the plain sum of its terms.
        std::vector<float> exact(nelem, 0.0f);
        for (size_t e = 0; e < nelem; ++e) {
            for (int i = 0; i < niter * 2; ++i) { exact[e] += iterms[e * niter * 2 + i]; }
        }
        for (const launch * L : { &D, &K8, &L1, &FO, &MU }) {
            if (ndiff(run(s, *L, iterms), exact) != 0) { exact_bad++; }
        }

        const std::vector<float> d = run(s, D, terms);
        aligned_bad += ndiff(d, run(s, K8, terms)) != 0;
        n_l1++; n_force++; n_mut++;
        d_l1    += ndiff(d, run(s, L1, terms)) != 0;
        d_force += ndiff(d, run(s, FO, terms)) != 0;
        d_mut   += ndiff(d, run(s, MU, terms)) != 0;
        cases++;
        (np == ntiles128 ? plain : split)++;
    }

    // Negative control for the completeness check itself: a fixup that mixes up the sub-blocks must fail it.
    size_t bug_caught = 0, bug_total = 0;
    for (int rows : { 3072, 5120, 6144 }) {
        shape s = { rows, 16, 12, 1, 1, 20, 1, 36 };
        const size_t nelem = (size_t) (rows / 64) * s.ntx() * s.nz();
        std::vector<float> it(nelem * 20 * 2), ex(nelem, 0.0f);
        for (size_t i = 0; i < it.size(); ++i) { it[i] = (float) (int) (rng() % 7) - 3.0f; }
        for (size_t e = 0; e < nelem; ++e) { for (int i = 0; i < 40; ++i) { ex[e] += it[e * 40 + i]; } }
        launch B = { 64, 2, plan_nparts(s, 128, 128), true };
        bug_total++;
        bug_caught += ndiff(run(s, B, it), ex) != 0;
    }

    // What the window's hash gates must see at 36 SMs (the RTX 5060 Ti), per np2 model shape: 1 = bits differ from the default tile.
    printf("model shapes at nsm=36 (k=5120, kb=20): rows cols default K8 L1-own-grid forced(grid=nsm) misaligned\n");
    for (int rows : { 1024, 2688, 3072, 5120, 5248, 6144, 8704, 9216, 11520, 17408 }) {
        for (auto c : { std::pair<int, int>{ 16, 12 }, std::pair<int, int>{ 24, 24 } }) {
            shape s = { rows, c.first, c.second, 1, 1, 20, 1, 36 };
            const int np = plan_nparts(s, 128, 128);
            const int ntiles128 = ((rows + 127) / 128) * s.ntx();
            const size_t nelem = (size_t) (rows / 64) * s.ntx() * s.nz();
            std::vector<float> terms(nelem * 20 * 2);
            std::uniform_real_distribution<float> mant(-1.0f, 1.0f);
            std::uniform_int_distribution<int> expo(-6, 8);
            for (float & t : terms) { t = std::ldexp(mant(rng), expo(rng)); }
            const std::vector<float> d = run(s, { 128, 1, np }, terms);
            printf("  %6d %2d %-8s %d %d %d %d\n", rows, c.second, np == ntiles128 ? "plain" : "stream-k",
                   (int) (ndiff(d, run(s, { 64, 2, np }, terms)) != 0), (int) (ndiff(d, run(s, { 64, 1, plan_nparts(s, 64, 64) }, terms)) != 0),
                   (int) (ndiff(d, run(s, { 64, 1, 36 }, terms)) != 0), (int) (ndiff(d, run(s, { 64, 2, 37 }, terms)) != 0));
        }
    }
    printf("fixup sub-block mix-up caught by the completeness check: %zu/%zu (must be all)\n", bug_caught, bug_total);

    printf("cases=%zu default_plain=%zu default_streamk=%zu\n", cases, plain, split);
    printf("K8 partition vs default tile: %zu cases differ (must be 0)\n", aligned_bad);
    printf("integer-term completeness failures over all variants: %zu (must be 0)\n", exact_bad);
    printf("negative controls, cases whose bits differ from the default tile: L1 own-tile grid %zu/%zu, forced small grid=nsm %zu/%zu, misaligned %zu/%zu (each must be > 0)\n",
           d_l1, n_l1, d_force, n_force, d_mut, n_mut);
    const bool ok = aligned_bad == 0 && exact_bad == 0 && d_l1 > 0 && d_force > 0 && d_mut > 0 && bug_caught == bug_total;
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
