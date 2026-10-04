#include "llama-kv-cells.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>

#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #cond); exit(1); } } while (0)

static uint32_t reference(llama_kv_cells & cells, llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    uint32_t first = cells.size();
    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (!cells.pos_in(i, p0, p1)) {
            continue;
        }
        if (seq_id < 0) {
            cells.rm(i);
            first = std::min(first, i);
        } else if (cells.seq_has(i, seq_id) && cells.seq_rm(i, seq_id)) {
            first = std::min(first, i);
        }
    }
    return first;
}

static void equal(const llama_kv_cells & a, const llama_kv_cells & b) {
    CHECK(a.get_used() == b.get_used());
    CHECK(a.used_min() == b.used_min());
    CHECK(a.used_max_p1() == b.used_max_p1());
    CHECK(a.get_has_shift() == b.get_has_shift());
    for (uint32_t i = 0; i < a.size(); ++i) {
        CHECK(a.is_empty(i) == b.is_empty(i));
        CHECK(a.seq_get_all(i) == b.seq_get_all(i));
        if (!a.is_empty(i)) {
            CHECK(a.pos_get(i) == b.pos_get(i));
            CHECK(a.get_shift(i) == b.get_shift(i));
            CHECK(a.ext_get(i).x == b.ext_get(i).x);
            CHECK(a.ext_get(i).y == b.ext_get(i).y);
            CHECK(a.ext_get(i).tok == b.ext_get(i).tok);
        }
    }
    for (llama_seq_id s = 0; s < 4; ++s) {
        CHECK(a.seq_pos_get(s) == b.seq_pos_get(s));
    }
}

static void regression() {
    std::mt19937 rng(1729);
    for (int trial = 0; trial < 100; ++trial) {
        llama_kv_cells initial;
        initial.resize(513);
        for (uint32_t i = 0; i < initial.size(); ++i) {
            if (rng() % 3 == 0) {
                continue;
            }
            initial.pos_set(i, rng() % 32);
            initial.ext_set(i, {(llama_pos) i, (llama_pos) (i + 1), (llama_token) (i + 2)});
            for (llama_seq_id s = 0; s < 4; ++s) {
                if (rng() % 2) {
                    initial.seq_add(i, s);
                }
            }
            initial.pos_add(i, 3);
            initial.pos_div(i, 2);
        }
        for (llama_seq_id s = -1; s < 4; ++s) {
            for (const auto & range : {std::pair<llama_pos, llama_pos>{0, 0}, {0, 1}, {3, 8}, {8, 3}, {0, INT32_MAX}, {20, INT32_MAX}}) {
                auto a = initial;
                auto b = initial;
                CHECK(a.seq_rm(s, range.first, range.second) == reference(b, s, range.first, range.second));
                equal(a, b);
                CHECK(a.seq_rm(s, range.first, range.second) == reference(b, s, range.first, range.second));
                equal(a, b);
            }
        }
    }
    llama_kv_cells empty;
    empty.resize(64);
    CHECK(empty.seq_rm(-1, 0, INT32_MAX) == empty.size());
    CHECK(empty.seq_rm(0, 0, INT32_MAX) == empty.size());
}

static void bench(uint32_t used) {
    llama_kv_cells cells;
    cells.resize(102400);
    for (uint32_t i = 0; i < used; ++i) {
        cells.pos_set(i, i);
        cells.seq_add(i, 0);
    }
    volatile uint32_t result = 0;
    for (bool fast : {false, true}) {
        const auto start = std::chrono::steady_clock::now();
        for (int repeat = 0; repeat < 1000; ++repeat) {
            result = fast ? cells.seq_rm(0, used, used + 5) : reference(cells, 0, used, used + 5);
        }
        const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count();
        printf("used %u, %s ns/call M: %.1f\n", used, fast ? "indexed" : "capacity", ns / 1000.0);
    }
    CHECK(result == cells.size());
}

int main(int argc, char **) {
    regression();
    if (argc > 1) {
        bench(8192);
        bench(90112);
    }
    puts("KV range removal matches capacity scan");
}
