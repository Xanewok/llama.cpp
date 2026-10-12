#!/usr/bin/env python3
"""Static check of the MMVQ launch-parameter port (no build, no GPU).

1. Text: the general tables in the port equal master abee0c847's; the port's final
   mmvq-params.h equals stack10 a588f2a44's.
2. Model: hand-encoded tables (from those sources) enumerate every
   (arch table, type, ncols 1..8, small_k, halve_iters) and compare
   master vs port (all arches) and stack10 vs port (sm_120).
Run from the worktree root: python3 codex/stack11-mmvq/compare_tables.py
"""
import os
import re
import subprocess
import sys

MASTER, STACK10, PORT_GENERAL, PORT = "abee0c847", "a588f2a44", "b7a0a5688", "b2a10f8b7"
os.environ["GIT_NO_LAZY_FETCH"] = "1"


def show(rev, path):
    return subprocess.run(["git", "show", f"{rev}:{path}"], check=True, capture_output=True, text=True).stdout


def between(s, a, b):
    i = s.index(a)
    return s[i:s.index(b, i)]


fails = []

# --- 1. text checks ---------------------------------------------------------
m = show(MASTER, "ggml/src/ggml-cuda/mmvq.cu").replace("__host__ __device__ ", "")
m_tables = between(m, "static constexpr int calc_nwarps", "template <ggml_type type, int ncols_dst, bool has_fusion")
g = show(PORT_GENERAL, "ggml/src/ggml-cuda/mmvq-params.h")
g_tables = between(g, "static constexpr int calc_nwarps", "template <int nwarps_")
# The port's only edits to the general tables: the type parameter, unused there.
g_norm = g_tables.replace("    GGML_UNUSED(type);\n", "").replace(", ggml_type type = GGML_TYPE_COUNT)", ")")
if m_tables.strip() != g_norm.strip():
    fails.append("text: port general tables != master tables")
if show(STACK10, "ggml/src/ggml-cuda/mmvq-params.h") != show(PORT, "ggml/src/ggml-cuda/mmvq-params.h"):
    fails.append("text: port mmvq-params.h != stack10 mmvq-params.h")
host_bw = "ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_BLACKWELL) {\n        return MMVQ_PARAMETERS_BLACKWELL;"
if host_bw not in show(PORT, "ggml/src/ggml-cuda/mmvq.cu") or host_bw not in show(STACK10, "ggml/src/ggml-cuda/mmvq.cu"):
    fails.append("text: host Blackwell table selection missing")

# --- 2. model ---------------------------------------------------------------
TYPES = ("Q1_0 Q2_0 Q4_0 Q4_1 Q5_0 Q5_1 Q8_0 MXFP4 NVFP4 Q2_K Q3_K Q4_K Q5_K Q6_K IQ2_XXS IQ2_XS "
         "IQ2_S IQ3_XXS IQ1_S IQ1_M IQ4_NL IQ4_XS IQ3_S").split()
GENERIC, TURING, GCN, RDNA2, RDNA3_0, RDNA4, GB10, BLACKWELL = range(8)
NAMES = "GENERIC TURING GCN RDNA2 RDNA3_0 RDNA4 GB10 BLACKWELL".split()


def nwarps_general(t, n, tab, small_k, halve):  # master abee0c847 mmvq.cu calc_nwarps
    if tab == GENERIC:
        return 4 if 1 <= n <= 4 else 2 if 5 <= n <= 8 else 1
    if tab == GCN:
        return 2 if 1 <= n <= 4 else 1
    if tab == RDNA4:
        return 8 if n == 1 and t in "Q4_0 Q4_1 Q5_0 Q5_1 Q8_0 Q2_K Q4_K Q5_K Q6_K IQ4_NL IQ4_XS".split() else 1
    if tab == RDNA3_0:
        if n == 1:
            return 8 if t in "Q4_0 Q4_1 Q5_0 Q5_1 Q8_0 IQ4_NL".split() else 2 if t == "Q6_K" else 1
        return 1
    if tab == TURING:
        if n == 1:
            return 2 if t in "Q2_K Q3_K Q4_K Q5_K Q6_K".split() else 4
        return 4 if 2 <= n <= 4 else 2 if 5 <= n <= 8 else 1
    if tab == GB10:
        gen = nwarps_general(t, n, GENERIC, False, False)
        if n == 1 and not small_k and halve and t in "Q4_0 Q4_1 Q5_0 Q5_1 Q8_0 Q4_K Q5_K Q6_K IQ4_NL".split():
            return 2 * gen
        return gen
    return 1  # RDNA2


def rows_general(n, tab, small_k, nw):  # master calc_rows_per_block
    if tab in (GENERIC, GCN, TURING, GB10):
        return (nw if small_k else 1) if n == 1 else 2 if 2 <= n <= 8 else 1
    return 1


def nwarps_port(t, n, tab, small_k, halve):  # stack10 a588f2a44 == port mmvq-params.h
    if tab == BLACKWELL:
        if 5 <= n <= 8:
            if t in ("Q4_K", "IQ4_XS"):
                return 1
            if t == "Q5_K":
                return 2 if n == 5 else 1
            if t == "Q6_K":
                return 1 if n == 7 else 2
        return nwarps_general(t, n, GENERIC, small_k, halve)
    return nwarps_general(t, n, tab, small_k, halve)


def rows_port(n, tab, small_k, nw, t):
    if tab == BLACKWELL:
        if 5 <= n <= 8:
            return 6 if t == "Q6_K" and n == 8 else 4
        return rows_general(n, GENERIC, small_k, nw)
    return rows_general(n, tab, small_k, nw)


def promoted(t):  # mmvq.cu c_promoted: halve_iters only instantiates where GB10 doubles nwarps
    return nwarps_general(t, 1, GB10, False, True) != nwarps_general(t, 1, GB10, False, False)


def variants(n):
    return [(False, False), (True, False), (False, True)] if n == 1 else [(False, False)]


# Arch -> table. Master: sm_120 builds GENERIC (no Blackwell table); port/stack10: BLACKWELL.
ARCHES = {"sm_70/75 (TURING)": (TURING, TURING), "GCN/CDNA": (GCN, GCN), "RDNA2/3.5": (RDNA2, RDNA2),
          "RDNA3.0": (RDNA3_0, RDNA3_0), "RDNA4": (RDNA4, RDNA4), "sm_121 GB10": (GB10, GB10),
          "sm_80/86/89/90/100 (GENERIC)": (GENERIC, GENERIC), "sm_120": (GENERIC, BLACKWELL)}

checked = 0
sm120_diffs = []
for arch, (tab_master, tab_port) in ARCHES.items():
    for t in TYPES:
        for n in range(1, 9):
            for small_k, halve in variants(n):
                halve = halve and promoted(t)
                nw_m = nwarps_general(t, n, tab_master, small_k, halve)
                cfg_m = (nw_m, rows_general(n, tab_master, small_k, nw_m))
                nw_p = nwarps_port(t, n, tab_port, small_k, halve)
                cfg_p = (nw_p, rows_port(n, tab_port, small_k, nw_p, t))
                checked += 1
                if cfg_m != cfg_p:
                    if arch == "sm_120":
                        sm120_diffs.append((t, n, small_k, halve, cfg_m, cfg_p))
                    else:
                        fails.append(f"model: {arch} {t} n={n} small_k={small_k} halve={halve}: master {cfg_m} port {cfg_p}")

# Intended sm_120 differences = exactly the Blackwell entries at 5-8 columns.
for t, n, small_k, halve, cfg_m, cfg_p in sm120_diffs:
    if not (5 <= n <= 8 and not small_k and not halve):
        fails.append(f"model: unexpected sm_120 diff {t} n={n} small_k={small_k} halve={halve}")

print(f"M: {checked} (arch, type, ncols, small_k, halve_iters) configurations compared")
print(f"sm_120 master -> port (= stack10) differences: {len(sm120_diffs)}, all at ncols 5-8, small_k/halve_iters off")
by_cfg = {}
for t, n, _, _, cfg_m, cfg_p in sm120_diffs:
    by_cfg.setdefault((cfg_m, cfg_p), []).append(f"{t}@{n}")
for (cfg_m, cfg_p), cells in sorted(by_cfg.items()):
    print(f"  (nwarps, rows) {cfg_m} -> {cfg_p}: {len(cells)} cells, e.g. {', '.join(cells[:6])}")
print("all other arches: port selection == master selection" if not any(f.startswith("model: ") and "sm_120" not in f for f in fails) else "")
if fails:
    print("FAIL:\n  " + "\n  ".join(fails))
    sys.exit(1)
print("OK")
