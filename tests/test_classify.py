"""Kernel-axis classification (judge items L2/H5) on names read from the real recorded traces; the 'old' column is the
classification before the change, so each assertion fails on the old code."""
import sqlite3
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import attrib_pack as ap  # noqa: E402
from tests.test_attrib_pack import HQ, NP1, RT, need  # noqa: E402


def names(path, pattern):
    db = sqlite3.connect(path.resolve().as_uri() + '?mode=ro&immutable=1', uri=True)
    return [r[0] for r in db.execute('select distinct s.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id = k.demangledName '
                                     'where s.value like ?', (pattern,))]


class Classify(unittest.TestCase):
    @need(RT)
    def test_llama_allreduce_stages_are_comm(self):
        for pat in ('%ggml_cuda_ar_add_kernel%', '%convert_unary_cont_vec4<float, __nv_bfloat16>%'):
            found = names(RT, pat)
            self.assertTrue(found, pat)
            for n in found:
                self.assertEqual(ap.classify(n), 'COMM', n)                     # old: ELEM

    @need(RT, NP1)
    def test_llama_gdn_state_kernels_are_kv_gdn(self):
        for pat in ('%concat_non_cont%', '%k_get_rows_float_vec%', '%cpy_scalar<&cpy_1_scalar<float, float>>%'):
            for n in names(NP1, pat):
                self.assertEqual(ap.classify(n), 'KV_GDN', n)                   # old: ELEM
        for n in names(NP1, '%k_get_rows_float<float, float>%'):
            self.assertEqual(ap.classify(n), 'ELEM', n)                         # the token-embedding gather stays ELEM

    @need(NP1)
    def test_llama_kv_dequant_and_hadamard_are_kv_attn(self):
        for pat in ('%dequantize_block_q8_0_f16%', '%fwht_cuda%'):
            found = names(NP1, pat)
            self.assertTrue(found, pat)
            for n in found:
                self.assertEqual(ap.classify(n), 'KV_ATTN', n)                  # old: ELEM

    @need(HQ)
    def test_hq_gdn_inproj_gemm_is_fwd(self):
        found = names(HQ, '%wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x1%')
        self.assertTrue(found)
        for n in found:
            self.assertEqual(ap.classify(n), 'FWD', n)                          # old: KV_GDN

    def test_hq_topk_kernels_are_sampling_but_index_select_is_not(self):
        self.assertEqual(ap.classify('_candidates_kernel'), 'SAMP')             # old: ELEM
        self.assertEqual(ap.classify('_select_kernel'), 'SAMP')
        self.assertEqual(ap.classify('void at::native::index_select_kernel<float>'), 'ELEM')

    @need(NP1)
    def test_item47_axis_shift(self):
        tr = ap.Trace(NP1, need_memcpy=True)
        r = ap.decode_budget(tr, ap.load_model('llama-swift15-iq4xs.json'), 8192, 'llama', k=3, min_rounds=30, draft_head_bytes=1042964480)
        self.assertGreater(r['axes']['KV']['measured_ms'], 2.4)                 # was 1.33 before the reclassification
        self.assertLess(r['axes']['ELEM']['measured_ms'], 1.0)                  # was 2.00
        self.assertEqual(set(r['per_gpu']), {'0', '1'})


if __name__ == '__main__':
    unittest.main()
