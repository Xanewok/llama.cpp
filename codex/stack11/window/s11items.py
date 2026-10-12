"""Placeholder filling of the stack11-base A/B item templates (items/*.toml.in) for render_items.py.

Contract: a template keeps every served flag and env of its wave-1 source and the wave-1 canary patterns of its shape; the s11 arm
differs from the s10 arm by the frozen build alone (no env switch); only canary boots carry log envs; the drop item's drop arm differs
from its s11 arm by the two switches alone and its canary proves both switched paths are not taken.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engagement

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
W = ROOT.parent.parent
EVAL = Path('/home/xanewok/repos/qwen38-eval')
CCX01 = W / 'work/aa-load/ccx01.sh'
COMPONENTS = 'p55 k7 k4 k6a k6b k10 k8 k11a k11b'.split()

# Wave-1 canary patterns per shape (stack10 s10items.WAVE1): the part of the s11 canary that is not a component line.
WAVE1 = {
    'np1': ["MTP chain: k=5\\b", "mtp_fold_decode: engaged n_seq=1\\b",
            "set_logits: using backend candidates: n_cand = [1-9]\\d*, n_ban = [1-9]\\d*, top_k = 20\\b", "(?s)^(?!.*set_sampler: [^\\n]*using CPU)"],
    'np2': ["gdn_cache_fusion: graph device=0 graph=\\d+ tokens=6 seqs=2 indexed=([1-9]\\d*) gdn=\\1 fused=\\1\\b",
            "gdn_cache_fusion: graph device=1 graph=\\d+ tokens=6 seqs=2 indexed=([1-9]\\d*) gdn=\\1 fused=\\1\\b",
            "mtp_fold_decode: engaged n_seq=2\\b", "MTP chain disabled: needs one sequence \\(n_seq=2\\)", "(?s)^(?!.*MTP chain: k=)"],
    'duo': ["MTP chain disabled: needs one sequence \\(n_seq=2\\)", "(?s)^(?!.*MTP chain: k=)", "n_kv_pad += 4096\\b", "mtp_fold_decode: engaged n_seq=1\\b"],
}
DROPPED = ('k7', 'k10')
DROP_ABSENT = ["(?s)^(?!.*rms_norm_scale_pair_fusion:)", "(?s)^(?!.*top_k_batched: engaged)"]


def shape_of(text):
    m = re.search(r'^# shape: (np1|np2|duo)$', text, re.M)
    assert m, 'template lacks its "# shape:" line'
    return m.group(1)


def extra_pins(template_name):
    own = ['s11items.py', 'render_items.py', 'engagement.py', f'items/{template_name}']
    return [str(HERE / n) for n in own] + [str(CCX01)]


def fill(text, *, s11_sha, base_sha, components, evidence, template_name):
    shape = shape_of(text)
    env = engagement.canary_env(components, shape)
    patterns = WAVE1[shape] + engagement.canary_patterns(components, shape)
    kept = [c for c in components if c not in DROPPED]
    drop = WAVE1[shape] + engagement.canary_patterns(kept, shape) + DROP_ABSENT
    if '@DROP_CANARY_LOG@' in text:
        # the absence lines prove nothing unless the canary prints them when engaged
        assert set(DROPPED) <= set(components) and env.get('GGML_CUDA_TOPK_LOG') == '1'
    values = {'@S11_SHA@': s11_sha, '@BASE_SHA@': base_sha, '@TREE@': str(ROOT), '@EVIDENCE@': str(evidence),
              '@EXTRA_PINS@': json.dumps(extra_pins(template_name)), '@CANARY_LOG@': json.dumps(patterns),
              '@DROP_CANARY_LOG@': json.dumps(drop),
              '@CANARY_ENV@': ''.join(f'{k} = {json.dumps(v)}\n' for k, v in sorted(env.items())),
              '@COMPONENTS@': ' '.join(components)}
    for key, value in values.items():
        text = text.replace(key, value)
    assert not re.search(r'@[A-Z_0-9]+@', text), 'unfilled placeholder'
    return text


def selftest():
    for p in sorted((HERE / 'items').glob('*.toml.in')):
        text = fill(p.read_text(), s11_sha='1' * 40, base_sha='2' * 40, components=COMPONENTS, evidence='/e', template_name=p.name)
        import tomllib
        x = tomllib.loads(text)
        arms = x['arms']
        if 'drop' in arms:
            assert set(arms) == {'s11', 'drop'}
            d, s = dict(arms['drop']['env']), dict(arms['s11']['env'])
            assert {k: v for k, v in d.items() if k not in s} == {'GGML_CUDA_RMS_NORM_SCALE_PAIR': '0', 'GGML_CUDA_TOPK_BATCHED': '0'}
            assert all(d[k] == v for k, v in s.items()) and arms['drop']['root'] == arms['s11']['root']
            assert not any('rms_norm_scale_pair_fusion: device' in q or 'top_k_batched: engaged ncols' in q for q in arms['drop']['canary_log'])
            for q in DROP_ABSENT:
                assert re.search(q, 'boot\nnothing fused') and not re.search(q, 'x\nrms_norm_scale_pair_fusion: device=0 ncols=1 rows=6\ntop_k_batched: engaged ncols=9 nrows=6 k=20\n')
        else:
            assert set(arms) == {'s10', 's11'} and arms['s10']['env'] == arms['s11']['env'], p.name
            assert arms['s10']['root'].endswith('/frozen/stack10') and arms['s11']['root'].endswith('/frozen/stack11-base')
            assert arms['s10']['canary_log'] == [] and x['canary']['plan'] == ['s11']
        assert x['out'].startswith(f'{EVAL}/results/2026-10-13-') and x['id'] == p.name[:-len('.toml.in')]
        if x['kind'] == 'depth':
            assert x['depth']['identity'] == ('within-arm' if shape_of(p.read_text()) == 'np1' else 'off')
    print('s11items: templates fill, arms differ by build (or by the two drop switches) only, canaries per shape PASS')


if __name__ == '__main__':
    if sys.argv[1:] == ['selftest']:
        selftest()
    else:
        raise SystemExit(__doc__)
