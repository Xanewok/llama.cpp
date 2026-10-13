"""Placeholder filling of the stack11-s2 A/B item templates (items/*.toml.in) for render_items.py.

Contract: a template keeps every served flag and env of its stack11-base source and the wave-1 canary patterns of its shape; the s2
arm (frozen/stack11-s2 = stack11-base + 87/88/91/90, K7 and K10 kept) differs from the s11 arm (frozen/stack11-base) by the build
alone, no env switch; only canary boots carry log envs. The drop item compares frozen/stack11-s2-nok (K7 and K10 removed from the
source) against frozen/stack11-s2 by build alone.
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
CROOT = W / 'wt/stack11-s2-k'
EVAL = Path('/home/xanewok/repos/qwen38-eval')
CCX01 = W / 'work/aa-load/ccx01.sh'
COMPONENTS = 'p55 k7 k4 k6a k6b k10 k8 k11a k11b p90'.split()

WAVE1 = {
    'np1': ["MTP chain: k=5\\b", "mtp_fold_decode: engaged n_seq=1\\b",
            "set_logits: using backend candidates: n_cand = [1-9]\\d*, n_ban = [1-9]\\d*, top_k = 20\\b", "(?s)^(?!.*set_sampler: [^\\n]*using CPU)"],
    'np2': ["gdn_cache_fusion: graph device=0 graph=\\d+ tokens=6 seqs=2 indexed=([1-9]\\d*) gdn=\\1 fused=\\1\\b",
            "gdn_cache_fusion: graph device=1 graph=\\d+ tokens=6 seqs=2 indexed=([1-9]\\d*) gdn=\\1 fused=\\1\\b",
            "mtp_fold_decode: engaged n_seq=2\\b", "MTP chain disabled: needs one sequence \\(n_seq=2\\)", "(?s)^(?!.*MTP chain: k=)"],
    'duo': ["MTP chain disabled: needs one sequence \\(n_seq=2\\)", "(?s)^(?!.*MTP chain: k=)", "n_kv_pad += 4096\\b", "mtp_fold_decode: engaged n_seq=1\\b"],
}


def shape_of(text):
    m = re.search(r'^# shape: (np1|np2|duo)$', text, re.M)
    assert m, 'template lacks its "# shape:" line'
    return m.group(1)


def extra_pins(template_name):
    own = ['s2items.py', 'render_items.py', 'engagement.py', f'items/{template_name}']
    return [str(HERE / n) for n in own] + [str(CCX01)]


def fill(text, *, s2_sha, nok_sha, base_sha, components, evidence, template_name):
    shape = shape_of(text)
    env = engagement.canary_env(components, shape)
    patterns = WAVE1[shape] + engagement.canary_patterns(components, shape)
    base_log = engagement.BASE_P90 if shape == 'np1' else []
    values = {'@S2_SHA@': s2_sha, '@NOK_SHA@': nok_sha, '@BASE_SHA@': base_sha, '@TREE@': str(ROOT), '@CTREE@': str(CROOT),
              '@EVIDENCE@': str(evidence), '@EXTRA_PINS@': json.dumps(extra_pins(template_name)), '@CANARY_LOG@': json.dumps(patterns),
              '@BASE_CANARY_LOG@': json.dumps(base_log),
              '@CANARY_ENV@': ''.join(f'{k} = {json.dumps(v)}\n' for k, v in sorted(env.items())),
              '@COMPONENTS@': ' '.join(components)}
    for key, value in values.items():
        text = text.replace(key, value)
    assert not re.search(r'@[A-Z_0-9]+@', text), 'unfilled placeholder'
    return text


def selftest():
    import tomllib
    for p in sorted((HERE / 'items').glob('*.toml.in')):
        text = fill(p.read_text(), s2_sha='1' * 40, nok_sha='3' * 40, base_sha='2' * 40, components=COMPONENTS, evidence='/e', template_name=p.name)
        x = tomllib.loads(text)
        arms = x['arms']
        shape = shape_of(p.read_text())
        if 'nok' in arms:
            assert set(arms) == {'s2', 'nok'}
            assert arms['nok']['env'] == arms['s2']['env'] and arms['nok']['root'].endswith('/frozen/stack11-s2-nok')
            assert arms['s2']['root'].endswith('/frozen/stack11-s2') and arms['nok']['canary_log'] == []
        else:
            assert set(arms) == {'s11', 's2'} and arms['s11']['env'] == arms['s2']['env'], p.name
            assert arms['s11']['root'].endswith('/frozen/stack11-base') and arms['s2']['root'].endswith('/frozen/stack11-s2')
            assert arms['s11']['commit'] == '2' * 40 and arms['s2']['commit'] == '1' * 40
            assert arms['s2']['repo'] == str(CROOT) and arms['s11']['repo'].endswith('/wt/stack11-base')
            if shape == 'np1':
                assert arms['s11']['canary_log'] == engagement.BASE_P90 and x['canary']['plan'] == ['s11', 's2']
                assert x['depth']['identity'] == 'cross-arm' and x['depth']['depths'] == [8192, 32768]
            else:
                assert arms['s11']['canary_log'] == [] and x['canary']['plan'] == ['s2']
        assert any('CUDA_Host_Mapped' in q for q in arms['s2']['canary_log'] if 'CPU_Mapped' not in q) or 'nok' in arms
        assert x['out'].startswith(f'{EVAL}/results/2026-10-13-s2-') and x['id'] == p.name[:-len('.toml.in')]
        assert not any(a in str(x['plan']) for a in ('s10',))
        if x['kind'] == 'depth' and shape == 'np2':
            assert x['depth']['identity'] == 'off'
    print('s2items: templates fill, arms differ by build only, canaries per shape PASS')


if __name__ == '__main__':
    if sys.argv[1:] == ['selftest']:
        selftest()
    else:
        raise SystemExit(__doc__)
