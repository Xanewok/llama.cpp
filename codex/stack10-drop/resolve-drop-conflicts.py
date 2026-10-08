#!/usr/bin/env python3
"""Resolve diff3 conflicts that appear when a rebase drops a commit whose added lines sit next to a later commit's additions.

Per conflict region: ours = base minus the dropped lines, theirs = base plus the replayed commit's lines. The merge is theirs minus the
base lines that ours lacks. A region where ours is not a pure deletion of base lines (the dropped commit edited rather than added
lines) is left in place and the exit status is 1.
"""
import difflib
import os
import re
import sys


TILES = ', tiles, prec_src1>'
PLAIN = ', prec_src1>'


def kernel_launch_rule(ours, base, theirs):
    """K9/K8 coupling in mmq.cuh: the replayed commit edits launch lines whose syntax the dropped commit changed (`<<<>>>` vs
    ggml_cuda_kernel_launch) or whose template list carries the `tiles` argument of the L1/K8 commit."""
    comps = set(os.environ.get('DROP_COMPONENT', '').replace(',', ' ').split())
    jt = '\n'.join(theirs)
    jb = '\n'.join(base)
    jo = '\n'.join(ours)
    if 'k9' in comps and 'k8' not in comps and '<<<' in jo and 'ggml_cuda_kernel_launch(' in jb and TILES in jt and TILES not in jb:
        return [l.replace('fallback' + PLAIN, 'fallback' + TILES) for l in ours]
    if 'k8' in comps and 'k9' not in comps and TILES in jt and TILES not in jo:
        return [l.replace('fallback' + TILES, 'fallback' + PLAIN) for l in theirs]
    return None


def resolve(ours, base, theirs):
    special = kernel_launch_rule(ours, base, theirs)
    if special is not None:
        return special
    sm = difflib.SequenceMatcher(None, base, ours, autojunk=False)
    gone = set()
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'delete':
            gone.update(range(i1, i2))
        elif tag != 'equal':
            return None
    sm2 = difflib.SequenceMatcher(None, base, theirs, autojunk=False)
    keep_theirs = [True] * len(theirs)
    mapped = 0
    for a, b, size in sm2.get_matching_blocks():
        for k in range(size):
            if a + k in gone:
                keep_theirs[b + k] = False
                mapped += 1
    if mapped != len(gone):
        return None
    return [l for l, keep in zip(theirs, keep_theirs) if keep]


CHAIN_FUNCS = ['run_alpha_gate_identity', 'run_ssm_conv_rs_identity', 'run_rms_norm_scale_pair_identity', 'run_fattn_shared_mask_identity']


def normalize_identity_chain(path):
    """The return of test_backend() chains one identity check per component, so every component edits the same lines. Rebuild the
    block from the run_*_identity functions the file defines; for the full stack this reproduces the committed text."""
    text = open(path, encoding='utf-8', errors='surrogateescape').read()
    a = text.find('        output_printer->print_failed_tests(failed_tests);\n\n        ')
    end = text.find('    if (mode == MODE_GRAD) {')
    if a < 0 or end < 0 or end < a:
        return
    defined = lambda f: re.search(r'^static bool ' + f + r'\(', text, re.M) is not None
    call = lambda f: f'{f}(backend, op_names_filter)'
    out = ['        output_printer->print_failed_tests(failed_tests);', '']
    if defined('run_shared_activation_identity'):
        out += [f'        if (!{call("run_shared_activation_identity")}) {{', '            return false;', '        }', '']
    out += ['        const bool slice_ok = run_fa_vec_slice(backend, backend_cpu.get(), op_names_filter);', '']
    if defined('run_add_rms_norm_identity'):
        out += [f'        if (!{call("run_add_rms_norm_identity")}) {{', '            return false;', '        }', '']
    out += [f'        const bool alpha_ok = {call("run_alpha_gate_identity")};']
    conv = defined('run_ssm_conv_rs_identity')
    if conv:
        out += [f'        const bool conv_ok  = {call("run_ssm_conv_rs_identity")};']
    chain = [c for c in ('run_rms_norm_scale_pair_identity', 'run_fattn_shared_mask_identity') if defined(c)]
    ret = '        return n_ok == tests_run && slice_ok && alpha_ok' + (' && conv_ok' if conv else '')
    out += [ret + (' &&' if chain else ';')]
    for k, c in enumerate(chain):
        out += [f'               {call(c)}' + (' &&' if k + 1 < len(chain) else ';')]
    out += ['    }', '']
    open(path, 'w', encoding='utf-8', errors='surrogateescape').write(text[:a] + '\n'.join(out) + '\n' + text[end:])


def main(path):
    if path.endswith('tests/test-backend-ops.cpp'):
        normalize_identity_chain(path)
    text = open(path, encoding='utf-8', errors='surrogateescape').read().split('\n')
    out, i, ok = [], 0, True
    while i < len(text):
        if not text[i].startswith('<<<<<<< '):
            out.append(text[i])
            i += 1
            continue
        j = i + 1
        sect = {'ours': [], 'base': [], 'theirs': []}
        cur = 'ours'
        while not text[j].startswith('>>>>>>> '):
            if text[j].startswith('||||||| '):
                cur = 'base'
            elif text[j] == '=======':
                cur = 'theirs'
            else:
                sect[cur].append(text[j])
            j += 1
        res = resolve(sect['ours'], sect['base'], sect['theirs'])
        if res is None:
            ok = False
            out.extend(text[i:j + 1])
        else:
            out.extend(res)
        i = j + 1
    open(path, 'w', encoding='utf-8', errors='surrogateescape').write('\n'.join(out))
    return ok


if __name__ == '__main__':
    if sys.argv[1] == '--normalize':
        normalize_identity_chain(sys.argv[2])
        sys.exit(0)
    sys.exit(0 if all([main(p) for p in sys.argv[1:]]) else 1)
