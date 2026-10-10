"""Build an unmasked-tail mutant with the CPU-only target's exact compiler/linker recipe."""
import json
import os
from pathlib import Path
import shlex
import subprocess

root = Path(__file__).resolve().parents[1]
source = root / 'tests/test-backend-ops.cpp'
mutant = root / 'tests/test-backend-ops-padinv-mutant.cpp'
obj = root / 'build-cpu/padinv-mutant.o'
binary = root / 'build-cpu/bin/padinv-mask-mutant'
buildgate = '/home/xanewok/repos/qwen38-eval-worktrees/bin/buildgate'
text = source.read_text()
needle = 'std::vector<ggml_fp16_t> masks(kv, ggml_fp32_to_fp16(-INFINITY));'
assert text.count(needle) == 1
mutant.write_text(text.replace(needle, 'std::vector<ggml_fp16_t> masks(kv, ggml_fp32_to_fp16(0));'))
try:
    entry = next(c for c in json.loads((root / 'build-cpu/compile_commands.json').read_text()) if c['file'] == str(source))
    compile_args = shlex.split(entry['command'])
    compile_args[compile_args.index('-o') + 1] = str(obj)
    compile_args[-1] = str(mutant)
    link_args = shlex.split((root / 'build-cpu/tests/CMakeFiles/test-backend-ops.dir/link.txt').read_text())
    link_args[link_args.index('-o') + 1] = str(binary)
    link_args = [str(obj) if a.endswith('test-backend-ops.cpp.o') else a for a in link_args]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', GIT_NO_LAZY_FETCH='1', OMP_NUM_THREADS='1')
    for command in (compile_args, link_args):
        subprocess.run([buildgate, *command], cwd=entry['directory'], env=env, check=True)
    evidence = root / 'codex/padinv-evidence'
    command = [str(binary), 'test', '-b', 'CPU', '-o', 'FLASH_ATTN_EXT', '-p',
               'padding=1,.*nh=2,.*visible=8193,.*q_scale=1,head_major=0$', '--output', 'sql', '-j', '1']
    with (evidence / 'mutant.sql').open('w') as out, (evidence / 'mutant.err').open('w') as err:
        result = subprocess.run(command, env=env, stdout=out, stderr=err, timeout=60)
    from padinv_summary import sql_rows, evidence as validate
    rows = sql_rows(evidence / 'mutant.sql')
    row = next(r for r in rows if r['op_params'].startswith('padding=1,'))
    data = validate(row)
    assert result.returncode == 1 and not data['invariant'] and data['finite'] and data['compute']
    (evidence / 'mutant-check.json').write_text(json.dumps({'returncode_M': result.returncode, 'invariant_M': data['invariant'],
        'control_M': data['control'], 'finite_M': data['finite'], 'mutation': 'all suffix mask entries set to zero'}, indent=2) + '\n')
finally:
    mutant.unlink(missing_ok=True)
