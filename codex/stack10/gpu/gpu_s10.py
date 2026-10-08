"""Stack10 A/B runner: the stack9/stack10 guards (frozen binaries, boot budget, scoped journal gate, untraced canaries, period report)
without gpu_stack10's exact-arm counter enforcement.

Contract: arms of the stack10 items are NOT exact (K-series kernels, p55 vs none), so unmatched np2 decode counters are recorded
per depth (`matched_counters`, `counters_equal_D`) and never void a run; an item with depth.counters keeps LLAMA_ROUND_TRACE in the
server env (the untraced guard of gpu_stack9 would strip it).
"""
import copy
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import tomllib

CODEX = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODEX))
sys.path.insert(0, str(HERE.parent))
import gpu_stack10 as s10
import s10items
import engagement
base = s10.base
lib, depth = base.lib, base.depth


def finish_recording(x, results):
    if depth.streams(x) > 1:
        for d, result in results.items():
            matched, counts = s10.counter_match(result['runs'], depth.arm_names(x))
            result['matched_counters'] = counts
            result['counters_equal_D'] = matched
    result = s10.original_finish(x, results)
    if depth.streams(x) > 1:
        for row in result.values():
            for arm in depth.arm_names(x):
                timings = [st['timings'] for r in row['runs'] if r['arm'] == arm for st in r['streams']]
                row[f'{arm}_tokens_per_round_D'] = 1 + depth.arm_k(x, arm)*sum(t['draft_n_accepted'] for t in timings)/sum(t['draft_n'] for t in timings)
    return result


def install(argv):
    lib.steady_intervals = s10.steady_with_periods
    depth.finish = finish_recording
    base.install_runtime_guards(argv)
    base.campaign.harness_cmd = lambda *args: [sys.executable, '-P', str(Path(__file__).resolve()), *map(str, args)]
    x = lib.read_experiment(argv[1])
    if x['kind'] == 'depth' and x['depth'].get('counters'):
        lib.server_env = base.original_env


def render_all(components, tmp):
    from harness import mkitem
    out = {}
    for source in sorted((HERE.parent/'items').glob('s10-*.toml.in')):
        text = s10items.fill(source.read_text(), sha='0'*40, base_sha='1'*40, components=components, tree=CODEX.parent,
                             evidence=Path(tmp)/'summary.json', template_name=source.name)
        path = Path(tmp)/source.name.removesuffix('.in')
        path.write_text(text)
        out[source.name] = (lib.read_experiment(path), mkitem, tomllib.loads(text))
    return out


def fixture_checks():
    s10.fixture_checks()
    from harness import mkitem
    wave1 = {  # served shape -> (wave-1 source, template)
        's10-depth-j5q.toml.in': CODEX/'items/s9p55-depth-j5q.toml.in', 's10-depth-iq4.toml.in': CODEX/'items/s9p55-depth-iq4.toml.in',
        's10-np2-real-j5q.toml.in': base.EVAL/'harness/experiments/s9p55-np2-real-j5q.toml',
        's10-np2-real-iq4.toml.in': CODEX/'items/s9p55-np2-real-iq4.toml.in',
        's10-duo-soak-j5q.toml.in': CODEX/'items/stack9-duo-soak-j5q.toml.in', 's10-duo-soak-iq4.toml.in': CODEX/'items/s9p55-duo-soak-iq4.toml.in'}
    with tempfile.TemporaryDirectory(dir=HERE.parent) as tmp:
        for components in (s10items.COMPONENTS, ['k7', 'k4']):
            rendered = render_all(components, tmp)
            assert sorted(rendered) == sorted(wave1), sorted(rendered)
            for name, (x, _, raw) in rendered.items():
                old = tomllib.loads(wave1[name].read_text().replace('@S9P55_SHA@', '0'*40).replace('@STACK9_SHA@', '0'*40))
                shape = s10items.shape_of((HERE.parent/'items'/name).read_text())
                # the same server, model, depth/campaign settings as the wave-1 item of this profile
                for section in ('model', 'draft_vocab'):
                    assert {k: str(v) for k, v in x[section].items()} == {k: str(v) for k, v in old[section].items()}, f'{name}: {section} differs from wave 1'
                assert [a for a in x['server']['args'] if a != '{cache_ram_flags}'] == [a for a in old['server']['args'] if a != '{cache_ram_flags}'], f'{name}: served flags differ from wave 1'
                assert set(old['server']['knobs']) <= set(x['server']['knobs'])
                assert set(s10items.SWITCH_ENVS) <= set(x['server']['knobs'])
                assert all(k in x['server']['knobs'] for c in engagement.ENGAGEMENT.values() for k in c['env'])
                if x['kind'] == 'depth':
                    for key in ('depths', 'k', 'streams', 'slot_ctx', 'wall_s', 'min_steady_intervals', 'min_predicted'):
                        assert x['depth'][key] == old['depth'][key], f'{name}: depth.{key}'
                    assert x['depth']['request']['ignore_eos'] is True and x['depth']['request'] == old['depth']['request']
                    assert x['depth'].get('decode_start', 'joint') == 'joint'
                    assert (x['depth']['identity'] == 'cross-arm') == (shape == 'np1'), f'{name}: identity mode'
                    assert bool(x['depth'].get('counters')) == (name == 's10-np2-real-j5q.toml.in'), f'{name}: round trace only on the np2 J5q item'
                else:
                    for key in ('agents', 'agent_ctx', 'tasks', 'wall_s', 'task_timeout_s', 'runaway_tokens'):
                        assert x['campaign'][key] == old['campaign'][key], f'{name}: campaign.{key}'
                assert '--cache-ram' not in x['server']['args']
                assert list(x['arms']) == ['s9p', 's10']
                a, b = x['arms']['s9p'], x['arms']['s10']
                strip = lambda e: {k: v for k, v in e.items() if k != 'LD_LIBRARY_PATH'}  # per-arm root
                assert strip(a['env']) == strip(b['env']), f'{name}: candidate arm env differs from the base arm (no switch allowed)'
                assert a['flags'] == b['flags'] == []
                assert a['canary_log'] == [] and x['canary']['plan'] == ['s10']
                assert x['plan'].count('s9p') == x['plan'].count('s10') == (3 if name == 's10-np2-real-j5q.toml.in' else 2)
                assert x['plan'][0] != x['plan'][1] and x['plan'][-1] != x['plan'][-2], f'{name}: boots not interleaved'
                assert type(x['priority']) is int and 26 <= x['priority'] <= 28 and x['budget']['canary_s'] <= 600
                assert len(x['plan'])*mkitem.boot_wall(x) + x['budget']['canary_s'] <= 0.7*mkitem.budget(x)['budget_s']
                wave1_log = old['arms'][list(old['arms'])[-1]]['canary_log']
                assert wave1_log == s10items.WAVE1[shape] and b['canary_log'][:len(wave1_log)] == wave1_log, f'{name}: wave-1 canary patterns drifted'
                for pattern in b['canary_log']:
                    re.compile(pattern)
                env = x['canary'].get('env', {})
                assert {k: v for k, v in env.items() if k in s10items.SWITCH_ENVS} == engagement.canary_env(components, shape), f'{name}: canary log envs'
                assert all(k in x['server']['knobs'] for k in env), f'{name}: canary env outside knobs'
                assert any('mmq_tiles' in p for p in b['canary_log']) == (shape == 'np2' and 'k8' in components)
                assert any('ssm_conv_rs engaged' in p for p in b['canary_log']) == ('k4' in components)
                assert x['cpu_evidence']['path'] == str(Path(tmp)/'summary.json')
    window = tomllib.loads((HERE.parent/'bw-stack10.toml').read_text())
    assert window['kind'] == 'filler' and window['priority'] == 19 and window['host_sensitive'] is False and window['preemptible'] is False
    assert window['run']['env']['CUDA_VISIBLE_DEVICES'].startswith('GPU-') and window['run']['ports'] == 'dynamic'
    print('stack10 item templates: wave-1 flags and env kept, candidate arm without switches, engagement canaries, window fields PASS')


if __name__ == '__main__':
    if sys.argv[1:] == ['--fixtures']:
        sys.exit(fixture_checks())
    install(sys.argv[1:])
    sys.exit(base.harness_main())
