#!/usr/bin/env python3
"""CPU contract checks for capture, onset, interventions, and plan safety."""
import contextlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

import catch
import catch_replay as replay
import onset


def fixture(path):
    effective = dict(logit_bias=[], preserved_tokens=[], mirostat=0, adaptive_target=-1,
                     xtc_probability=0, grammar='', min_p=.05, temperature=1,
                     presence_penalty=0, repeat_last_n=64,
                     samplers=['top_k', 'top_p', 'min_p', 'temperature'])
    header = dict(event='request', version=1, parsed={}, effective=effective, target_seed=123,
                  budget=dict(tokens=-1, start=[], end=[], forced=[], control=0), replay_force=[])
    pieces = [(9, 'prelude \u00f3\n')]
    pieces += [(i + 1, piece) for _ in range(400) for i, piece in enumerate(['[', '0, ', '0, ', '0, ', '0],\n'])]
    events = [header, dict(event='prompt', ids=[10, 11]), dict(event='round', round=0, draft=[1], draft_q=[[dict(id=1, p=1)]])]
    for i, (ident, piece) in enumerate(pieces):
        probability = .1 if i == 1 else .9
        row = dict(top=[dict(id=ident, p=probability), dict(id=20, p=1 - probability)], omitted_mass=0, budget_state=0)
        events.append(dict(event='token', index=i, round=0, id=ident, piece_hex=piece.encode().hex(), target=row))
    events.append(dict(event='end', emitted=len(pieces)))
    path.write_text(''.join(json.dumps(e) + '\n' for e in events))
    return events


def main():
    with tempfile.TemporaryDirectory(dir=catch.ROOT / 'build-catch-check') as temporary:
        directory = Path(temporary)
        path = directory / 'capture.jsonl'
        events = fixture(path)
        header, _, tokens = replay.load(path)
        found = onset.find_onset(tokens)
        assert found['onset_token_M'] == 1 and found['period_tokens_M'] == 5, found
        result = onset.analyze(path, directory / 'analysis')
        assert result['entry']['sampled_target_p_M'] == .1
        assert result['entry']['repeat_target_p_M'] == .1
        assert result['entry']['min_p_cut_size_D'] == 2
        assert abs(result['entry']['entropy_nats_D'] + .1 * math.log(.1) + .9 * math.log(.9)) < 1e-12
        assert all(result['entry']['stage_survival_D'].values())
        assert result['after']['repeat_p_mean_D'] > .89
        control = onset.analyze(path, directory / 'matched', match=found)
        assert control['entry']['repeat_target_p_M'] == .1
        try:
            replay.body(path, 0, 8, 901)
        except ValueError:
            pass
        else:
            raise AssertionError('unsupported zero boundary accepted')
        for variant in catch.VARIANTS:
            request = catch.branch_request(path, 1, 8, 901, variant, 1e-6)
            assert request['__capture_force'] == [9]
            assert request['__capture_budget'] == header['budget']
            assert request['__capture_allow_drift'] == (variant != 'served')
        presence = catch.branch_request(path, 1, 8, 901, 'presence', 1e-6)
        assert presence['presence_penalty'] == 1 and presence['repeat_last_n'] == 102400
        assert catch.branch_request(path, 1, 8, 901, 'temp07', 1e-6)['temperature'] == .7
        # The analysis must reject a damaged probability row, not manufacture entropy.
        tokens[0]['target']['top'][0]['p'] = .01
        try:
            onset.distribution(tokens[0])
        except ValueError:
            pass
        else:
            raise AssertionError('broken normalized support was accepted')
        # A token perturbation leaves a low-diversity span, but changes the true period start.
        tokens[1000]['id'] = 99
        changed = onset.find_onset(tokens)
        assert not changed['exact_token_period'] or changed['onset_token_M'] != 1
        def forbidden(*_, **__):
            raise AssertionError('plan attempted process/HTTP/GPU work')
        with patch.dict(os.environ, RUN_PARENT_GPU='1', PLAN='1'), patch('sys.argv', ['catch.py']), \
                patch('subprocess.run', forbidden), patch('subprocess.Popen', forbidden), \
                patch('urllib.request.OpenerDirector.open', forbidden), contextlib.redirect_stdout(io.StringIO()) as output:
            catch.main()
        plan = json.JSONDecoder().raw_decode(output.getvalue())[0]
        assert plan['phase0_trials_D'] == 48 and plan['phase1_max_trials_D'] == 128
        assert plan['phase2_max_continuations_D'] == 1152
        with patch.dict(os.environ, RUN_PARENT_GPU='0', PLAN='0'):
            try:
                catch.gate()
            except RuntimeError:
                pass
            else:
                raise AssertionError('one-key gate was accepted')
        frozen = directory / 'frozen'
        frozen.mkdir()
        (frozen / 'capture-srv').write_bytes(b'capture')
        (frozen / 'llama-server').write_bytes(b'capture')
        pins = {'sha256': {p.name: catch.study.digest(p) for p in frozen.iterdir()}}
        with patch.object(catch, 'PINS', pins):
            catch.freeze(frozen, directory / 'copy')
            (frozen / 'capture-srv').write_bytes(b'broken')
            try:
                catch.freeze(frozen, directory / 'bad')
            except RuntimeError:
                pass
            else:
                raise AssertionError('changed frozen bytes accepted')
    print('PASS: exact onset, tail entry, entropy, controls, probability/period negative controls, replay roles, interventions, zero boundary, plan sentinels, GPU gate, frozen negative control')


if __name__ == '__main__':
    main()
