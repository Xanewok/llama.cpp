"""Model-level identity of embd-dev on Meta(CPU,CPU) (bw-embd-mac.sh stage 22). Usage: check_model.py DIR | --selftest"""
import json
import re
import sys
from pathlib import Path

ENGAGED = 'embd_rows: engaged node={node} backend=Meta\\S* shared={shared} type=\\S+ rows=[1-9]\\d* row_bytes=[1-9]\\d* mutant={m}'


def engaged(log, node, shared, mutant=0):
    return re.search(ENGAGED.format(node=node, shared=shared, m=mutant), log) is not None


def check(d):
    j = {p.stem: json.loads(p.read_text()) for p in d.glob('*.json')}
    log = {p.stem: p.read_text(errors='replace') for p in d.glob('*.log')}
    errors = []
    def need(ok, what):
        if not ok:
            errors.append(what)
    need(j['np2-patch'] == j['np2-off'], 'np2: patch and the old CPU path differ in tokens or counters')
    need(j['np1-chain-patch'] == j['np1-chain-off'], 'np1 chain: patch and the old CPU path differ')
    need(j['np2-row'] != j['np2-patch'], 'np2: the row mutant left tokens and counters unchanged')
    for arm in ('np2-patch', 'np1-chain-patch'):
        need(engaged(log[arm], '(?!mtp_)\\S+', 0), f'{arm}: target rows not on the meta backend')
        need(engaged(log[arm], 'mtp_tok_embd-\\d+', 0), f'{arm}: draft rows not on the meta backend')
    need(engaged(log['np2-patch'], 'mtp_tok_embd-\\d+', 1), 'np2-patch: fold rows not shared on the meta backend')
    need('mtp_fold_decode: engaged n_seq=2' in log['np2-patch'], 'np2-patch: fold not engaged')
    need(re.search('MTP chain: k=\\d+', log['np1-chain-patch']) is not None, 'np1-chain-patch: chain not engaged')
    need(engaged(log['np2-row'], '(?!mtp_)\\S+', 0, 2), 'np2-row: mutant not engaged')
    for arm in ('np2-off', 'np1-chain-off', 'np2-plain'):
        need('embd_rows:' not in log[arm], f'{arm}: device path engaged')
    return errors


def selftest():
    import tempfile
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        good = {'phases': [{'tokens': [[1, 2]]}]}
        line = 'W embd_rows: engaged node={} backend=Meta(CPU,CPU) shared={} type=q6_K rows=4 row_bytes=2100 mutant={}\n'
        logs = {
            'np2-patch': line.format('model.input_embed', 0, 0) + line.format('mtp_tok_embd-32', 0, 0) + line.format('mtp_tok_embd-32', 1, 0)
                         + 'mtp_fold_decode: engaged n_seq=2\n',
            'np1-chain-patch': line.format('model.input_embed', 0, 0) + line.format('mtp_tok_embd-32', 0, 0) + 'MTP chain: k=3\n',
            'np2-row': line.format('model.input_embed', 0, 2), 'np2-off': '', 'np1-chain-off': '', 'np2-plain': ''}
        for k, v in logs.items():
            (d / f'{k}.log').write_text(v)
            (d / f'{k}.json').write_text(json.dumps({'phases': [{'tokens': [[9]]}]} if k == 'np2-row' else good))
        assert check(d) == [], check(d)
        (d / 'np2-off.json').write_text(json.dumps({'phases': [{'tokens': [[1, 3]]}]}))
        (d / 'np2-plain.log').write_text(line.format('model.input_embed', 0, 0))
        assert len(check(d)) == 2, check(d)


if __name__ == '__main__':
    if sys.argv[1] == '--selftest':
        selftest()
        print('check_model: PASS selftest')
        sys.exit(0)
    errs = check(Path(sys.argv[1]))
    for e in errs:
        print('check_model: FAIL', e)
    print('check_model:', 'FAIL' if errs else 'PASS')
    sys.exit(1 if errs else 0)
