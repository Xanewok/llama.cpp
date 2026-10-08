"""Emit shell functions and variable lines of the component windows, so the combined window runs each component's own gate code.

usage: shfn.py emit SCRIPT NAME... [--var NAME]... [--cut START_RE END_RE]  -> shell text on stdout
       shfn.py selftest                                                     -> every extraction used by gates.sh resolves and parses

A function is the block from `NAME() {` at column 0 to the next line that is exactly `}` (a one-line definition ends on its own line). A `--var` is the single line `NAME=...`
at column 0. `--cut` (once, applies to the last NAME) drops the body lines from the first line matching START_RE up to, not
including, the first later line matching END_RE; both must match, so a gate that changed shape fails closed.
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def function(text, name):
    lines = text.splitlines()
    start = [i for i, l in enumerate(lines) if re.match(rf'{re.escape(name)}\(\) \{{', l)]
    if len(start) != 1:
        raise SystemExit(f'shfn: {name}: {len(start)} definitions')
    if lines[start[0]].rstrip().endswith('}'):
        return [lines[start[0]]]
    end = next((i for i in range(start[0] + 1, len(lines)) if lines[i] == '}'), None)
    if end is None:
        raise SystemExit(f'shfn: {name}: no closing brace')
    return lines[start[0]:end + 1]


def variable(text, name):
    hits = [l for l in text.splitlines() if re.match(rf'{re.escape(name)}=', l)]
    if len(hits) != 1:
        raise SystemExit(f'shfn: variable {name}: {len(hits)} definitions')
    return hits[0]


def cut(body, start_re, end_re):
    a = next((i for i, l in enumerate(body) if re.search(start_re, l)), None)
    if a is None:
        raise SystemExit(f'shfn: --cut start /{start_re}/ not found')
    b = next((i for i in range(a + 1, len(body)) if re.search(end_re, body[i])), None)
    if b is None:
        raise SystemExit(f'shfn: --cut end /{end_re}/ not found after line {a}')
    return body[:a] + body[b:]


def emit(argv):
    script, rest = Path(argv[0]), argv[1:]
    text = script.read_text()
    out, names, i, window = [], [], 0, None
    while i < len(rest):
        if rest[i] == '--var':
            out.append(variable(text, rest[i + 1]))
            i += 2
        elif rest[i] == '--cut':
            window = (rest[i + 1], rest[i + 2])
            i += 3
        else:
            names.append(rest[i])
            i += 1
    for k, name in enumerate(names):
        body = function(text, name)
        if window and k == len(names) - 1:
            body = cut(body, *window)
        out.extend(body)
    return '\n'.join(out) + '\n'


def selftest():
    """Every extraction gates.sh uses, with the same arguments, and a `bash -n` of the result."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import gates_table
    n = 0
    for comp, script, args in gates_table.EXTRACTIONS:
        text = emit([str(ROOT / script), *args])
        with tempfile.NamedTemporaryFile('w', suffix='.sh') as f:
            f.write(text)
            f.flush()
            subprocess.run(['bash', '-n', f.name], check=True)
        n += 1
        print(f'{comp}: {script} {" ".join(args)}: {len(text.splitlines())} lines')
    print(f'shfn selftest PASS extractions={n}')


if __name__ == '__main__':
    if sys.argv[1:2] == ['emit']:
        sys.stdout.write(emit(sys.argv[2:]))
    elif sys.argv[1:] == ['selftest']:
        selftest()
    else:
        raise SystemExit(__doc__)
