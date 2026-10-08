"""Early-exit consumers under pipefail: in `producer | grep -q X` grep exits at the first match, the producer's next write
gets SIGPIPE and the pipeline returns 141, so a presence check fails although X is there. Whether it fires depends on
output size and timing, so a passing run proves nothing. `python3 -m gpuq.shlint FILE...` lints files directly."""
from pathlib import Path
import re, sys

EXEMPT = re.compile(r'#\s*shlint:\s*ok\s+\S')
PREFIX = {'!', 'if', 'then', 'elif', 'else', 'do', 'while', 'until', 'time', '{', 'command', 'builtin', 'exec', 'env',
          'nice', 'nohup', 'stdbuf', 'timeout', 'taskset'}
SAFE_GREP = '`grep -q PAT FILE` (no pipe) or `| grep -c PAT >/dev/null` (reads all input)'
HINTS = {'grep-q': SAFE_GREP, 'grep-m': "`| grep PAT | sed -n '1,Np'` (reads all input)",
         'head': "`| sed -n '1,Np'` (reads all input)", 'sed': "`sed -n 'Np'` without q (reads all input)",
         'awk': 'a flag instead of `exit` (reads all input)'}
SED_QUIT = re.compile(r'(?:^|[;{}\s\d$/])[qQ]\s*\d*\s*(?:[;}]|$)')

def dquote_end(text, i):
    i += 1
    while i < len(text) and text[i] != '"':
        if text[i] == '\\': i += 1
        elif text.startswith('$(', i): i = close(text, i + 2)
        elif text[i] == '`': i = tick_end(text, i)
        i += 1
    return i

def tick_end(text, i):
    i += 1
    while i < len(text) and text[i] != '`': i += 2 if text[i] == '\\' else 1
    return i

def close(text, i, o='(', c=')'):
    """Index of the bracket closing the one opened just before i."""
    depth = 1
    while i < len(text):
        ch = text[i]
        if ch == '\\': i += 1
        elif ch == "'":
            i = text.find("'", i + 1)
            if i < 0: return len(text)
        elif ch == '"': i = dquote_end(text, i)
        elif ch == o: depth += 1
        elif ch == c:
            depth -= 1
            if not depth: return i
        i += 1
    return len(text)

def pipelines(text, line=1):
    """(stages, first_line, last_line) per pipeline in source order; a stage is (line, words). Not a parser: quoting,
    substitutions, [[ ]], comments and heredocs are tracked well enough that a `|` inside them is not taken for a pipe.
    Process substitutions are skipped: the shell ignores their status. Words keep no quote characters."""
    out, stages, words, heredocs = [], [], [], []
    word, sline, i, n, depth = None, line, 0, len(text), 0

    def add(s):
        nonlocal word, sline
        if word is None:
            if not words: sline = line
            word = ''
        word += s
    def flush():
        nonlocal word
        if word is not None: words.append(word)
        word = None
    def stage():
        nonlocal words
        flush()
        if words: stages.append((sline, words))
        words = []
    def end():
        nonlocal stages
        stage()
        if stages: out.append((stages, stages[0][0], line))
        stages = []
    def raw(k):
        nonlocal line, i
        add(text[i:k]); line += text.count('\n', i, k); i = k
    def substitution(j, k):
        nonlocal line, i
        out.extend(pipelines(text[j:k], line)); add('$()'); line += text.count('\n', i, k); i = k + 1

    while i < n:
        c, nxt = text[i], text[i + 1:i + 2]
        if c == '\\':
            if nxt == '\n': line += 1
            else: add(text[i:i + 2])
            i += 2
        elif c == "'":
            k = text.find("'", i + 1)
            k = n if k < 0 else k
            i += 1; raw(k); i += 1
        elif c == '"':
            add(''); i += 1
            while i < n and text[i] != '"':
                if text.startswith('$((', i): raw(close(text, i + 2) + 1)
                elif text.startswith('$(', i): substitution(i + 2, close(text, i + 2))
                elif text[i] == '`': substitution(i + 1, tick_end(text, i))
                else: raw(i + (2 if text[i] == '\\' else 1))
            i += 1
        elif text.startswith('$((', i): raw(close(text, i + 2) + 1)
        elif text.startswith('$(', i): substitution(i + 2, close(text, i + 2))
        elif text.startswith('${', i): raw(close(text, i + 2, '{', '}') + 1)
        elif c == '`': substitution(i + 1, tick_end(text, i))
        elif c == '#' and word is None:
            k = text.find('\n', i)
            i = n if k < 0 else k
        elif c in ' \t':
            flush(); i += 1
        elif c == '\n':
            flush()
            if not (stages and not words): end()  # a newline right after `|` continues the pipeline
            i += 1; line += 1
            for delim, tabs in heredocs:
                while i < n:
                    k = text.find('\n', i)
                    k = n if k < 0 else k
                    body, i, line = text[i:k], k + 1, line + 1
                    if (body.lstrip('\t') if tabs else body) == delim: break
            heredocs = []
        elif c == '|':
            if nxt == '|': end(); i += 2
            elif word is not None and word.endswith('>'): add(c); i += 1
            else: stage(); i += 2 if nxt == '&' else 1
        elif c == '&':
            if nxt == '&': end(); i += 2
            elif nxt == '>' or (word is not None and word.endswith(('>', '<'))): add(c); i += 1
            else: end(); i += 1
        elif c == ';':
            end(); i += 1
        elif c == '(':
            if word is not None and word.endswith(('<', '>')): raw(close(text, i + 1) + 1)
            elif nxt == '(' and word is None: raw(close(text, i + 1) + 1)
            else: end(); depth += 1; i += 1
        elif c == ')':
            if depth: end(); depth -= 1
            else: stages, words, word = [], [], None  # an unopened `)` closes a case pattern: `a|b)` is no pipe
            i += 1
        elif text.startswith('<<<', i):
            add('<<<'); i += 3
        elif text.startswith('<<', i):
            j = i + 2
            tabs = text[j:j + 1] == '-'
            j += tabs
            while j < n and text[j] in ' \t': j += 1
            m = re.match(r"'([^']*)'|\"([^\"]*)\"|([^\s;|&<>()]+)", text[j:])
            delim = '' if not m else next(g for g in m.groups() if g is not None).replace('\\', '')
            if delim: heredocs.append((delim, tabs))
            add(text[i:j] + delim); i = j + (m.end() if m else 0)
        elif word is None and text.startswith('[[', i) and text[i + 2:i + 3] in (' ', '\t', '\n'):
            k = text.find(']]', i + 2)
            raw(n if k < 0 else k + 2)
        else:
            add(c); i += 1
    end()
    return out

def grep_flag(args, early, takes_arg):
    skip = False
    for a in args:
        if skip: skip = False
        elif a == '--': break
        elif a.startswith('--'):
            name = a[2:].split('=')[0]
            if name in ('quiet', 'silent', 'files-with-matches', 'files-without-match'): return 'grep-q'
            if name == 'max-count': return 'grep-m'
        elif a.startswith('-'):
            for k, ch in enumerate(a[1:], 2):
                if ch in early: return 'grep-m' if ch == 'm' else 'grep-q'
                if ch in takes_arg:  # the argument is the rest of this word, else the next word
                    skip = k == len(a)
                    break
    return None

def consumer(words):
    """HINTS key when this stage may exit before draining its stdin."""
    for k, w in enumerate(words):
        name = w.rsplit('/', 1)[-1]
        if name in ('grep', 'egrep', 'fgrep', 'zgrep', 'rg', 'head', 'sed', 'gsed', 'awk', 'gawk', 'mawk', 'nawk'): break
        if not (w in PREFIX or w.startswith('-') or re.fullmatch(r'[A-Za-z_]\w*=.*|[\d.]+[smhd]?|[\d,-]+', w)): return None
    else: return None
    args = words[k + 1:]
    if name == 'rg': return grep_flag(args, 'qlm', 'efABCdEgjMtT')
    if name.endswith('grep'): return grep_flag(args, 'qlLm', 'efABCdD')
    if name == 'head':
        tail = ' '.join(args)
        return None if re.search(r'(?:^|\s)-[nc]\s*-\d|--(?:lines|bytes)=-', tail) else 'head'
    if name.endswith('sed'):
        return 'sed' if any(SED_QUIT.search(a) for a in args if not a.startswith('-') or a.startswith('-e')) else None
    return 'awk' if any(re.search(r'\bexit\b', a) for a in args) else None

def sets_pipefail(words):
    return words[0] == 'set' and any(w == 'pipefail' and re.fullmatch(r'-[A-Za-z]*o', words[k - 1]) for k, w in enumerate(words) if k)

def shebang_pipefail(text):
    return text.startswith('#!') and 'pipefail' in text.split('\n', 1)[0]

def enables_pipefail(text):
    return shebang_pipefail(text) or any(sets_pipefail(stages[0][1]) for stages, _, _ in pipelines(text))

def lint(text, pipefail=False):
    """(line, stage, hint) per early-exit consumer after the first stage of a pipeline that runs with pipefail on;
    `pipefail` is the state inherited at line 1 (a sourced library)."""
    lines, found = text.split('\n'), []
    pipefail = pipefail or shebang_pipefail(text)
    for stages, first, last in pipelines(text):
        pipefail = pipefail or sets_pipefail(stages[0][1])
        if not pipefail or any(EXEMPT.search(l) for l in lines[first - 1:last]): continue
        for line, words in stages[1:]:
            kind = consumer(words)
            if kind: found.append((line, ' '.join(words)[:80], HINTS[kind]))
    return found

def lint_file(path, inherit=False):
    """Contract details `path:line: ...`; an unreadable file yields none (pins validation owns missing files).
    With `inherit`, a file without a shebang runs under pipefail from line 1: it is sourced by a pipefail script."""
    try: text = Path(path).read_text(errors='replace')
    except OSError: return []
    pipefail = inherit and not text.startswith('#!')
    return [f'{path}:{line}: `{stage}` can exit before draining its stdin: under pipefail the producer gets SIGPIPE '
            f'and the pipeline returns 141; safe: {hint}; else append `# shlint: ok <reason>`'
            for line, stage, hint in lint(text, pipefail)]

if __name__ == '__main__':
    found = [d for p in sys.argv[1:] for d in lint_file(p)]
    for d in found: print(d)
    sys.exit(1 if found else 0)
