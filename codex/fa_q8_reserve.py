"""Boot one served J5q np2 server, record its `compute buffer size` lines and stop it.

usage: fa_q8_reserve.py OUT.log EXE LIBDIR UB PORT [--draft-q8] [ENV=VALUE ...]
Flags are the swift15-j5q-duo profile (np2, 2 x 102400 cells, q8_0 KV, MTP draft, tensor split 50,50, kv-pad 4096). Exits 0 when
the server reached `all slots are idle`, 3 when it exited first (e.g. out of memory), 4 on timeout. The log lines are copied to
OUT.log.sizes as `MiB<TAB>line` in log order (main context first, draft context after it).
"""
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

MODEL = Path.home()/'models/Swift-1.5-Qwen3.8-27B-J5q-bf16.gguf'
VOCAB = Path.home()/'models/hyperqwen/Swift-1.5-Qwen3.8-27B-W4A16-syv-fast/swift_draft_vocab_ids.json'
TEMPLATE = Path.home()/'repos/qwen38-eval/corpora/templates/swift-leading-instructions-v1.jinja'
SIZE = re.compile(r'compute buffer size\s*=\s*([0-9.]+) MiB')


def main():
    args = sys.argv[1:]
    log, exe, lib, ub, port = Path(args[0]), args[1], args[2], args[3], args[4]
    rest = args[5:]
    draft_q8 = '--draft-q8' in rest
    extra_env = dict(a.split('=', 1) for a in rest if '=' in a)
    draft_type = 'q8_0' if draft_q8 else 'f16'
    cmd = [exe, '-m', str(MODEL), '-c', '204800', '-np', '2', '-ctk', 'q8_0', '-ctv', 'q8_0', '-ngl', '99', '-fa', 'on',
           '--spec-type', 'draft-mtp', '--spec-draft-n-max', '5', '--spec-draft-n-min', '0', '--spec-draft-p-min', '0',
           '--spec-draft-type-k', draft_type, '--spec-draft-type-v', draft_type, '-sm', 'tensor', '--tensor-split', '50,50',
           '-b', '2048', '-ub', ub, '--jinja', '--chat-template-file', str(TEMPLATE), '--temp', '1', '--top-k', '20',
           '--top-p', '0.95', '--min-p', '0.05', '--host', '127.0.0.1', '--port', port, '--metrics',
           '--spec-draft-sampling', 'probabilistic', '--spec-draft-backend-sampling', '--backend-sampling', '-t', '8', '-tb', '8',
           '--spec-draft-vocab', str(VOCAB)]
    env = dict(os.environ, LD_LIBRARY_PATH=lib, GGML_SCHED_INPUT_STAGING='1', LLAMA_ARG_KV_PAD='4096', LLAMA_MTP_CHAIN='1', **extra_env)
    rc = 4
    with open(log, 'w') as fh:
        proc = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.time() + 480
        try:
            while time.time() < deadline:
                if proc.poll() is not None:
                    rc = 3
                    break
                if 'all slots are idle' in log.read_text(errors='replace'):
                    rc = 0
                    break
                time.sleep(2)
        finally:
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)
                try:
                    proc.wait(60)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
    sizes = [(m.group(1), line) for line in log.read_text(errors='replace').splitlines() if (m := SIZE.search(line))]
    Path(str(log) + '.sizes').write_text(''.join(f'{mib}\t{line}\n' for mib, line in sizes))
    print(f'rc={rc} compute_buffers={[s[0] for s in sizes]}')
    return rc


if __name__ == '__main__':
    sys.exit(main())
