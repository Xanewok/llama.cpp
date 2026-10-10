"""Extract the exact private read-only tensor map layout from a pinned engine source."""
import argparse
import hashlib
from pathlib import Path
import subprocess


def extract(source, name):
    start = source.index('struct ' + name + ' {')
    depth = 0
    for i in range(source.index('{', start), len(source)):
        depth += (source[i] == '{') - (source[i] == '}')
        if depth == 0:
            return source[start:i + 2]
    raise ValueError('unterminated struct')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--engine', type=Path, required=True)
    p.add_argument('--commit', required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    source = subprocess.check_output(['git', '-C', str(args.engine), 'show',
                                     args.commit + ':ggml/src/ggml-backend-meta.cpp'], text=True)
    bodies = '\n'.join(extract(source, name) for name in
                       ('ggml_backend_meta_simple_tensor_container', 'ggml_backend_meta_buffer_context'))
    args.out.write_text('#pragma once\n' + bodies + '\n')
    print('private_meta_layout_sha256=' + hashlib.sha256(bodies.encode()).hexdigest())


if __name__ == '__main__':
    main()
