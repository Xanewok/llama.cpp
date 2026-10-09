"""Use the stack10 runtime guards, then record every switch arm against the archived reference."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'gpu'))
import gpu_s10
import compare


def main():
    argv = sys.argv[1:]
    gpu_s10.install(argv)
    rc = gpu_s10.base.harness_main()
    if '--execute' in argv:
        out = argv[argv.index('--out') + 1]
        diagnostic_rc = compare.analyze(argv[1], out)
        return rc or diagnostic_rc
    return rc


if __name__ == '__main__':
    sys.exit(main())
