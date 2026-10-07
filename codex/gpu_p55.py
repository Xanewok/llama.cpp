"""Stack7 runtime guards plus a saved proof of the running frozen executable."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpu_stack7
from harness import lib


def install_binary_proof(argv):
    x = lib.read_experiment(argv[1])
    original_start = lib.Server.start

    def start(server):
        exe = Path(server.argv[0]).resolve()
        arm = next((a for a in x['arms'].values() if a['exe'].resolve() == exe), None)
        if arm is None:
            raise lib.Fail('frozen bytes changed: unknown executable', 'integrity')
        files = lib.check_frozen(arm)
        pid = original_start(server)
        running = Path(f'/proc/{pid}/exe').resolve()
        if running != exe:
            raise lib.Fail('frozen bytes changed: running executable differs', 'integrity')
        proof = dict(arm=arm['name'], source=arm['commit'], exe=str(running), sha256=files[str(arm['exe'])])
        (server.log.parent / 'binary-proof.json').write_text(json.dumps(proof, indent=2) + '\n')
        print('frozen_binary_proof: ' + json.dumps(proof), flush=True)
        return pid

    lib.Server.start = start


if __name__ == '__main__':
    if sys.argv[1:] == ['--fixtures']:
        sys.exit(gpu_stack7.fixture_checks())
    gpu_stack7.install_runtime_guards(sys.argv[1:])
    install_binary_proof(sys.argv[1:])
    sys.exit(gpu_stack7.harness_main())
