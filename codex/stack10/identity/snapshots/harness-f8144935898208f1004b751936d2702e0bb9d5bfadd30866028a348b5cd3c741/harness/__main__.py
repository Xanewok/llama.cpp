"""python -m harness <campaign|depth|probe> experiment.toml ...; report RUN_DIR; bw-item PAYLOAD.sh."""
import argparse
import os
import sys
from pathlib import Path

from . import lib


DESCRIPTION = """One parametrized harness per measurement shape: every experiment fact lives in experiment.toml, every gate in
lib.py. kind: campaign = Codex task campaigns per frozen arm (agents = 2: two agents sharing one server); depth = interleaved
fresh boots, one forced-length request per depth (streams = 2: two concurrent slots); probe = one boot per arm plus the
experiment's own driver. Exit codes: 3 hazard, 4 transient, 5 insufficient data, 6 integrity, 7 timeout, 8 deterministic."""
EPILOG = """typical sequence (the experiment's keys are documented in the docstrings of depth.py, agents.py and campaign.py):
  bash run.sh KIND experiments/X.toml --preflight --out OUT                     CPU: frozen bytes, model, launch contract
  python3 -m harness.mkitem experiments/X.toml --out X-item.toml                 write the gpuq item
  ~/repos/gpuq/bin/gpuq submit --check X-item.toml                               validate it (submit only on go-ahead)
  GPUQ_ACTIVE=1 bash run.sh KIND experiments/X.toml --canary --out OUT-canary    GPU proofs, <= 15 min (through gpuq)
  GPUQ_ACTIVE=1 bash run.sh KIND experiments/X.toml --execute --out OUT          the measurement (through gpuq)
  bash run.sh campaign experiments/X.toml --summarize OUT                        rebuild summary.md/json
  bash run.sh depth experiments/X.toml --reanalyze OUT                           recompute summary.json from records
  python3 -m harness report RUN_DIR [--json] [--aa]                              CPU: gates, boot SE, notes/ledger block
  python3 -m harness.mkitem experiments/X.toml --submit                         render, gpuq submit --check, submit on success
  python3 -m harness report RUN_DIR --record --lever L --axis A --cause C --relation R --verdict V
                                                                              append gated notes, INDEX and ledger draft
  python3 -m harness bw-item codex/bw-X.sh --priority N --out-root WINDOW        preflight, write external build-window item"""


def build_parser():
    p = argparse.ArgumentParser(prog="harness", description=DESCRIPTION, epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("kind", choices=("campaign", "depth", "probe"), help="measurement shape; must equal the experiment's own kind")
    p.add_argument("experiment", type=Path, help="experiment.toml: arms, server contract, workload, budget (the only place facts live)")
    stage = p.add_mutually_exclusive_group(required=True)
    helps = {"--execute": "run the measurement (needs GPUQ_ACTIVE=1: start it through gpuq)",
             "--canary": "short GPU proof run: identity across arms, knobs engaged, every parser and gate exercised",
             "--capture": "CPU only: write the launch commands/envs the run would use, start nothing",
             "--preflight": "CPU only: frozen bytes, pins, model digest, launch contract; the plan is not run",
             "--check-queue": "campaign internal: validate one arm's runner queue and engine commit (with --arm)",
             "--child": "campaign internal: run one arm's campaign as the orchestrator's child process (with --arm)"}
    for flag, text in helps.items():
        stage.add_argument(flag, action="store_true", help=text)
    stage.add_argument("--reanalyze", type=Path, metavar="RUN_DIR", help="depth: recompute summary.json from the stored records of RUN_DIR")
    stage.add_argument("--summarize", type=Path, metavar="RUN_DIR", help="campaign: rebuild summary.md/json from the campaign dirs of RUN_DIR")
    p.add_argument("--out", type=Path, help="output directory (fresh for --execute/--canary; the run's records and verdicts go here)")
    p.add_argument("--port", type=int, default=int(os.environ.get("GPUQ_PORT") or 0), help="server port (default $GPUQ_PORT, else the experiment's server.port, else a free one)")
    p.add_argument("--budget-s", type=float, help="campaign: wall budget of the whole run in seconds (default budget.run_s of the experiment)")
    p.add_argument("--stub", action="store_true", help="CPU stub server in place of the frozen binary (tests); never a GPU")
    p.add_argument("--fixtures", type=Path, help="depth --preflight: directory of recorded real-artifact fixtures (default depth.fixtures)")
    p.add_argument("--arm", help="campaign internal (--child, --capture, --check-queue): the arm to act on")
    p.add_argument("--task", help="campaign --capture: the task whose launch is captured (default campaign.canary_task)")
    p.add_argument("--fit", action="store_true", help="campaign --capture: capture the capacity-fit launch instead of the task launch")
    p.add_argument("--deadline-epoch", type=float, help="campaign internal (--child): epoch seconds after which no new work starts")
    p.add_argument("--probe-timeout", type=int, default=0, help="campaign internal (--child): override the agentic probe's --timeout, seconds (0 = keep)")
    p.add_argument("--max-attempts", type=int, default=0, help="campaign internal (--child): override the runner's attempts per job (0 = keep)")
    return p


def parse(argv):
    return build_parser().parse_args(argv)


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    if argv and argv[0] == "bw-item":
        from . import bw_item
        return bw_item.main(argv[1:])
    if argv and argv[0] == "report":
        from . import report
        return report.main(argv[1:])
    a = parse(argv)
    out = a.out.resolve() if a.out else None

    def run():
        x = lib.read_experiment(a.experiment)
        if x["kind"] != a.kind:
            raise lib.Fail(f"inputs changed: {a.experiment.name} is kind {x['kind']}, not {a.kind}", "integrity")
        port = a.port or x["server"].get("port") or lib.free_port()
        budget = a.budget_s or x["budget"].get("run_s", 0)
        if a.kind == "campaign":
            from . import campaign
            a.port = port
            a.budget_s = budget
            return campaign.main(x, a)
        from . import depth, probe
        mod = depth if a.kind == "depth" else probe
        if a.capture:
            return mod.capture(x, out, port)
        if a.reanalyze:
            import json
            print(json.dumps(depth.reanalyze(x, a.reanalyze), indent=2))
            return 0
        if a.preflight:
            return depth.preflight(x, out, a.fixtures or x["depth"].get("fixtures")) if a.kind == "depth" else probe_preflight(x, out)
        if a.canary or a.execute:
            return depth.execute(x, out, port, canary=a.canary, stub=a.stub) if a.kind == "depth" else probe.execute(x, out, port, a.stub)
        raise lib.Fail(f"{a.kind} has no such stage", "deterministic")

    return lib.run_main(run, out if a.execute or a.canary or a.preflight else None)


def probe_preflight(x, out):
    for arm in x["arms"].values():
        lib.check_frozen(arm)
    lib.check_model(x["model"], full=True)
    lib.log(out, f"preflight ok: plan {' '.join(x['plan'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
