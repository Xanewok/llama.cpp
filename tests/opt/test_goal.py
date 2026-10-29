from pathlib import Path

import pytest

from scripts.opt import goal

ROOT = Path(__file__).resolve().parents[2]
BASE = '[goal]\nworkload = "w"\nprimary = "wall_est"\n'


@pytest.mark.parametrize("f", sorted((ROOT / "results/db/goals").glob("*.toml")))
def test_shipped_goals_parse(f):
    g = goal.load(f)
    assert g["primary"] == "wall_est" and "kld@REFBF" in g["constraints"]["bounds"]


def test_example_from_the_design():
    g = goal.parse('''[goal]
workload = "codex-8task-1agent"
primary = "wall_est"
tiebreak = ["kld@REFBF", "vram_min_free_mib"]
[constraints]
fits = { np = 2, ctx_per_slot = 98304 }
kld_REFBF = { max = 0.0035 }
vram_min_free_mib = { min = 300 }
runaway_rate = { max = 0.15 }
headless_only = false
''')
    assert g["tiebreak"] == ["kld@REFBF", "vram_min_free_mib"]
    assert g["constraints"]["bounds"]["kld@REFBF"] == {"max": 0.0035}
    assert g["constraints"]["fits"] == {"np": 2, "ctx_per_slot": 98304}
    assert goal.direction("vram_min_free_mib") == "max" and goal.direction("kld@REF6") == "min"


@pytest.mark.parametrize("text,match", [
    (BASE + "[constraints]\nkld_REFBFF = { max = 1 }\n", "unknown KLD reference"),
    (BASE + "[constraints]\nvram_free = { min = 1 }\n", "unknown constraint"),
    (BASE + "[constraints]\nvram_min_free_mib = { mn = 1 }\n", "expected"),
    (BASE + '[constraints]\nvram_min_free_mib = { min = "x" }\n', "not a number"),
    (BASE + "[constraints]\nfits = { gpus = 2 }\n", "allowed keys"),
    (BASE + "[constraints]\nheadless_only = 1\n", "boolean"),
    ('[goal]\nworkload = "w"\nprimary = "kld@REFBF"\n', "must be wall_est"),
    (BASE + 'tiebreak = ["latency"]\n', "unknown metric"),
    ('[goal]\nprimary = "wall_est"\n', "workload"),
    (BASE + "[extra]\na = 1\n", "unknown top-level"),
    ("[goal\n", "invalid TOML"),
])
def test_goal_errors_are_loud(text, match):
    with pytest.raises(goal.GoalError, match=match):
        goal.parse(text)
