"""goal.toml: hard constraints, one primary scalar, ordered lexicographic tie-breaks.

`model = ["swift-1.5:*"]` restricts candidates to model ids matching any fnmatch pattern; a goal without it
accepts every model, which is almost never what a fidelity goal means.
Contract: unknown keys are errors (a typo must not silently drop a constraint). `kld_REFBF` in
[constraints] is the metric `kld@REFBF`; the reference is always explicit.
"""
import re
import tomllib

DIRECTION = {"wall_est": "min", "vram_min_free_mib": "max", "runaway_rate": "min", "solve_rate": "max"}
BOUNDED = {"vram_min_free_mib", "runaway_rate", "solve_rate"}
FITS_KEYS = {"np", "ctx_per_slot", "engine"}
REFS = {"REF6", "REF8", "REFBF"}


class GoalError(ValueError):
    pass


def metric_key(k):
    m = re.fullmatch(r"kld_(\w+)", k)
    return f"kld@{m.group(1)}" if m else k


def direction(metric):
    if metric.startswith("kld@"):
        return "min"
    if metric in DIRECTION:
        return DIRECTION[metric]
    raise GoalError(f"unknown metric {metric!r}; known: wall_est, vram_min_free_mib, runaway_rate, solve_rate, kld@{{REF6,REF8,REFBF}}")


def _bound(metric, spec):
    if not isinstance(spec, dict) or not spec or set(spec) - {"min", "max"}:
        raise GoalError(f"constraint {metric}: expected {{ min = x }} and/or {{ max = x }}, got {spec!r}")
    for k, v in spec.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise GoalError(f"constraint {metric}.{k}: not a number: {v!r}")
    return dict(spec)


def parse(text):
    try:
        raw = tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise GoalError(f"invalid TOML: {e}")
    extra = set(raw) - {"goal", "constraints"}
    if extra:
        raise GoalError(f"unknown top-level tables {sorted(extra)}")
    g = raw.get("goal")
    if not isinstance(g, dict):
        raise GoalError("missing [goal]")
    bad = set(g) - {"name", "workload", "primary", "tiebreak", "top"}
    if bad:
        raise GoalError(f"unknown [goal] keys {sorted(bad)}")
    for k in ("workload", "primary"):
        if k not in g:
            raise GoalError(f"[goal].{k} is required")
    primary = metric_key(g["primary"])
    if primary != "wall_est":
        raise GoalError("[goal].primary must be wall_est (the only scalar with a cost model)")
    tie = [metric_key(t) for t in g.get("tiebreak", [])]
    for t in tie:
        direction(t)
    cons = {"fits": {}, "bounds": {}, "engine": None, "model": None, "headless_only": False}
    c = raw.get("constraints", {})
    for k, v in c.items():
        if k == "fits":
            if not isinstance(v, dict) or set(v) - FITS_KEYS:
                raise GoalError(f"fits: allowed keys {sorted(FITS_KEYS)}, got {v!r}")
            cons["fits"] = dict(v)
        elif k == "headless_only":
            if not isinstance(v, bool):
                raise GoalError("headless_only must be a boolean")
            cons["headless_only"] = v
        elif k == "engine":
            if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
                raise GoalError("engine must be a list of strings")
            cons["engine"] = v
        elif k == "model":
            if not isinstance(v, list) or not v or not all(isinstance(x, str) and ":" in x for x in v):
                raise GoalError("model must be a non-empty list of model-id patterns like 'swift-1.5:*'")
            cons["model"] = v
        else:
            m = metric_key(k)
            if m.startswith("kld@"):
                if m[4:] not in REFS:
                    raise GoalError(f"unknown KLD reference in {k!r} (known: {sorted(REFS)})")
            elif m not in BOUNDED:
                raise GoalError(f"unknown constraint {k!r}")
            cons["bounds"][m] = _bound(k, v)
    return {"name": g.get("name", g["workload"]), "workload": g["workload"], "primary": primary, "tiebreak": tie,
            "constraints": cons, "top": int(g.get("top", 15))}


def load(path):
    with open(path, encoding="utf-8") as f:
        return parse(f.read())
