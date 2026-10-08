"""Depth A/B: interleaved fresh boots of frozen arms, one seeded forced-length request per depth, steady full-K ms/round.

Every number comes from the stored per-depth records: the live run and --reanalyze share evaluate_boot()/finish().

experiment.toml, [depth] (the keys this module reads):
  depths, k, identity ("within-arm" | "cross-arm" | "off": needs `identity_off_reason`; streams > 1 only, because two slots
                share ubatches by arrival timing and kernels pick paths by batch size, so even one build does not repeat its tokens), distinct_builds, min_steady_intervals, min_predicted, request = {seed,
  n_predict, temperature, top_k, top_p, min_p, ignore_eos}, counters, expected, fixtures, wall_s.
  prompt        {path, sha} of a KLDT token file of real served sessions (pinned like draft_vocab). Absent: uniform random
                ids, whose output degenerates into tokens outside the draft vocab (J5q 32k: 0.0008 acceptance), so rounds
                are all-reject and ms/round misses per-accepted-token work.
  min_acceptance  a depth where any arm's acceptance is below this is insufficient data (default 0: off).
  request.ignore_eos  default true (forced length). It reaches the server as an EOG logit_bias, which defeats the backend
                candidate path under -sm tensor (verify rows fall back to a host scan): real serving sends no bias. false = the
                production request; a stream may then end on EOS, and below min_predicted tokens it is insufficient data.
  [arms.X] request  {seed, n_predict, temperature, top_k, top_p, min_p, ignore_eos} merged over [depth] request for this arm's boots
                only (keys outside that set are refused at load). The effective request of each arm is written to every
                boot's request.json, to pins.json and to every depth of summary.json (request_by_arm).
  [arms.X] k    this arm's draft length (default [depth] k): it gates the steady full-K rounds and fills `{k}` in server.args.
                Arms of different k generate different tokens: use identity "within-arm".
  streams       concurrent requests, one per server slot (default 1). >1: each stream prefills alone from scratch, then all
                decode together from their own slot's cache; per-stream ms/round and tok/s are taken inside the overlap window.
  min_overlap   streams > 1: overlap window / shortest stream decode below this is insufficient data, rc 5 (default 0.8).
                Streams of equal n_predict end apart when their acceptance differs, so the shortest one is the yardstick.
                The ratio only bounds start misalignment (<= 1 - min_overlap of the shorter decode, 20 % by default); that the
                window holds enough rounds is the min_steady_intervals gate applied inside it.
  slot_ctx      streams > 1: per-slot context; default = deepest depth + n_predict + ctx_pad, rounded up to 256; a given
                value below that is refused. The server -np and -c must equal streams and streams x slot_ctx ({np} {ctx}
                expand in server.args), and /props must report them.
  ctx_pad       streams > 1: slack added to the derived slot_ctx (default 1024).
  [canary] depths  depths of the canary only (default [depth] depths); streams > 1 canaries need fewer to fit their cap.
  max_replay    streams > 1 or texts > 1: a decode request that re-evaluates more prompt tokens than this missed its cache (default 1000).
  texts         N sampled texts per depth (default 1 = unchanged records and summary; np1 only). Each depth runs N requests on the
                same prompt with seeds seed + i: text 0 prefills from scratch (cache_n 0), texts 1..N-1 send cache_prompt and must reuse
                it (prompt_n + cache_n = depth, prompt_n <= max_replay), so only decode is paid again. Every text is gated alone
                (min_predicted, min_acceptance, steady intervals, cache reuse); a failing text fails the depth with its class.
                Needs `text_s` for mkitem (seconds of one extra text's decode, G) and [canary] texts may shorten the canary.
                depth-<d>.json = {"texts": [record per text], "mono_ns"}; every record carries its own mono_ns (round-trace window).
                SAMPLE UNIT. The texts are the sample: across boots of one arm they repeat exactly (within-arm identity compares the
                whole list of N texts), so boots measure timing noise only and texts measure content variance. All texts of a depth
                share one prompt: the spread is sampling variance given that prompt, not variance across prompts.
                summary.json per depth: {arm}_per_text (mean over the arm's boots per text), {arm}_stats (mean, sample SD, n over
                texts), {other}_minus_{first}_stats (delta of means, SE, Welch t and df, seeded bootstrap 95 % CI over texts), and the
                boot-to-boot ms/round delta as before.
"""
import copy
import json
import math
import os
import random
import statistics
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import lib
from .lib import Fail

DEFAULT_KNOBS = ("GGML_SCHED_INPUT_STAGING", "LLAMA_ROUND_TRACE", "GGML_SCHED_DEBUG")


GENERATION_PROMPT = [248046, 198, 248045, 74455, 198, 248068, 198]  # <|im_end|>\n<|im_start|>assistant\n<think>\n


def kldt_sequences(path):
    import struct
    data = Path(path).read_bytes()
    if data[:4] != b"KLDT":
        raise Fail(f"inputs changed: {path} is not a KLDT token file", "integrity")
    count, off, out = struct.unpack_from("<i", data, 4)[0], 8, []
    for _ in range(count):
        n = struct.unpack_from("<i", data, off)[0]
        out.append(list(struct.unpack_from(f"<{n}i", data, off + 4)))
        off += 4 + 4 * n
    return out


def corpus_prompt(sequences, depth, stream=0):
    """Real session text cut to exactly depth tokens, closed by the served generation prompt so decode is an assistant turn.
    Stream s starts at sequence s: concurrent streams share no prefix."""
    need = depth - len(GENERATION_PROMPT)
    order = sequences[stream % len(sequences):] + sequences[:stream % len(sequences)]
    body = [t for seq in order for t in seq][:need]
    if len(body) < need:
        raise Fail(f"prompt corpus has {len(body)} tokens < {need}", "integrity")
    return body + GENERATION_PROMPT


def prompt_tokens(depth, seed=1729, corpus=None, stream=0):
    """corpus: real text (see [depth] prompt). Else varied random ids, seed-dependent (a repeated token hid divergence)."""
    if corpus is not None:
        return corpus_prompt(corpus, depth, stream)
    state, out = seed, []
    for _ in range(depth):
        state = (state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        out.append(1000 + (state >> 33) % 149000)
    return out


def knobs(x):
    return tuple(x["server"].get("knobs", DEFAULT_KNOBS))


def depths_of(x, canary=False):
    """[canary] depths shortens the canary (a deep np > 1 boot does not fit its cap); the run always uses [depth] depths."""
    return x["canary"].get("depths", x["depth"]["depths"]) if canary else x["depth"]["depths"]


def arm_names(x):
    return list(dict.fromkeys(x["plan"]))


def canary_plan(x):
    return list(x["canary"].get("plan", arm_names(x)))


def arm_k(x, name):
    return x["arms"][name].get("k", x["depth"]["k"])


def arm_request(x, name):
    """The arm's effective request: [depth] request over the forced-length default, then the arm's own override."""
    return {"ignore_eos": True} | x["depth"]["request"] | x["arms"][name].get("request", {})


def streams(x):
    return x["depth"].get("streams", 1)


def texts_of(x, canary=False):
    d = x["depth"]
    return x["canary"].get("texts", d.get("texts", 1)) if canary else d.get("texts", 1)


def slot_ctx(x):
    d = x["depth"]
    n_predict = max(arm_request(x, a)["n_predict"] for a in arm_names(x))
    need = -(-(max(d["depths"]) + n_predict + d.get("ctx_pad", 1024)) // 256) * 256
    have = d.get("slot_ctx", need)
    if have < need:
        raise Fail(f"inputs changed: depth.slot_ctx {have} < {need} (deepest depth + n_predict + ctx_pad)", "integrity")
    return have


def boot(x, arm, port, dest, canary, stub=False):
    """argv and environment of one boot, exactly as the server will get them."""
    shape = {"np": streams(x), "ctx": streams(x) * slot_ctx(x)} if streams(x) > 1 else None
    command = [str(arm["exe"]), *lib.expand_args(x["server"]["args"], dict(arm, k=arm_k(x, arm["name"])), x, port, shape)]
    if shape:
        lib.check_shape(command, shape["np"], shape["ctx"])
    base = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(x.get("devices", [])), OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8")
    extra = dict(x["canary"].get("env", {})) if canary else {}
    if canary or x["depth"].get("counters"):
        extra["LLAMA_ROUND_TRACE"] = str(dest / "round-trace.jsonl")
    if stub:
        extra |= {k: v for k, v in os.environ.items() if k.startswith("STUB_")} | {"STUB_ARM": arm["name"]}
    return command, lib.server_env(base, arm, knobs(x), extra)


def capture(x, out, port):
    """CPU proof: write each planned boot's command.json/env.json without starting anything."""
    out = Path(out)
    for i, name in enumerate(x["plan"]):
        dest = out / f"{i}-{name}"
        dest.mkdir(parents=True, exist_ok=True)
        command, env = boot(x, x["arms"][name], port, dest, canary=False)
        (dest / "request.json").write_text(json.dumps(arm_request(x, name)))
        (dest / "command.json").write_text(json.dumps(command))
        (dest / "env.json").write_text(json.dumps(lib.env_record(env, knobs(x))))
    return 0


def check_inputs(x, full_model_sha=False):
    pins = {}
    for arm in x["arms"].values():
        pins |= lib.check_frozen(arm)
    if x["depth"].get("distinct_builds"):
        up, patch = (x["arms"][a] for a in arm_names(x)[:2])
        bins = lambda arm: sorted(v for k, v in pins.items() if Path(k).parent == arm["bindir"])
        if up["commit"] == patch["commit"] or bins(up) == bins(patch):
            raise Fail("frozen bytes changed: the arms are not distinct builds", "integrity")
    lib.check_input(x["draft_vocab"], "draft vocab")
    pins[str(x["draft_vocab"]["path"].resolve())] = x["draft_vocab"]["sha"]
    if "prompt" in x["depth"]:
        lib.check_input(x["depth"]["prompt"], "depth prompt corpus")
        pins[str(Path(x["depth"]["prompt"]["path"]).resolve())] = x["depth"]["prompt"]["sha"]
    lib.check_model(x["model"], full=full_model_sha)
    pins[str(x["model"]["path"].resolve())] = x["model"]["sha"]
    for path in sorted(Path(__file__).parent.glob("*.py")) + [x["file"]]:
        pins[str(path.resolve())] = lib.file_sha(path)
    return pins


def request_body(x, arm, depth, stream=0, decode=False, text=0):
    """Stream s has its own prompt and seed (stream 0 is the single-stream request). streams > 1: one slot per stream; the
    decode request reuses the slot's prefill cache, the prefill request never does. Text i > 0 (np1) keeps the prompt, samples
    with seed + i and reuses the cache text 0 left."""
    r = arm_request(x, arm)
    corpus = kldt_sequences(x["depth"]["prompt"]["path"]) if "prompt" in x["depth"] else None
    body = {"prompt": prompt_tokens(depth, r["seed"] + stream, corpus, stream), "n_predict": r["n_predict"], "ignore_eos": r["ignore_eos"], "cache_prompt": text > 0,
            "seed": r["seed"] + stream + text, "temperature": r["temperature"], "top_k": r["top_k"], "top_p": r["top_p"], "min_p": r["min_p"],
            "stream": True, "timings_per_token": True, "return_tokens": True}
    if streams(x) > 1:
        body |= {"id_slot": stream, "cache_prompt": decode} | ({} if decode else {"n_predict": 1})
    return body


def gate(d, record, prefill_gate, window=None, k=None):
    """One request's gates, in order; returns (steady result, token identity)."""
    if "error" in record:
        raise Fail(f"request failed: {record['error']}", "transient")
    result = lib.steady_intervals(record, d["min_steady_intervals"], k or d["k"], window)
    ident = lib.token_identity(record)
    t = result["timings"]
    if t["predicted_n"] < d["min_predicted"]:
        raise Fail(f"too few predicted tokens: {t['predicted_n']} < {d['min_predicted']} (requested {record['request']['n_predict']}; "
                   "with ignore_eos false the stream may end on EOS)", "insufficient_data")
    prefill_gate(t)
    if len(ident[0]) != t["predicted_n"]:
        raise Fail(f"missing final token identity: {len(ident[0])} tokens for predicted_n {t['predicted_n']}", "insufficient_data")
    return result, ident


def settle(results, mode, arm, depth, ident, unit=""):
    """The identity entry of this depth, created by its first boot (cross-arm) or per arm (within-arm); a later boot must
    reproduce it. Returns the entry the boot's run is appended to."""
    if mode == "off":
        return results.setdefault(depth, {"identity": None, "runs": []})
    if mode == "within-arm":
        entry = results.setdefault(depth, {"identity": {}, "runs": []})
        expected = entry["identity"].get(arm)
    else:
        expected = results.get(depth, {}).get("identity")
    if expected is not None and list(expected) != list(ident):
        where = f" {unit} {next(s for s, (a, b) in enumerate(zip(expected, ident)) if a != b)}" if unit else ""
        raise Fail(f"{mode} identity failed at depth {depth}{where}: tokens or draft counters differ", "integrity")
    if mode == "within-arm":
        entry["identity"][arm] = ident
        return entry
    return results.setdefault(depth, {"identity": ident, "runs": []})


def first_prefill(depth):
    def check(t):
        if t.get("cache_n") != 0 or t.get("prompt_n") != depth:
            raise Fail(f"effective params mismatch (prefill): prompt_n {t.get('prompt_n')} cache_n {t.get('cache_n')}", "integrity")
    return check


def replay_gate(d, depth):
    """A request meant to reuse the slot's cache: the prompt is split exactly into cached and re-evaluated tokens, few of the latter."""
    def check(t):
        if (t.get("prompt_n") or 0) + (t.get("cache_n") or 0) != depth:
            raise Fail(f"effective params mismatch (prefill): prompt_n {t.get('prompt_n')} + cache_n {t.get('cache_n')} != {depth}", "integrity")
        if t["prompt_n"] > d.get("max_replay", 1000):
            raise Fail(f"decode request re-evaluated {t['prompt_n']} tokens > {d.get('max_replay', 1000)}: its cache was missed", "insufficient_data")
    return check


def evaluate_boot(x, dest, arm, results, canary=False):
    """Gate one boot's stored records and add them to results; returns the failures (Fail objects), never raising them."""
    if streams(x) > 1:
        return evaluate_streams(x, dest, arm, results, canary)
    if texts_of(x, canary) > 1:
        return evaluate_texts(x, dest, arm, results, canary)
    d, mode = x["depth"], x["depth"]["identity"]
    if mode == "off":
        raise Fail("identity = \"off\" is only for streams > 1", "integrity")
    failures = []
    for depth in depths_of(x, canary):
        try:
            record = json.loads((dest / f"depth-{depth}.json").read_text())
            result, ident = gate(d, record, first_prefill(depth), k=arm_k(x, arm))
            entry = settle(results, mode, arm, depth, ident)
            run = {"arm": arm, **result}
            if d.get("counters"):
                run["counters"] = lib.trace_window(dest / "round-trace.jsonl", *record.get("mono_ns", (0, -1)), d["counters"])
            entry["runs"].append(run)
        except Fail as error:
            failures.append(Fail(f"depth {depth}: {error}", error.cls))
        except (OSError, ValueError, KeyError) as error:
            failures.append(Fail(f"depth {depth}: unreadable record: {error}", "deterministic"))
    if canary:
        failures += proofs(x, dest, arm)
    return failures


def evaluate_texts(x, dest, arm, results, canary=False):
    """texts > 1 (np1): per depth, text 0's fresh prefill and texts 1.. replaying the cache, each gated alone; the identity entry
    is the list of all N texts' identities."""
    d, mode, n = x["depth"], x["depth"]["identity"], texts_of(x, canary)
    if mode == "off":
        raise Fail("identity = \"off\" is only for streams > 1", "integrity")
    failures = []
    for depth in depths_of(x, canary):
        try:
            records = json.loads((dest / f"depth-{depth}.json").read_text())["texts"]
            gated = []
            for i, record in enumerate(records):
                try:
                    result, ident = gate(d, record, first_prefill(depth) if i == 0 else replay_gate(d, depth), k=arm_k(x, arm))
                    t = result["timings"]
                    if t["draft_n_accepted"] / t["draft_n"] < d.get("min_acceptance", 0):
                        raise Fail(f"acceptance {t['draft_n_accepted'] / t['draft_n']:.3f} < min_acceptance {d['min_acceptance']}: degenerate decode, "
                                   "ms/round covers all-reject rounds only", "insufficient_data")
                    if d.get("counters"):
                        result["counters"] = lib.trace_window(dest / "round-trace.jsonl", *record.get("mono_ns", (0, -1)), d["counters"])
                except Fail as error:
                    raise Fail(f"text {i}: {error}", error.cls)
                gated.append((result, ident))
            if len(gated) != n:
                raise Fail(f"request failed: {len(gated)} texts recorded for {n}", "transient")
            entry = settle(results, mode, arm, depth, [ident for _, ident in gated], unit="text")
            entry["runs"] += [{"arm": arm, "text": i, **result} for i, (result, _) in enumerate(gated)]
        except Fail as error:
            failures.append(Fail(f"depth {depth}: {error}", error.cls))
        except (OSError, ValueError, KeyError) as error:
            failures.append(Fail(f"depth {depth}: unreadable record: {error}", "deterministic"))
    if canary:
        failures += proofs(x, dest, arm)
    return failures


def stamps(record):
    """(arrival ns, cumulative predicted_n) of every chunk that carried tokens."""
    return [(c["t"], c["timings"]["predicted_n"]) for c in record["chunks"] if c.get("tokens") and "t" in c and
            isinstance(c.get("timings"), dict) and isinstance(c["timings"].get("predicted_n"), (int, float))]


def overlap(records, minimum):
    """Window in which every stream was decoding: [latest first token, earliest last token]. Tokens per stream inside it come
    from the cumulative counts at the window edges. A window below `minimum` of the shortest stream is not data. That bounds
    start misalignment only (starts differ by <= (1 - minimum) of the shorter decode); sufficiency is min_steady_intervals."""
    marks = [stamps(r) for r in records]
    if any(len(m) < 2 for m in marks):
        raise Fail("too few stamped decode chunks to find the overlap window", "insufficient_data")
    lo, hi = max(m[0][0] for m in marks), min(m[-1][0] for m in marks)
    shortest = max(min(m[-1][0] - m[0][0] for m in marks), 1)
    if hi <= lo or (hi - lo) / shortest < minimum:
        raise Fail(f"decode overlap {max(hi - lo, 0) / shortest:.2f} < {minimum}: the streams did not decode together", "insufficient_data")
    at = lambda m, t: max((n for stamp, n in m if stamp <= t), default=0)
    tokens = [at(m, hi) - at(m, lo) for m in marks]
    seconds = (hi - lo) / 1e9
    return {"window_ns": [lo, hi], "overlap_s": seconds, "overlap_frac": (hi - lo) / shortest, "window_tokens": tokens,
            "tok_s": [n / seconds for n in tokens], "agg_tok_s": sum(tokens) / seconds}


def final_timings(record):
    final = next((c for c in reversed(record["chunks"]) if c.get("stop") is True and isinstance(c.get("timings"), dict)), None)
    if "error" in record or final is None:
        raise Fail(f"request failed: {record.get('error', 'no final timings')}", "transient" if "error" in record else "insufficient_data")
    return final["timings"]


def evaluate_streams(x, dest, arm, results, canary=False):
    """streams > 1: per depth, each stream's fresh prefill, then the decode records gated inside the overlap window."""
    d, mode, n = x["depth"], x["depth"]["identity"], streams(x)
    if mode == "off" and not d.get("identity_off_reason"):
        raise Fail("identity = \"off\" needs identity_off_reason", "integrity")
    failures, vram = [], lib.read_json(dest / "vram.json")
    for depth in depths_of(x, canary):
        try:
            record = json.loads((dest / f"depth-{depth}.json").read_text())
            if len(record["prefill"]) != n or len(record["streams"]) != n:
                raise Fail(f"request failed: {len(record['prefill'])} prefill and {len(record['streams'])} decode records for {n} streams", "transient")
            for p in record["prefill"]:
                first_prefill(depth)(final_timings(p))

            replayed = replay_gate(d, depth)
            for r in record["streams"]:
                if "error" in r:
                    raise Fail(f"request failed: {r['error']}", "transient")
            window = overlap(record["streams"], d.get("min_overlap", 0.8))
            gated = [gate(d, r, replayed, tuple(window["window_ns"]), k=arm_k(x, arm)) for r in record["streams"]]
            entry = settle(results, mode, arm, depth, [ident for _, ident in gated], unit="stream")
            run = {"arm": arm, "streams": [{**res, "tok_s": window["tok_s"][s], "window_tokens": window["window_tokens"][s]} for s, (res, _) in enumerate(gated)],
                   **{k: window[k] for k in ("overlap_s", "overlap_frac", "agg_tok_s")}, "vram_min_free_mib": vram}
            if d.get("counters"):
                run["counters"] = lib.trace_window(dest / "round-trace.jsonl", *record.get("mono_ns", (0, -1)), d["counters"])
            entry["runs"].append(run)
        except Fail as error:
            failures.append(Fail(f"depth {depth}: {error}", error.cls))
        except (OSError, ValueError, KeyError) as error:
            failures.append(Fail(f"depth {depth}: unreadable record: {error}", "deterministic"))
    if canary:
        failures += proofs(x, dest, arm)
    return failures


def proofs(x, dest, arm):
    """Knob engaged on the real path: compact masks in the deepest decode window, staged inputs, the arm's own log lines."""
    c, out, failures = x["canary"], {}, []
    log = dest / "server.log"
    try:
        if "compact_mask" in c.get("proofs", []):
            window = lib.read_json(dest / "window.json")
            if not window or len(window) != 2:
                raise Fail("path-engaged proof inconclusive: no deepest-depth decode window", "insufficient_data")
            out["mask"] = lib.check_compact_masks(lib.mask_uploads(log, *window), c["max_suffix_kib"])
        if "staging" in c.get("proofs", []):
            out["staging"] = lib.check_staging(lib.trace_counts(dest / "round-trace.jsonl"))
        lib.verify_log_patterns(log.read_text(errors="replace"), x["arms"][arm]["canary_log"], f"arm {arm}")
        out["log_patterns"] = x["arms"][arm]["canary_log"]
    except Fail as error:
        failures.append(Fail(f"engagement: {error}", error.cls))
    (dest / "engaged.json").write_text(json.dumps(out, indent=2))
    return failures


def spread(values):
    """Sample SD over boots (None below two) and n."""
    return {"sd": statistics.stdev(values) if len(values) > 1 else None, "n": len(values)}


def boot_stats(result, arms, runs, streams=1):
    """Per-boot delta list, spread and SE of every later arm against the first; returns the per-arm boot means."""
    effect = lib.boot_delta(runs, streams)
    for other, pair in effect["pairs"].items():
        delta = f"{other}_minus_{arms[0]}_ms_D"
        result[f"{delta}_se"] = pair["se"]
        if pair["boots"]:
            result[f"{delta}_boots"] = pair["boots"]
            result[f"{delta}_boot_spread"] = spread(pair["boots"])
    return effect["arms"]


STREAMS_NOTE = ("s0/s1 ms/round are the same batch steps, not independent samples: the boot is the unit. The delta is the mean of "
                "the per-boot deltas (boot i of each arm paired, streams averaged); its SD is between boots.")


def finish_streams(x, results):
    """Per arm and stream the mean of the boots; the delta is the mean of the per-stream deltas (each also kept). Per-boot
    lists and spreads keep the boot as the unit of variance (see STREAMS_NOTE)."""
    arms, d, n = arm_names(x), x["depth"], streams(x)
    delta = f"{arms[1]}_minus_{arms[0]}_ms_D"
    for depth, result in results.items():
        runs = {a: [r for r in result["runs"] if r["arm"] == a] for a in arms}
        ms = {(a, s): statistics.mean(r["streams"][s]["steady_ms_per_round_D"] for r in runs[a]) for a in arms for s in range(n)}
        boot_ms = boot_stats(result, arms, runs, n)
        result["streams_note"] = STREAMS_NOTE
        result["request_by_arm"] = {a: arm_request(x, a) for a in arms}
        for s in range(n):
            result[f"s{s}_{delta}"] = ms[arms[1], s] - ms[arms[0], s]
        result[delta] = statistics.mean(result[f"s{s}_{delta}"] for s in range(n))
        if str(depth) in d.get("expected", {}):
            result[f"expected_{delta}"] = d["expected"][str(depth)]
            result[f"measured_minus_expected_{delta}"] = result[delta] - d["expected"][str(depth)]
        for kind in d.get("counters", []):
            for a in arms:
                result[f"{a}_{kind}_per_round_M"] = statistics.mean(r["counters"][kind]["count"] / r["counters"]["rounds"] for r in runs[a])
                result[f"{a}_{kind}_host_us_per_round_M"] = statistics.mean(r["counters"][kind]["host_us"] / r["counters"]["rounds"] for r in runs[a])
        for a in arms:
            result[f"{a}_ms_per_round_boots_D"], result[f"{a}_ms_per_round_boot_spread"] = boot_ms[a], spread(boot_ms[a])
            for s in range(n):
                result[f"{a}_s{s}_ms_per_round_D"] = ms[a, s]
                result[f"{a}_s{s}_ms_per_round_boots_D"] = [r["streams"][s]["steady_ms_per_round_D"] for r in runs[a]]
                result[f"{a}_s{s}_intervals_M"] = [r["streams"][s]["intervals_M"] for r in runs[a]]
                result[f"{a}_s{s}_tok_s_M"] = statistics.mean(r["streams"][s]["tok_s"] for r in runs[a])
            result[f"{a}_agg_tok_s_M"] = statistics.mean(r["agg_tok_s"] for r in runs[a])
            result[f"{a}_overlap_frac_min"] = min(r["overlap_frac"] for r in runs[a])
            timings = [st["timings"] for r in runs[a] for st in r["streams"]]
            result[f"{a}_acceptance_M"] = sum(t["draft_n_accepted"] for t in timings) / sum(t["draft_n"] for t in timings)
            boots = [r["vram_min_free_mib"] for r in runs[a] if r.get("vram_min_free_mib")]
            result[f"{a}_vram_min_free_mib"] = {g: min(b[g] for b in boots) for g in boots[0]} if boots else None
            if len(boots) < len(runs[a]):  # --reanalyze of a dir without vram.json must not read as "no VRAM data" silently
                result[f"{a}_vram_note"] = f"vram.json missing or empty in {len(runs[a]) - len(boots)} of {len(runs[a])} boots"
        result[f"{arms[1]}_over_{arms[0]}_agg_tok_s"] = result[f"{arms[1]}_agg_tok_s_M"] / result[f"{arms[0]}_agg_tok_s_M"]
    return results


TEXTS_NOTE = ("The texts are the sample unit: boots of one arm repeat the same texts exactly (within-arm identity), so they measure timing "
              "noise only; the texts of a depth share one prompt and differ by sampling seed. {arm}_per_text is each text's mean over the arm's "
              "boots; {arm}_stats are mean, sample SD and n over texts; the delta of means has a Welch SE over texts and a seeded bootstrap CI.")
TEXT_METRICS = ("tokens_per_round_M", "acceptance_M", "steady_ms_per_round_D", "steady_ms_per_token_D", "decode_tok_s_M")
BOOTSTRAPS = 10000


def moments(values):
    return {"mean": statistics.mean(values), "sd": statistics.stdev(values) if len(values) > 1 else None, "n": len(values)}


def compare(a, b, resamples=BOOTSTRAPS):
    """b - a over independent texts: delta of means, Welch SE/t/df, percentile bootstrap 95 % CI (seeded: reanalysis reproduces it).
    SE, t, df and CI are None below two texts per arm; t and df None when both arms have zero variance."""
    out = {"delta": statistics.mean(b) - statistics.mean(a), "n": [len(a), len(b)], "se": None, "t": None, "df": None, "ci95_boot": None}
    if min(len(a), len(b)) < 2:
        return out
    va, vb = statistics.variance(a) / len(a), statistics.variance(b) / len(b)
    out["se"] = math.sqrt(va + vb)
    if va + vb > 0:
        out["t"] = out["delta"] / out["se"]
        out["df"] = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    rng = random.Random(0)
    deltas = sorted(statistics.mean(rng.choices(b, k=len(b))) - statistics.mean(rng.choices(a, k=len(a))) for _ in range(resamples))
    out["ci95_boot"] = [deltas[int(0.025 * resamples)], deltas[int(0.975 * resamples) - 1]]
    return out


def finish_texts(x, results):
    """texts > 1: per arm and text the mean over the arm's boots; the arm's figures are over texts (see TEXTS_NOTE). The
    legacy keys hold the arm mean over texts; min_acceptance was gated per text at evaluation."""
    arms, d, n = arm_names(x), x["depth"], texts_of(x)
    delta = f"{arms[1]}_minus_{arms[0]}_ms_D"
    for depth, result in results.items():
        runs = {a: [[r for r in result["runs"] if r["arm"] == a and r["text"] == i] for i in range(n)] for a in arms}
        boots = {a: len(runs[a][0]) for a in arms}
        if any(len(t) != boots[a] for a in arms for t in runs[a]):
            raise Fail(f"depth {depth}: arms have unequal boots per text", "integrity")
        per_text = {a: [] for a in arms}
        for a in arms:
            for i, rs in enumerate(runs[a]):
                per_round = [1 + r["timings"]["draft_n_accepted"] / (r["timings"]["draft_n"] / arm_k(x, a)) for r in rs]
                per_text[a].append({
                    "seed": arm_request(x, a)["seed"] + i,
                    "tokens_per_round_M": statistics.mean(per_round),
                    "acceptance_M": statistics.mean(r["timings"]["draft_n_accepted"] / r["timings"]["draft_n"] for r in rs),
                    "steady_ms_per_round_D": statistics.mean(r["steady_ms_per_round_D"] for r in rs),
                    "steady_ms_per_token_D": statistics.mean(r["steady_ms_per_round_D"] / p for r, p in zip(rs, per_round)),
                    "decode_tok_s_M": statistics.mean(1000 * r["timings"]["predicted_n"] / r["timings"]["predicted_ms"] for r in rs),
                    "steady_ms_per_round_boots_D": [r["steady_ms_per_round_D"] for r in rs]})
        boot_ms = {a: [statistics.mean(per_text[a][i]["steady_ms_per_round_boots_D"][j] for i in range(n)) for j in range(boots[a])] for a in arms}
        result["request_by_arm"] = {a: arm_request(x, a) for a in arms}
        result["texts_note"] = TEXTS_NOTE
        result["texts_n"] = n
        stats = {a: {m: moments([t[m] for t in per_text[a]]) for m in TEXT_METRICS} for a in arms}
        for other in arms[1:]:
            result[f"{other}_minus_{arms[0]}_ms_D"] = stats[other]["steady_ms_per_round_D"]["mean"] - stats[arms[0]]["steady_ms_per_round_D"]["mean"]
            result[f"{other}_minus_{arms[0]}_stats"] = {m: compare([t[m] for t in per_text[arms[0]]], [t[m] for t in per_text[other]]) for m in TEXT_METRICS}
            if other == arms[1] and str(depth) in d.get("expected", {}):
                result[f"expected_{delta}"] = d["expected"][str(depth)]
                result[f"measured_minus_expected_{delta}"] = result[delta] - d["expected"][str(depth)]
        if boots[arms[0]] == boots[arms[1]]:
            result[f"{delta}_boots"] = [b - a for a, b in zip(boot_ms[arms[0]], boot_ms[arms[1]])]
            result[f"{delta}_boot_spread"] = spread(result[f"{delta}_boots"])
        for kind in d.get("counters", []):
            for a in arms:
                flat = [r["counters"] for rs in runs[a] for r in rs]
                result[f"{a}_{kind}_per_round_M"] = statistics.mean(c[kind]["count"] / c["rounds"] for c in flat)
                result[f"{a}_{kind}_host_us_per_round_M"] = statistics.mean(c[kind]["host_us"] / c["rounds"] for c in flat)
        for a in arms:
            result[f"{a}_per_text"], result[f"{a}_stats"] = per_text[a], stats[a]
            result[f"{a}_ms_per_round_boots_D"], result[f"{a}_ms_per_round_boot_spread"] = boot_ms[a], spread(boot_ms[a])
            result[f"{a}_acceptance_M"] = stats[a]["acceptance_M"]["mean"]
            result[f"{a}_tokens_per_round_M"] = stats[a]["tokens_per_round_M"]["mean"]
            result[f"{a}_steady_ms_per_token_D"] = stats[a]["steady_ms_per_token_D"]["mean"]
            result[f"{a}_decode_tok_s_mean_M"] = stats[a]["decode_tok_s_M"]["mean"]
    return results


def finish(x, results):
    if streams(x) > 1:
        return finish_streams(x, results)
    if texts_of(x) > 1:
        return finish_texts(x, results)
    arms, d = arm_names(x), x["depth"]
    for depth, result in results.items():
        result["request_by_arm"] = {a: arm_request(x, a) for a in arms}
        boot_ms = boot_stats(result, arms, {a: [r for r in result["runs"] if r["arm"] == a] for a in arms})
        for a in arms:
            result[f"{a}_ms_per_round_boots_D"], result[f"{a}_ms_per_round_boot_spread"] = boot_ms[a], spread(boot_ms[a])
        mean = {a: statistics.mean(r["steady_ms_per_round_D"] for r in result["runs"] if r["arm"] == a) for a in arms}
        for other in arms[1:]:
            delta = f"{other}_minus_{arms[0]}_ms_D"
            result[delta] = mean[other] - mean[arms[0]]
            if other == arms[1] and str(depth) in d.get("expected", {}):
                result[f"expected_{delta}"] = d["expected"][str(depth)]
                result[f"measured_minus_expected_{delta}"] = result[delta] - d["expected"][str(depth)]
        for kind in d.get("counters", []):
            for a in arms:
                runs = [r["counters"] for r in result["runs"] if r["arm"] == a]
                result[f"{a}_{kind}_per_round_M"] = statistics.mean(c[kind]["count"] / c["rounds"] for c in runs)
                result[f"{a}_{kind}_host_us_per_round_M"] = statistics.mean(c[kind]["host_us"] / c["rounds"] for c in runs)
        for a in arms:
            runs = [r for r in result["runs"] if r["arm"] == a]
            t = runs[0]["timings"]
            result[f"{a}_acceptance_M"] = t["draft_n_accepted"] / t["draft_n"]
            result[f"{a}_decode_tok_s_M"] = 1000 * t["predicted_n"] / t["predicted_ms"]
            # Whole-decode figures averaged over the arm's boots (arms of different k or build generate different text).
            # Tokens per round = 1 + accepted drafts per round; ms per token = steady ms/round over it (D: full-K rounds only).
            per_boot = [1000 * r["timings"]["predicted_n"] / r["timings"]["predicted_ms"] for r in runs]
            result[f"{a}_decode_tok_s_boots_M"] = per_boot
            result[f"{a}_decode_tok_s_mean_M"] = statistics.mean(per_boot)
            per_round = [1 + r["timings"]["draft_n_accepted"] / (r["timings"]["draft_n"] / arm_k(x, a)) for r in runs]
            result[f"{a}_tokens_per_round_M"] = statistics.mean(per_round)
            result[f"{a}_steady_ms_per_token_D"] = statistics.mean(r["steady_ms_per_round_D"] / n for r, n in zip(runs, per_round))
            low = d.get("min_acceptance", 0)
            if result[f"{a}_acceptance_M"] < low:
                raise Fail(f"depth {depth}: arm {a} acceptance {result[f'{a}_acceptance_M']:.3f} < min_acceptance {low}: degenerate decode, "
                           "ms/round covers all-reject rounds only", "insufficient_data")
    return results


def worst(failures):
    return lib.worst_class(f.cls for f in failures)


def reanalyze(x, out):
    """Recompute summary.json from the stored records of a finished run, boot order from the directory names."""
    out, results, failures = Path(out), {}, []
    boots = sorted((p for p in out.iterdir() if p.is_dir() and p.name.split("-", 1)[0].isdigit()), key=lambda p: int(p.name.split("-", 1)[0]))
    for dest in boots:
        failures += evaluate_boot(x, dest, dest.name.split("-", 1)[1], results)
    if failures:
        raise Fail("INVALID: " + "; ".join(map(str, failures)), worst(failures))
    return finish(x, results)


def stream_depth(base, body, timeout, on_first, stamp=False):
    """stamp: each chunk carries its arrival time "t" (monotonic ns), the clock of the overlap window."""
    record = {"request": body, "chunks": []}
    req = urllib.request.Request(base + "/completion", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as reply:
            for line in reply:
                if line.startswith(b"data: ") and line.strip() != b"data: [DONE]":
                    if not record["chunks"]:
                        on_first()
                    record["chunks"].append(json.loads(line[6:]) | ({"t": time.monotonic_ns()} if stamp else {}))
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"
        if isinstance(error, urllib.error.HTTPError):
            record["error_body"] = error.read().decode(errors="replace")
    return record


def run_depth(x, arm, base, dest, depth, text=0):
    """One request; returns (record, [server-log offset of the first chunk])."""
    window, t0 = [], time.monotonic_ns()
    record = stream_depth(base, request_body(x, arm, depth, text=text), x["server"].get("request_cap_s", 600), lambda: window.append((dest / "server.log").stat().st_size))
    record["mono_ns"] = [t0, time.monotonic_ns()]
    return record, window


def run_texts(x, arm, base, dest, depth, canary=False):
    """The depth's N texts in order, each its own request and round-trace window; stops at the first failed request. The window is the last text's."""
    records, window = [], []
    for i in range(texts_of(x, canary)):
        record, window = run_depth(x, arm, base, dest, depth, i)
        records.append(record)
        if "error" in record:
            break
    return {"texts": records, "mono_ns": [records[0]["mono_ns"][0], records[-1]["mono_ns"][1]]}, window


def run_streams(x, arm, base, dest, depth):
    """Each stream prefills alone, from scratch; then every stream decodes at once from its own slot's cache."""
    cap, n, window = x["server"].get("request_cap_s", 600), streams(x), []
    prefill = [stream_depth(base, request_body(x, arm, depth, s), cap, lambda: None) for s in range(n)]
    records = [{"error": "not run: a prefill failed"}] * n
    t0 = time.monotonic_ns()
    if not any("error" in p for p in prefill):
        def decode(s):
            records[s] = stream_depth(base, request_body(x, arm, depth, s, True), cap,
                                      lambda: window.__setitem__(slice(None), window or [(dest / "server.log").stat().st_size]), stamp=True)
        threads = [threading.Thread(target=decode, args=(s,)) for s in range(n)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
    return {"prefill": prefill, "streams": records, "mono_ns": [t0, time.monotonic_ns()]}, window


def execute(x, out, port, canary=False, stub=False):
    if os.environ.get("GPUQ_ACTIVE") != "1":
        raise Fail("execution requires GPUQ_ACTIVE=1 (run it through gpuq)", "deterministic")
    out = Path(out)
    if out.exists() and any(p.name != "console.log" for p in out.iterdir()):
        raise Fail(f"output dir {out} is not fresh", "deterministic")
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    order = canary_plan(x) if canary else x["plan"]
    pins = check_inputs(x)
    d = x["depth"]
    plan = {"order": order, "depths": depths_of(x, canary), "arm_env": {a: x["arms"][a]["env"] for a in arm_names(x)},
            "arm_flags": {a: x["arms"][a]["flags"] for a in arm_names(x)}, "k": {a: arm_k(x, a) for a in arm_names(x)},
            "request": {a: arm_request(x, a) for a in arm_names(x)}, "pins": pins,
            "steady_minimum": d["min_steady_intervals"], "experiment": str(x["file"])}
    if streams(x) > 1:
        plan |= {"streams": streams(x), "slot_ctx": slot_ctx(x), "min_overlap": d.get("min_overlap", 0.8)}
    if texts_of(x, canary) > 1:
        plan |= {"texts": texts_of(x, canary)}
    (out / "pins.json").write_text(json.dumps(plan, indent=2))
    if not stub:
        lib.verify_nice(os.getpriority(os.PRIO_PROCESS, 0))
    lib.event("measure_start")
    results = {}
    for i, name in enumerate(order):
        arm = x["arms"][name]
        if i and x.get("devices") and not stub:
            lib.wait_gpu_idle(x["devices"])
        lib.check_frozen(arm)
        dest = out / f"{i}-{name}"
        dest.mkdir()
        command, env = boot(x, arm, port, dest, canary, stub)
        (dest / "env.json").write_text(json.dumps(lib.env_record(env, knobs(x))))
        (dest / "request.json").write_text(json.dumps(arm_request(x, name)))
        (dest / "command.json").write_text(json.dumps(command))
        server = lib.Server(command, env, dest / "server.log")
        began, smi = time.time(), None
        memlog = lib.MemLog(dest / "memlog.jsonl", lambda: server.proc.pid if server.proc else None).start()
        try:
            if streams(x) > 1 and x.get("devices"):
                smi = lib.start_smi(dest / "smi.csv", [str(arm["exe"]), "--smi", *x["devices"]] if stub else None)
            (dest / "pid").write_text(str(server.start()))
            base = f"http://127.0.0.1:{port}"
            server.wait_ready(base, x["server"].get("boot_cap_s", 180))
            outside = lib.libs_outside(server.proc.pid, arm["root"])
            if outside:
                raise Fail("frozen bytes changed: loaded outside frozen root: " + ", ".join(outside), "integrity")
            if streams(x) > 1:
                lib.verify_props(lib.http_json(base + "/props"), slot_ctx(x), streams(x))
            for depth in depths_of(x, canary):
                if streams(x) > 1:
                    record, window = run_streams(x, name, base, dest, depth)
                elif texts_of(x, canary) > 1:
                    record, window = run_texts(x, name, base, dest, depth, canary)
                else:
                    record, window = run_depth(x, name, base, dest, depth)
                (dest / f"depth-{depth}.json").write_text(json.dumps(record))
                if depth == depths_of(x, canary)[-1] and window:
                    (dest / "window.json").write_text(json.dumps(window + [(dest / "server.log").stat().st_size]))
        finally:
            server.stop(grace=30)
            lib.stop_child(smi, 5)
            lib.record_journal(began, dest / "journal.txt")
            (dest / "mem.json").write_text(json.dumps(memlog.stop()))
        if smi:
            (dest / "vram.json").write_text(json.dumps(lib.min_free_vram(dest / "smi.csv", x["devices"])))
        failures = evaluate_boot(x, dest, name, results, canary)
        (dest / "verdict.json").write_text(json.dumps({"failures": [str(f) for f in failures]}, indent=2))
        if failures:
            raise Fail("INVALID: " + "; ".join(map(str, failures)), worst(failures))
        lib.log(out, f"boot {i} {name}: {len(depths_of(x, canary))} depths ok in {time.time() - began:.0f}s")
    if not canary:
        (out / "summary.json").write_text(json.dumps(finish(x, results), indent=2))
    lib.log(out, f"{'canary' if canary else 'run'} ok: {len(order)} boots in {time.time() - start:.0f}s")
    return 0


def fixture_test(directory, devices=("GPU-ee9a3956-297c-604b-b4c1-f07930d27e46", "GPU-3fe72007-d2f3-11c0-4d57-cc6fbde1451f")):
    """Real-artifact replay: the parsers on a recorded llama.cpp SSE stream, nvidia-smi output and kernel journal lines."""
    import re
    directory = Path(directory)
    record = json.loads((directory / "llama-sse/long-tool-s1-1.json").read_text())
    result = lib.steady_intervals(record, 1)
    changed = copy.deepcopy(record)
    changed["chunks"].extend([{}, {"timings": None}, {"timings": {"draft_n": None}}, record["chunks"][-1]])
    assert lib.steady_intervals(changed, 1) == result
    try:
        lib.steady_intervals(record, 40)
    except ValueError as e:
        assert "insufficient" in str(e)
    else:
        raise AssertionError("real short fixture escaped size gate")
    assert lib.gpu_memory((directory / "nvidia-smi-query.csv").read_text(), list(devices)) == 123
    assert re.search(lib.XID_RE, (directory / "xid-lines.txt").read_text())
    assert not re.search(lib.XID_RE, "stop_xid_hash_s_G")
    # Recorded chat output has no native token IDs. The live canary must exercise that gate.
    try:
        lib.token_identity(record)
    except ValueError:
        pass
    else:
        raise AssertionError("identity accepted a record without native token IDs")
    print(json.dumps({"recorded_fixture": str(directory), "parser_result": result, "negative_controls": "PASS"}, indent=2))
    return 0


def preflight(x, out, fixtures):
    check_inputs(x, full_model_sha=True)
    fixture_test(fixtures)
    lib.log(out, f"preflight ok: plan {' '.join(x['plan'])}, canary {' '.join(canary_plan(x))}, depths {x['depth']['depths']}")
    return 0
