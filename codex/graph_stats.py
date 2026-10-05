"""graph_stats.py TRACE...: per-phase graph misses per round; 'repeat' misses exclude the first sighting of each (phase, n_tokens)."""
import collections, json, sys
for path in sys.argv[1:]:
    rounds = 0; miss = collections.Counter(); rep = collections.Counter(); calls = collections.Counter()
    build_us = 0.0; seen = set(); d1 = collections.Counter()
    for line in open(path):
        r = json.loads(line); rounds += 1
        for ph in r["phases"]:
            name = ph["phase"]; key = (name, ph.get("n_tokens"))
            calls[name] += 1
            if name == "D1": d1[ph.get("n_tokens")] += 1
            c = {k["kind"]: k for k in ph.get("counters", [])}
            m = c.get("graph_miss", {}).get("count", 0)
            if m:
                miss[name] += m
                if key in seen: rep[name] += m
            seen.add(key)
            build_us += c.get("llama_build", {}).get("host_us", 0.0)
    print(json.dumps({"trace": path.split("/")[-2], "rounds": rounds, "phase_calls": dict(calls), "misses": dict(miss),
                      "repeat_misses": dict(rep), "repeat_misses_per_round": round(sum(v for k, v in rep.items() if k != "V") / rounds, 3),
                      "llama_build_us_per_round": round(build_us / rounds, 1), "D1_rows": dict(sorted(d1.items()))}))
