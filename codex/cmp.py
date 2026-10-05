"""cmp.py A B: identity of tokens and draft counters per request."""
import json, sys
a = json.load(open(sys.argv[1] + "/records.json")); b = json.load(open(sys.argv[2] + "/records.json"))
assert len(a) == len(b), (len(a), len(b))
key = lambda r: (r["tokens"], r["draft_n"], r["draft_n_accepted"])
diff = [(x["prompt"], x["seed"], x.get("turn", 0)) for x, y in zip(a, b) if key(x) != key(y)]
print(json.dumps({"requests": len(a), "differ": len(diff), "which": diff[:20],
                  "draft_n": [sum(r["draft_n"] for r in a), sum(r["draft_n"] for r in b)],
                  "accepted": [sum(r["draft_n_accepted"] for r in a), sum(r["draft_n_accepted"] for r in b)]}))
