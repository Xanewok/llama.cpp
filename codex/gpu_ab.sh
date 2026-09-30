#!/usr/bin/env bash
# Parent-only GPU experiment. Freeze each build, rename its server to verify-ab,
# and generate bin/SHA256SUMS plus SOURCE before setting UP_DIR and PATCH_DIR.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
T0=$SECONDS
SP=/home/xanewok/.cache/claude-code-tmp/claude-1000/-home-xanewok-repos-qwen38-eval/93663eaa-5db0-437a-ac16-7d725d755e5b/scratchpad
UP_DIR=${UP_DIR:-$HOME/repos/llama.cpp-backend-verify-sampling/codex/frozen/patch/bin}
PATCH_DIR=${PATCH_DIR:-$ROOT/codex/frozen/patch/bin}
MODEL=${MODEL:-$HOME/models/Swift-1.5-Qwen3.8-27B-IQ4_XS.gguf}
TEMPLATE=${TEMPLATE:-$HOME/repos/qwen38-eval/results/session2/pair-5060ti-2060s/campaign-swift-fullsuite-20260915/swift-iq4-xs/attempts/66ebeeaec6a7043b-01/chat-template.jinja}
SMOKE=$SP/prefilter2/codex_smoke.py
OUT=${OUT:-$ROOT/codex/gpu-ab-results}
PORT=${PORT:-8096}
BUDGET_S=${BUDGET_S:-1200}
UUIDS=GPU-ee9a3956-297c-604b-b4c1-f07930d27e46,GPU-3fe72007-d2f3-11c0-4d57-cc6fbde1451f
ARMS=(up patch patch up)
CMD=(-m "$MODEL" -np 1 -fa on --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-n-min 0 --spec-draft-p-min 0
     -b 2048 -ub 128 --jinja --temp 1 --top-k 20 --top-p 0.95 --min-p 0.05 --host 127.0.0.1 --port "$PORT"
     -c 32768 -ctk q8_0 -ctv q8_0 -ngl 99 --spec-draft-type-k f16 --spec-draft-type-v f16 -sm tensor --tensor-split 50,50
     --chat-template-file "$TEMPLATE" --metrics --spec-draft-sampling probabilistic --backend-sampling -t 8 -tb 8)

if [[ ${DRY_RUN:-1} == 1 ]]; then
    export CUDA_VISIBLE_DEVICES=""
    for arm in "${ARMS[@]}"; do
        printf 'DRY_RUN: %s/verify-ab %s\n' "$([[ $arm == up ]] && echo "$UP_DIR" || echo "$PATCH_DIR")" "${CMD[*]}"
        printf 'DRY_RUN: backend target sampling; warm-up, tool seed 1, think seed 1\n'
    done
    exit 0
fi
[[ ${RUN_GPU_AB:-0} == 1 ]] || { echo 'Parent must set RUN_GPU_AB=1 DRY_RUN=0'; exit 4; }
renice -n 19 -p "$$" >/dev/null
export CUDA_VISIBLE_DEVICES=$UUIDS OMP_NUM_THREADS=8
mkdir -p "$OUT"
[[ -z $(find "$OUT" -mindepth 1 -maxdepth 1 -print -quit) ]] || { echo "Refuse nonempty OUT: $OUT"; exit 6; }
for dir in "$UP_DIR" "$PATCH_DIR"; do
    [[ -x $dir/verify-ab && -f $dir/SHA256SUMS && -f $dir/../SOURCE ]] || { echo "Missing frozen build: $dir"; exit 6; }
    rg -q ' [ *]?verify-ab$' "$dir/SHA256SUMS" || { echo "verify-ab is not in manifest: $dir"; exit 6; }
    (cd "$dir" && sha256sum -c --quiet SHA256SUMS)
    head -1 "$dir/../SOURCE"
done
rg -q "^60c8cbfb2" "$UP_DIR/../SOURCE" || { echo "Up must be the frozen clean backend-verify implementation"; exit 6; }
[[ $(stat -c %s "$MODEL") == 15475951296 && -f $TEMPLATE && -f $SMOKE ]] || exit 6
sha256sum "$MODEL" "$TEMPLATE" "$SMOKE" > "$OUT/input.sha256"
PID=
stop_server() {
    [[ -n $PID ]] || return 0
    if kill -0 "$PID" 2>/dev/null; then
        kill -TERM "$PID" 2>/dev/null || true
        for ((i=0; i<30; ++i)); do
            kill -0 "$PID" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "$PID" 2>/dev/null; then kill -KILL "$PID" 2>/dev/null || true; fi
    fi
    wait "$PID" 2>/dev/null || true
    PID=
}
trap stop_server EXIT
trap 'exit 130' INT TERM
xid_check() {
    journalctl -k -b --no-pager > "$OUT/kernel.log" || return 3
    if rg -qi 'Xid|fallen off' "$OUT/kernel.log"; then return 3; fi
}
busy_check() {
    if pgrep -x llama-server >/dev/null || pgrep -x verify-ab >/dev/null; then return 4; fi
    if [[ -n $(ss -Hltn "sport = :$PORT") ]]; then return 4; fi
    local used
    used=$(nvidia-smi --id="$UUIDS" --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s += $1} END {print s}')
    [[ $used =~ ^[0-9]+$ && $used -le 1500 ]]
}
request() {
    local tag=$1 conv=$2 seed=$3 ntok=$4 think=$5 remaining
    remaining=$((BUDGET_S - SECONDS + T0 - 30))
    (( remaining > 0 )) || return 7
    timeout "$remaining" nice -n 19 python3 - "$SMOKE" "$PORT" "$conv" "$seed" "$ntok" "$think" > "$OUT/$tag.json" <<'PY'
import hashlib, importlib.util, json, sys
path, port, conv, seed, ntok, think = sys.argv[1:]
spec = importlib.util.spec_from_file_location("codex_smoke", path)
cs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cs)
body = {"messages": dict(cs.conversations())[conv], "tools": cs.TOOLS, "tool_choice": "auto",
        "max_tokens": int(ntok), "seed": int(seed), "cache_prompt": False,
        "chat_template_kwargs": {"enable_thinking": think == "1"}}
r = cs.call(f"http://127.0.0.1:{port}/v1/chat/completions", body, timeout=300)
m = r["choices"][0]["message"]
text = {"content": m.get("content"), "reasoning_content": m.get("reasoning_content"),
        "tool_calls": [c["function"] for c in m.get("tool_calls") or []]}
digest = hashlib.sha256(json.dumps(text, sort_keys=True).encode()).hexdigest()
print(json.dumps({"timings": r["timings"], "sha256": digest, "message": text}))
PY
}

declare -A NBOOT=([up]=0 [patch]=0)
for arm in "${ARMS[@]}"; do
    xid_check
    busy_check
    (( SECONDS - T0 < BUDGET_S - 200 )) || exit 7
    NBOOT[$arm]=$((NBOOT[$arm] + 1))
    tag=$arm-${NBOOT[$arm]}
    dir=$UP_DIR
    if [[ $arm == patch ]]; then dir=$PATCH_DIR; fi
    (cd "$dir" && sha256sum -c --quiet SHA256SUMS)
    nice -n 19 "$dir/verify-ab" "${CMD[@]}" > "$OUT/$tag.server.log" 2>&1 &
    PID=$!
    healthy=0
    for ((poll=0; poll<300; ++poll)); do
        (( SECONDS - T0 < BUDGET_S - 30 )) || exit 7
        kill -0 "$PID" 2>/dev/null || exit 2
        if curl -sf -m 3 "http://127.0.0.1:$PORT/health" | rg -q '"ok"'; then healthy=1; break; fi
        sleep 1
    done
    (( healthy )) || exit 2
    request "$tag-warmup" fresh 1 128 1
    request "$tag-tool" follow 1 384 0
    request "$tag-think" follow 1 1024 1
    stop_server
    xid_check
done
sha256sum -c --quiet "$OUT/input.sha256"
nice -n 19 python3 - "$OUT" > "$OUT/summary.txt" <<'PY'
import json, pathlib, statistics, sys
out = pathlib.Path(sys.argv[1])
for req in ("tool", "think"):
    runs = {a: [json.loads((out / f"{a}-{i}-{req}.json").read_text()) for i in (1, 2)] for a in ("up", "patch")}
    assert len({r["sha256"] for rs in runs.values() for r in rs}) == 1, f"{req}: seed identity FAILED"
    assert len({r["timings"]["prompt_n"] for rs in runs.values() for r in rs}) == 1, f"{req}: prompt sizes differ"
    assert all(r["timings"]["cache_n"] == 0 for rs in runs.values() for r in rs), f"{req}: prompt cache used"
    values = {}
    prompts = {}
    for arm, rs in runs.items():
        values[arm] = []
        prompts[arm] = []
        for r in rs:
            t = r["timings"]
            assert t["draft_n"] > 0, "no speculative rounds"
            values[arm].append(t["predicted_ms"] / (t["draft_n"] / 3))
            prompts[arm].append(t["prompt_ms"])
            print(f"M: {arm} {req} timings={t}")
        print(f"D: {arm} {req} mean prompt_ms={statistics.mean(prompts[arm]):.3f}; M: boots={prompts[arm]}")
        print(f"D: {arm} {req} us/round={statistics.mean(values[arm])*1000:.1f}, two boots, draft_n/3 proxy")
    prompt_delta = statistics.mean(prompts["patch"]) - statistics.mean(prompts["up"])
    prompt_percent = 100*prompt_delta/statistics.mean(prompts["up"])
    print(f"D: {req} patch-up prompt_ms={prompt_delta:.3f} ({prompt_percent:.2f}%)")
    delta = statistics.mean(values["patch"]) - statistics.mean(values["up"])
    print(f"D: {req} patch-up={delta*1000:.1f} us/round; M: seed identity passed")
PY
cat "$OUT/summary.txt"
(( SECONDS - T0 <= BUDGET_S )) || exit 7
