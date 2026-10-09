#!/usr/bin/env bash
# K11a Mac CPU gates: build, getter sync elision identity (np1/np2, plain and Meta(CPU,CPU)), debug check, negative controls,
# state I/O under a stream-like backend.
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
[[ ${BW_CPU_ONLY:-0} == 1 && $(uname) == Darwin ]] || exit 8
OUT=${1:?output}
mkdir -p "$OUT/run"
GATE=$ROOT/codex/mac-model-gate.sh
MODEL=${BW_MODEL:?}
TOKENS=${K11A_TOKENS:-48}
# the debug check queries the backend (exact); the threshold only sets when a slow sync on an idle backend is logged
DEBUG_US=80
# np,meta,fold: -duo shape (np2 fold off), np1 and Meta variants, fold C on
COMBOS=${K11A_COMBOS:-"1,0,1 1,1,0 2,0,0 2,1,0 2,1,1"}
stage() {
    local code=$1 name=$2 rc=0 t0=$SECONDS
    shift 2
    [[ -z ${K11A_ONLY:-} || " $K11A_ONLY " == *" $code "* ]] || return 0
    set +e
    (set -e; "$@") > "$OUT/$name.log" 2>&1
    rc=$?
    set -e
    printf '%s\t%s\t%s\t%s\n' "$code" "$name" "$rc" "$((SECONDS-t0))" >> "$OUT/stages.tsv"
    echo "bw-k11a: $name rc=$rc wall_s_M=$((SECONDS-t0))"
    [[ $rc == 0 ]] || exit "$code"
}
flags=(-DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_CCACHE=ON -DGGML_OPENMP_FETCH=OFF
    -DGGML_LLAMAFILE=OFF -DGGML_CPU_REPACK=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF
    -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_APP=OFF
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DGGML_ACCELERATE=OFF)
read -r -a extra <<< "${BW_CPU_EXTRA:?}"
build() {
    "$BW_BUILDGATE" cmake -S codex/stack7-test -B build-cpu "${flags[@]}" "${extra[@]}"
    "$BW_BUILDGATE" cmake --build build-cpu --target bw-np2-driver k11a-state -j4
}
# drive NAME NP META FOLD [ENV=VAL ...]; the exit code is the driver's
drive() {
    local name=$1 np=$2 meta=$3 fold=$4
    shift 4
    "$GATE" env "$@" HOST_KV_Q8=1 NP2_THREADS=2 LLAMA_MTP_CHAIN=0 LLAMA_MTP_FOLD_C="$fold" \
        build-cpu/bin/bw-np2-driver "$MODEL" "$OUT/run/$name.json" "$meta" "$np" 1729 "$TOKENS" > "$OUT/run/$name.log" 2>&1
}
identity() {
    local combo np meta fold t
    for combo in $COMBOS; do
        IFS=, read -r np meta fold <<< "$combo"
        t=$np-$meta-$fold
        drive "$t-off" "$np" "$meta" "$fold" LLAMA_SYNC_ELIDE=0
        drive "$t-on" "$np" "$meta" "$fold"
        drive "$t-defer-off" "$np" "$meta" "$fold" HOST_DEFER=1 LLAMA_SYNC_ELIDE=0
        drive "$t-defer-debug" "$np" "$meta" "$fold" HOST_DEFER=1 LLAMA_SYNC_ELIDE_DEBUG=$DEBUG_US
    done
    python3 - "$OUT/run" $COMBOS <<'PY_COMPARE'
import json, re, sys
from pathlib import Path
d = Path(sys.argv[1])
for combo in sys.argv[2:]:
    t = combo.replace(',', '-')
    ref = json.loads((d/f'{t}-off.json').read_text())
    for arm in ('on', 'defer-off', 'defer-debug'):
        assert json.loads((d/f'{t}-{arm}.json').read_text()) == ref, f'{t}: {arm} tokens or counters differ'
    log = (d/f'{t}-defer-debug.log').read_text(errors='replace')
    sums = [(int(a), int(b)) for a, b in re.findall(r'sync_elide: elided=(\d+) missed=(\d+)', log)]
    assert sums and all(m == 0 for _, m in sums) and sum(e for e, _ in sums) > 0, f'{t}: debug check {sums}'
    assert 'sync_elide: missed:' not in log, f'{t}: debug check reported a miss'
    assert log.count('sync_elide: debug check engaged (backend query exact)') == log.count('sync_elide: debug check engaged') > 0, f'{t}: debug check not query-based'
    ck = (log.count('created context checkpoint'), log.count('restored context checkpoint'))
    assert min(ck) > 0, f'{t}: the debug arm did not save and restore a checkpoint {ck}'
    defer = re.search(r'defer: sync_calls=(\d+) busy=(\d+)', log)
    assert defer and int(defer[2]) > 0, f'{t}: the deferred backend never had queued work at a sync'
    print(f'{t}: tokens/counters identical on/off/deferred; elided/missed per context {sums}; checkpoints created/restored {ck}; defer sync_calls={defer[1]} busy={defer[2]}')
print('sync elision identity PASS')
PY_COMPARE
}
# Mutants 1 and 2 drop one of the two enqueue marks; on the exercised paths the other mark still covers the getter, so they can
# only be seen by the debug check (the backend query finds queued work at a skipped sync). Mutant 2 shows at logits_deferred_fetch. Mutant 1 showed
# only at prompt_save's state read (a4); state I/O now always syncs, so it has no visible site left and is reported, not required. Mutant 3 clears the flag after every enqueue: stale reads,
# which the token comparison under the deferred backend must catch in every combo (the test's sensitivity control).
mutants() {
    local combo np meta fold t m dbg rc
    for combo in ${K11A_MUTANT_COMBOS:-2,0,0 1,0,1 2,1,0}; do
        IFS=, read -r np meta fold <<< "$combo"
        t=$np-$meta-$fold
        [[ -s $OUT/run/$t-off.json ]] || drive "$t-off" "$np" "$meta" "$fold" LLAMA_SYNC_ELIDE=0
        for m in 1 2 3; do
            for dbg in 0 1; do
                rc=0
                drive "$t-mut$m-dbg$dbg" "$np" "$meta" "$fold" HOST_DEFER=1 LLAMA_SYNC_ELIDE_MUTANT=$m LLAMA_SYNC_ELIDE_DEBUG=$(( dbg ? DEBUG_US : 0 )) || rc=$?
                echo "$t mutant=$m debug=$dbg driver_rc=$rc"
            done
        done
    done
    python3 - "$OUT/run" ${K11A_MUTANT_COMBOS:-2,0,0 1,0,1 2,1,0} <<'PY_MUTANTS'
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
caught = {1: set(), 2: set(), 3: set()}
fired = set()
tok3 = []
for combo in sys.argv[2:]:
    t = combo.replace(',', '-')
    ref = json.loads((d/f'{t}-off.json').read_text())
    for m in (1, 2, 3):
        for dbg in (0, 1):
            name = f'{t}-mut{m}-dbg{dbg}'
            jf, lf = d/f'{name}.json', d/f'{name}.log'
            log = lf.read_text(errors='replace')
            how = []
            if not jf.exists():
                how.append('driver failed')
            elif json.loads(jf.read_text()) != ref:
                how.append('tokens differ')
            if dbg and 'sync_elide: missed: backend busy' in log:
                how.append('debug check fired')
                fired.add((m, t))
            print(f'{name}: {", ".join(how) or "not caught"}')
            if how:
                caught[m].add(name)
            if m == 3 and not dbg:
                tok3.append(bool(how))
# mutant 3 leaves a just-launched graph unsynced at the next getter: the query must find it busy in every combo
dbg3 = all((3, c.replace(',', '-')) in fired for c in sys.argv[2:])
ok = bool(caught[2]) and bool(caught[3]) and all(tok3) and dbg3
print({m: len(v) for m, v in caught.items()}, 'mutant 3 token-level catches per combo:', tok3, 'debug check on mutant 3 in every combo:', dbg3)
print('mutant controls', 'PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
PY_MUTANTS
}
# State save/restore copy through the buffer API, ordered after queued graph work only by the sync in front of them. Mutant 3
# (flag cleared after every enqueue) must not reach them since they always sync; mutant 4 (=3 with those syncs elidable, the
# behaviour before the forced sync) must corrupt the save or the restore: the fixture's sensitivity control.
state_io() {
    local arm sw
    for arm in off:LLAMA_SYNC_ELIDE=0 on:LLAMA_SYNC_ELIDE_DEBUG=0 mut3:LLAMA_SYNC_ELIDE_MUTANT=3 mut4:LLAMA_SYNC_ELIDE_MUTANT=4; do
        sw=${arm#*:}
        arm=${arm%%:*}
        "$GATE" env "$sw" build-cpu/bin/k11a-state "$MODEL" "$OUT/run/state-$arm.json" > "$OUT/run/state-$arm.log" 2>&1
    done
    python3 - "$OUT/run" <<'PY_STATE'
import json, re, sys
from pathlib import Path
d = Path(sys.argv[1])
r = {a: json.loads((d/f'state-{a}.json').read_text()) for a in ('off', 'on', 'mut3', 'mut4')}
for a in r:
    busy = re.search(r'defer: busy=(\d+)', (d/f'state-{a}.log').read_text(errors='replace'))
    assert busy and int(busy[1]) > 0, f'{a}: the deferred backend never had queued work'
    print(a, r[a], 'busy', busy[1])
assert r['off']['restore_holds'], 'reference arm: restore does not hold'
for a in ('on', 'mut3'):
    assert r[a] == r['off'], f'{a}: state I/O differs from the reference'
assert r['mut4']['save_fnv'] != r['off']['save_fnv'] or not r['mut4']['restore_holds'], 'mutant 4 not caught: the fixture cannot see a missing sync'
print('state I/O PASS (mutant 4 caught:', 'save' if r['mut4']['save_fnv'] != r['off']['save_fnv'] else '', 'restore' if not r['mut4']['restore_holds'] else '', ')')
PY_STATE
}
static_checks() {
    bash -n codex/bw-k11a-mac.sh
}
stage 20 cpu-build build
stage 21 sync-identity identity
stage 22 sync-mutants mutants
stage 23 state-io state_io
stage 31 static static_checks
python3 - "$OUT" <<'PY_SUMMARY'
import hashlib, json, subprocess, sys
from pathlib import Path
p = Path(sys.argv[1])
files = subprocess.check_output(['git','ls-files','src','ggml','tests','common'], text=True).splitlines()
h = hashlib.sha256()
for name in files:
    h.update(name.encode()+b'\0'+Path(name).read_bytes())
stages = [dict(code=int(c),name=n,rc=int(rc),wall_s_M=int(t)) for c,n,rc,t in (l.split('\t') for l in (p/'stages.tsv').read_text().splitlines())]
(p/'summary.json').write_text(json.dumps(dict(result='PASS',commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_files=files,source_sha256=h.hexdigest(),stages=stages),indent=2)+'\n')
PY_SUMMARY
