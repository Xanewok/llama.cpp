# K11a getter sync elision identity (np1/np2, plain and Meta(CPU,CPU)), debug check, negative controls. Codes 110-119.
TOKENS=${K11A_TOKENS:-48}
# the debug check queries the backend (exact); the threshold only sets when a slow sync on an idle backend is logged
DEBUG_US=80
# np,meta,fold: -duo shape (np2 fold off), np1 and Meta variants, fold C on
COMBOS=${K11A_COMBOS:-"1,0,1 1,1,0 2,0,0 2,1,0 2,1,1"}
RUN=$D/run
mkdir -p "$RUN"
# drive NAME NP META FOLD [ENV=VAL ...]; the exit code is the driver's
drive() {
    local name=$1 np=$2 meta=$3 fold=$4
    shift 4
    "$MGATE" env "$@" HOST_KV_Q8=1 NP2_THREADS=2 LLAMA_MTP_CHAIN=0 LLAMA_MTP_FOLD_C="$fold" \
        build-cpu/bin/bw-np2-driver "$MODEL" "$RUN/$name.json" "$meta" "$np" 1729 "$TOKENS" > "$RUN/$name.log" 2>&1
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
    python3 - "$RUN" $COMBOS <<'PY_COMPARE'
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
    defer = re.search(r'defer: sync_calls=(\d+) busy=(\d+)', log)
    assert defer and int(defer[2]) > 0, f'{t}: the deferred backend never had queued work at a sync'
    print(f'{t}: tokens/counters identical on/off/deferred; elided/missed per context {sums}; defer sync_calls={defer[1]} busy={defer[2]}')
print('sync elision identity PASS')
PY_COMPARE
}
# Mutants 1 and 2 drop one of the two enqueue marks; on the exercised paths the other mark still covers the getter, so they can
# only be seen by the debug check (the backend query finds queued work at a skipped sync). Mutant 1's only site was the state read in prompt_save,
# which now always syncs (9afd77984): it is reported, not required. Mutant 3 clears the flag after every enqueue: stale reads,
# which the token comparison under the deferred backend must catch in every combo (the test's sensitivity control).
mutants() {
    local combo np meta fold t m dbg rc
    for combo in ${K11A_MUTANT_COMBOS:-2,0,0 1,0,1 2,1,0}; do
        IFS=, read -r np meta fold <<< "$combo"
        t=$np-$meta-$fold
        [[ -s $RUN/$t-off.json ]] || drive "$t-off" "$np" "$meta" "$fold" LLAMA_SYNC_ELIDE=0
        for m in 1 2 3; do
            for dbg in 0 1; do
                rc=0
                drive "$t-mut$m-dbg$dbg" "$np" "$meta" "$fold" HOST_DEFER=1 LLAMA_SYNC_ELIDE_MUTANT=$m LLAMA_SYNC_ELIDE_DEBUG=$(( dbg ? DEBUG_US : 0 )) || rc=$?
                echo "$t mutant=$m debug=$dbg driver_rc=$rc"
            done
        done
    done
    python3 - "$RUN" ${K11A_MUTANT_COMBOS:-2,0,0 1,0,1 2,1,0} <<'PY_MUTANTS'
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
k11a_model() {
    stage 110 sync-elide-identity identity
    stage 111 sync-elide-mutants mutants
}
