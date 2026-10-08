#!/usr/bin/env bash
# usage: drop-component.sh COMPONENT[,COMPONENT...] [BASE] — rebase the current branch without the commits of the listed components (subjects in dropmap.tsv).
# Run on a scratch branch or tree only. Exit 0 = clean drop; 1 = rebase conflict (left in progress: resolve or `git rebase --abort`).
set -euo pipefail
export GIT_NO_LAZY_FETCH=1
comp=${1:?component}
export DROP_COMPONENT=$comp
base=${2:-def4a4e20}
src=$(cd "$(dirname "$0")" && pwd)
# The rebase checks out older revisions without this directory: work from a copy.
here=$(mktemp -d)
cp "$src/dropmap.tsv" "$src/resolve-drop-conflicts.py" "$here/"
list=$(mktemp)
# k6b reuses k6a's test helper fill_identity_data: dropping k6a drops k6b too.
comps=" ${comp//,/ } "
[[ $comps == *" k6a "* && $comps != *" k6b "* ]] && comps="$comps k6b"
awk -F'\t' -v c="$comps" 'BEGIN { n = split(c, a, " "); for (i = 1; i <= n; i++) w[a[i]] = 1 } $1 in w { print $2 }' "$here/dropmap.tsv" > "$list"
[[ -s $list ]] || { echo "unknown component: $comp" >&2; exit 2; }
git log --reverse --format='%h %s' "$base..HEAD" > "$list.log"
python3 - "$list" "$list.log" > "$list.drop" <<'PY'
import sys
subs = set(open(sys.argv[1]).read().splitlines())
hits = [l.split(' ', 1)[0] for l in open(sys.argv[2]).read().splitlines() if l.split(' ', 1)[1] in subs]
if len(hits) != len(subs):
    sys.exit(f'dropmap subjects matched {len(hits)} of {len(subs)} commits')
print('\n'.join(hits))
PY
GIT_SEQUENCE_EDITOR="python3 -c 'import re,sys; d=set(open(\"$list.drop\").read().split()); p=sys.argv[1]; L=open(p).read().splitlines(); open(p,\"w\").write(chr(10).join((\"drop\"+l[4:]) if l.startswith(\"pick \") and l.split()[1][:7] in {x[:7] for x in d} else l for l in L)+chr(10))'" \
    git -c merge.conflictstyle=diff3 rebase -i "$base" || true
# Conflicts from additions that sit next to the dropped commit's lines are resolved by resolve-drop-conflicts.py.
# Editing conflicts (mmq.cuh launch lines of K8/K9) stay: see DROPMAP.md for their manual resolution.
while [[ -d $(git rev-parse --git-path rebase-merge) ]]; do
    mapfile -t files < <(git diff --name-only --diff-filter=U)
    if ((${#files[@]} == 0)) || ! python3 "$here/resolve-drop-conflicts.py" "${files[@]}"; then
        echo "unresolved conflict in: ${files[*]}" >&2
        exit 1
    fi
    git add -- "${files[@]}"
    git -c core.editor=true -c merge.conflictstyle=diff3 rebase --continue > /dev/null || true
done
# The identity-check chain in test_backend() is edited by several components: rebuild it from the checks the tree defines.
python3 "$here/resolve-drop-conflicts.py" --normalize tests/test-backend-ops.cpp
if ! git diff --quiet -- tests/test-backend-ops.cpp; then
    git commit -q -m "tests : canonical identity-check chain after dropping $comp" -- tests/test-backend-ops.cpp
fi
