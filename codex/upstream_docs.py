import difflib
import hashlib
from pathlib import Path

MAIN = Path('/home/xanewok/repos/qwen38-eval')
OUT = Path(__file__).resolve().parent / 'upstream-out'
documents = {name: (MAIN / name).read_text() for name in ('UPSTREAM.md', 'docs/upstream/llama-cpp.md')}
new = dict(documents)

new['UPSTREAM.md'] = new['UPSTREAM.md'].replace('Next free: `llama.cpp` 68+', 'Next free: `llama.cpp` 74+')
new['UPSTREAM.md'] = new['UPSTREAM.md'].replace('- 22-backend-sampling-grammar:', '- 68-backend-candidates-neginf: FEATURE (performance; distinct from 01, extends 16)\n- 22-backend-sampling-grammar:')
new['UPSTREAM.md'] = new['UPSTREAM.md'].replace('- 21-mtp-draft-chain, 29-draft-chain-subset: no verdict.', '- 21-mtp-draft-chain: FEATURE (performance; current device chain, np2 lifetime fallback folded in; 29 superseded).')
new['UPSTREAM.md'] = new['UPSTREAM.md'].replace('Fold `wt/chain-np2` da787bb40 (chain only when `n_seq == 1`; the gate counted drafting sequences, the chain graph counts attached samplers) and its test 374a465b8 into package 21 before any PR; not a standalone fix.', 'The proposed 7049ff0cb series folds da787bb40 into 21; 374a465b8 and 4be92a4fa remain validation provenance. Rebased Mac verification is recorded in the package; dupcheck pending (no network).')
new['UPSTREAM.md'] = new['UPSTREAM.md'].replace('- 58-recurrent-rollback-guard:', '- 69-ssm-out-proj-2d: FEATURE (performance; standalone graph reshape)\n- 71-mtp-verify-bulk-read: FEATURE (performance; independent host trim)\n- 72-mtp-fold-catchup: FEATURE (performance; after 51/73)\n- 58-recurrent-rollback-guard:')
new['UPSTREAM.md'] = new['UPSTREAM.md'].replace('- no package: host bundle (three host-side fixes): no verdict recorded', '- 70-kv-seq-rm-indexed: FEATURE (performance; indexed removal from host-trims-v2)\n- 73-meta-host-view-plan: REAL BUG (CPU Meta planner; extends 51)\n- no package: host bundle prompt borrowing / host-buffer overrides: no verdict recorded; indexed removal is covered by 70')
new['UPSTREAM.md'] = '\n'.join(line for line in new['UPSTREAM.md'].splitlines() if not line.startswith('- candidate (no package yet, 10-07): backend candidates under')) + '\n'

def row(slug, what, why, refs, verdict, evidence, action):
    return f'| [{slug}](../../patches/upstream/llama.cpp/{slug}/) | {what} | {why} | {refs} | {verdict} | {evidence} | {action} |'

rows = {
'68-backend-candidates-neginf': row('68-backend-candidates-neginf', 'Keep backend top-(k+1+n_ban), remove unique -inf-only bans on host, retain full-row fallback on ties/non-finite logits', 'ignore_eos or client bans otherwise disable the compact backend candidate path', 'Distinct from 01; extends 16/04/12; related to 22 and the broader leading-bias experiment; dupcheck pending (no network)', 'FEATURE (performance)', 'M recorded 2026-10-07: cand-neginf-depth-j5q/full-a1, two boots per arm, tokens/counters identical; M fresh Mac sampler seeds/control pass; D: -0.649/-0.260/-0.077 ms/round at 8k/32k/90k, 90k overlaps zero; stack source 062667905, not a rebased GPU run', 'Proposed 7049ff0cb series and Mac evidence in EVIDENCE.md/apply.txt; issue first; owner review; no filing clearance'),
'21-mtp-draft-chain': row('21-mtp-draft-chain', 'Current device-side MTP chain, one graph per draft round, compact embeddings, sorted candidates; single-sequence slot sets only', 'Fewer host draft hops; every np2 slot set keeps the exact step-loop fallback, including idle peers', 'Supersedes old 21/29 prototype; needs 04/12/16/18 and compact-embedding prerequisite, not 20; dupcheck pending (no network)', 'FEATURE (performance)', 'M recorded 2026-10-07 stack7-depth-j5q/full-a5: np1 tokens/acceptance identical; D integrated s7 -0.58/-0.62/-0.61 ms/round; chain alone earlier same-binary -0.07/-0.21/-0.21, small; historical CPU np2 fallback identity in chain-np2 report', 'da787bb40 folded in; 374a465b8/4be92a4fa validation provenance; general supported-condition default, opt-in control under local/; issue first and fresh rebase validation before filing'),
'69-ssm-out-proj-2d': row('69-ssm-out-proj-2d', 'Flatten SSM output-projection input across token and sequence dimensions in existing recurrent model builders', 'Expose a normal 2D matmul at np2; backend-portable graph change', 'No new kernel, sampler or public API; dupcheck pending (no network)', 'FEATURE (performance)', 'M recorded 2026-10-07 in-model Q6_K ssm_out 67 -> 38 us at np2; D integrated stack7-np2-real-j5q/full-a3 -1.130/-1.012/-1.277 ms/round, not isolated to this patch', 'Source ce555540a; standalone 7049ff0cb series, Mac np2 full-logit/token byte identity and old-rank control pass (M); issue first; no standalone GPU speed claim'),
'70-kv-seq-rm-indexed': row('70-kv-seq-rm-indexed', 'Use seq_pos_get for nonnegative sequence range removal; preserve the minimum freed physical head', 'Avoid scanning the entire KV capacity on each tail trim', 'Existing position-index contract; overlaps host-bundle indexed removal, not a separate bug; dupcheck pending (no network)', 'FEATURE (performance)', 'M historical CPU proxy at capacity 102400: np1 100.542 -> 1.505 us, np2 204.761 -> 3.178 us; D host-trims-v2 np2 combined -0.63/-0.16 (matched -0.33)/-0.41 ms/round', 'Source 36cb09139; independent scan/head oracle moved into existing test-llama-archs --seq-rm; issue first; Mac checks in apply.txt'),
'71-mtp-verify-bulk-read': row('71-mtp-verify-bulk-read', 'Read contiguous unmasked target nextn rows after catch-up, then copy each verify interval in one memcpy', 'Avoid repeated dense-output getter/synchronization calls', 'Independent from 70; v2 late pointer acquisition; dupcheck pending (no network)', 'FEATURE (performance)', 'M historical CPU/Meta np1/np2 identity and wrong-slot control in host-trims-v2 report; D combined host-trims-v2 np2 -0.63/-0.16 (matched -0.33)/-0.41 ms/round, not isolated', 'Sources 3b5176efd/f8a00adc5, final 81f954b2a path without knobs or instrumentation; issue first; fresh Mac np2 identity and wrong-slot-row control pass (M), see apply.txt'),
'72-mtp-fold-catchup': row('72-mtp-fold-catchup', 'Append the separate MTP catch-up cache writes to the target verify graph under existing safety predicates', 'Skip one draft catch-up decode for supported single-sequence batches', 'Different from old 20 shifted/fused-catch-up prototype; prerequisites 51/73; dupcheck pending (no network)', 'FEATURE (performance)', 'M recorded 2026-10-07 stack7-depth-j5q/full-a5: tokens and acceptance identical; D fold alone -0.23/-0.19/-0.22 ms/round, two boots per arm; rebased automatic-default form unmeasured on GPU', 'Runtime 7f1fe261f, report-only 186226ceb, Meta correction 376e5c8d9 split into 73; issue first; CPU/GPU failure rollback and engagement gates remain'),
'73-meta-host-view-plan': row('73-meta-host-view-plan', 'Exclude Meta buffers from both planner host-leaf-view skip conditions', 'Terminal CPU Meta VIEW/CPY/SET_ROWS otherwise omit the final subgraph boundary and abort planning', 'Extends 51 compute-path exclusion; dupcheck pending (no network)', 'REAL BUG (CPU Meta)', 'M historical rc 134 before / six mirrored-split VIEW/CPY/SET_ROWS cases pass after, nh1-fold-c REPORT-nh1-meta.md; no GPU perf effect claimed', 'Source 376e5c8d9; separate prerequisite of 72; test port excludes unrelated package 47 recursion case; fresh Mac negative control in apply.txt; owner review/dupcheck'),
}
lines = new['docs/upstream/llama-cpp.md'].splitlines()
for n, line in enumerate(lines):
    if line.startswith('| [21-mtp-draft-chain]'):
        lines[n] = rows['21-mtp-draft-chain']
    if line.startswith('| [67-gdn-indexed-rollback]'):
        lines[n] = line[:-1] + ' Getter refresh only: 2026-10-07 stack7 integrated np1 -0.58/-0.62/-0.61 ms/round (D), tokens/acceptance identical (M); getter contribution not isolated; see getter/EVIDENCE-wip.md. |'
    if line.startswith('| no package: host bundle,'):
        lines[n] = line.replace('no package: host bundle, three independent fixes', 'host bundle prompt borrowing / host-buffer override (indexed removal now 70)').replace('needs package |', 'prompt borrowing/override still need packaging; indexed removal now 70 |')

def insert_after(prefix, values):
    at = next(i for i, line in enumerate(lines) if line.startswith(prefix))
    lines[at + 1:at + 1] = values

insert_after('| [22-backend-sampling-grammar]', [rows['68-backend-candidates-neginf']])
insert_after('| [21-mtp-draft-chain]', [rows[s] for s in ('69-ssm-out-proj-2d', '71-mtp-verify-bulk-read', '72-mtp-fold-catchup')])
insert_after('| [51-meta-host-views]', [rows[s] for s in ('70-kv-seq-rm-indexed', '73-meta-host-view-plan')])
new['docs/upstream/llama-cpp.md'] = '\n'.join(lines) + '\n'
diff = ''.join(''.join(difflib.unified_diff(documents[name].splitlines(True), text.splitlines(True),
                                          fromfile='a/' + name, tofile='b/' + name)) for name, text in new.items())
(OUT / 'docs.diff').write_text(diff)
(OUT / 'docs-base-sha256.txt').write_text(''.join(hashlib.sha256(text.encode()).hexdigest() + '  ' + name + '\n' for name, text in documents.items()))
