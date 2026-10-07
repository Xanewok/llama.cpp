# Private package facts: 70-kv-seq-rm-indexed

Verdict: FEATURE (performance).

## Relation to existing packages

Independent part of host trims; not bundled with 71. It reuses the existing position index introduced upstream, not a new index. The old scan is correct, so this is not a bug verdict.

## Design

For a nonnegative sequence ID, walk [p0,p1) in seq_pos_get, advancing before seq_rm erases the entry. Distinct cell removals commute; choosing the minimum freed physical index preserves the old scan head. Keep the negative-sequence full scan. Tests compare position/membership/index state with an independent scan oracle, reversed and repeated ranges, shared cells, nonmonotonic positions and the next physical head.

## Changes beyond original commits

Move the original test_seq_rm oracle from the stack-only test-kv-mask.cpp into existing test-llama-archs.cpp; add --seq-rm test dispatch and private llama-kv-cache.h/llama-model.h/limits includes. Use the existing GGUF mock-model constructor instead of the stack-only llama_model_create declaration. No production changes. No new upstream test file.

## Upstream design intent (2026-10-08, offline)

FEATURE (performance): The existing implementation is correct; this is a capability/performance extension, not a documented bug.

Dupcheck pending (no network). Offline references are inherited from the named prerequisite packages; no new remote state was checked. No claim of maintainer endorsement. Features need an owner-filed issue first. These are private technical notes, not PR, issue, commit or reviewer text. The owner reviews the code and writes any public submission.
