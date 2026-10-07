# Private package facts: 73-meta-host-view-plan

Verdict: REAL BUG (CPU Meta planning).

## Relation to existing packages

Extends 51 to the planner census and graph-plan creation paths. 51 protects compute dispatch only. It is a distinct missing-condition defect in the same host-backed Meta view contract; keep 51 as prerequisite so all relevant paths obey the contract.

## Design

Meta buffers can report host when backed by CPU, while their data pointers remain Meta placeholders. Planning cannot skip these views as native host no-ops. Six fixture cases compare actual cache bytes after three direct computations with the same graph UID (the Meta internal planning cache is reused).

## Changes beyond original commits

Move the existing test_meta_split_state_mirrored helper and six host-view cases into master test-alloc.cpp without the unrelated package 47 recursion test; add the cstring include. Production changes unchanged. Package 51 remains a prerequisite.

## Upstream design intent (2026-10-08, offline)

REAL BUG (CPU Meta planning): The existing host-backed Meta buffer contract excludes treating placeholder views as native host pointers.

Dupcheck pending (no network). Offline references are inherited from the named prerequisite packages; no new remote state was checked. No claim of maintainer endorsement. Features need an owner-filed issue first. These are private technical notes, not PR, issue, commit or reviewer text. The owner reviews the code and writes any public submission.

Test-port correction after Mac compile: use the existing ggml-backend.h declarations; no additional Meta header exists or is required.

Mac test-port correction: master has no Meta graph_plan_create/compute hooks. Use three graph_compute calls with one graph UID, testing initial planning and internal-plan reuse instead of the stack-only explicit graph-plan API. No production changes.
