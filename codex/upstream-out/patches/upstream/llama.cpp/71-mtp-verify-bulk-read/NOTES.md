# Private package facts: 71-mtp-verify-bulk-read

Verdict: FEATURE (performance).

## Relation to existing packages

Independent host trim, no prerequisites. v2 fixes the early-pointer lifetime assumption in the local v1; knobs/proof logging/mutants are not production code.

## Design

The target nextn rows are unmasked and indexed by raw token row; each sequence already requires a contiguous interval. Acquire the dense pointer after catch-up, then copy that interval in one memcpy. Keep the earlier pointer scoped to catch-up input construction. Sampling/RNG are unchanged.

## Changes beyond original commits

Reconstruct the final v2 default path against master: leave the catch-up getter where upstream has it, add one post-catch-up getter, replace the verify per-row loop with a single memcpy. Exclude local knobs, counters, cross-checks and mutants.

## Upstream design intent (2026-10-08, offline)

FEATURE (performance): The existing implementation is correct; this is a capability/performance extension, not a documented bug.

Dupcheck pending (no network). Offline references are inherited from the named prerequisite packages; no new remote state was checked. No claim of maintainer endorsement. Features need an owner-filed issue first. These are private technical notes, not PR, issue, commit or reviewer text. The owner reviews the code and writes any public submission.
