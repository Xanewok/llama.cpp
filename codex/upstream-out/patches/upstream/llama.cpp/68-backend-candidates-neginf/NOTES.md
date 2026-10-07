# Private package facts: 68-backend-candidates-neginf

Verdict: FEATURE (performance).

## Relation to existing packages

Distinct from 01: 01 maps vocabulary IDs inside backend logit-bias sampling. This topic keeps a backend top-k candidate prefix under unique -inf-only host bans. It extends 16, which depends on 04 and 12; 22 offloads the ordinary chain under grammar/budget, a separate mechanism.

## Design

Only unique -inf bans qualify. Device top-(k+1+n_ban) contains the best k+1 unbanned IDs. Finite logits and strict ordering certify the host candidate set; ties or non-finite logits read the full row. Finite/duplicate bias entries or an oversized candidate prefix retain the ordinary chain. No RNG algorithm changes. The no-extension mutant must fail with the shipped synthetic/Meta(CPU,CPU) test.

## Changes beyond original commits

Test include conflict: retain llama-sampler.h, omit the unrelated pre-existing ggml-backend-impl.h include from the stack test. Bound candidate_k and compare the ban count in size_t before narrowing it; this avoids signed overflow or truncation for extreme caller-supplied counts. Valid candidate paths are unchanged. Omit 935c302b9 debug logging from the production patch; it belongs to the old engagement harness.

## Upstream design intent (2026-10-08, offline)

FEATURE (performance): The existing implementation is correct; this is a capability/performance extension, not a documented bug.

Dupcheck pending (no network). Offline references are inherited from the named prerequisite packages; no new remote state was checked. No claim of maintainer endorsement. Features need an owner-filed issue first. These are private technical notes, not PR, issue, commit or reviewer text. The owner reviews the code and writes any public submission.
