# Private package facts: 21-mtp-draft-chain

Verdict: FEATURE (performance).

## Relation to existing packages

Replaces the obsolete 21 prototype, including 29 subset work. No package 20 fused-catch-up prerequisite. The current implementation requires the compact draft head/embedding table from 18 plus 96e025496 and backend candidate infrastructure 04/12/16. np2 guard is folded into this feature; master has no chain, so it is not a standalone upstream bug fix.

## Design

One single-sequence slot set can draft all steps in one decode. Multi-sequence slot sets always use the ordinary loop, including idle peers; output capacity also keeps fallback limits at np2. Sorted backend top-k and per-step uniforms supply q for rejection verification. Unsupported model/backend/shape conditions fall back. General default is automatic on supported conditions; local form restores the owner opt-in switch. The old control patches in validation/ are provenance, not an apply series.

## Changes beyond original commits

Resolve build_sampling helper conflict by moving sampler-input installation to build_sampling_end. Resolve the MTP token/embedding assertion conflict in favor of the existing chain mixed-input path. Remove local round-trace calls. Keep mtp_chain as private implementation state; omit the common parameter and --spec-mtp-chain/LLAMA_MTP_CHAIN from root. Reserve multi-row output capacity only for eligible probabilistic draft-mtp settings and single-slot configurations. local/ restores the explicit opt-in parameter and the original diagnostics. Consolidate the coherent production change into one mbox with the original 21ba77aa8 message. Preserve 4be92a4fa and 374a465b8 as validation provenance, not production code; no test/debug environment switches ship in root.

## Upstream design intent (2026-10-08, offline)

FEATURE (performance): The existing implementation is correct; this is a capability/performance extension, not a documented bug.

Dupcheck pending (no network). Offline references are inherited from the named prerequisite packages; no new remote state was checked. No claim of maintainer endorsement. Features need an owner-filed issue first. These are private technical notes, not PR, issue, commit or reviewer text. The owner reviews the code and writes any public submission.

local/ also retains the 4be92a4fa dump, shifted-embedding and reversed-uniform controls in the local control commit. Root sampling is unchanged by these diagnostics.

Default-routing adaptation: automatic chain capacity must not change unrelated speculative types, greedy mode, p_min early-stop mode or unsupported lengths. The root capacity predicate now filters those configurations. The local series is consolidated into one diagnostic/control commit on the new general tree.

2026-10-08 np2 schedule investigation: no production defect was found. Full recorded schedules were already equal; validation lost records prefixed by a newline-free chat parser log. The parser fix keeps strict schedule equality, with real-artifact missing/changed/malformed controls. The root and local mboxes remain unchanged. Fresh Mac np1/np2 model gate and all chain/np2 mutation controls pass; see EVIDENCE.md and apply.txt. M2 can replace the whole-slot-set guard after its multi-sequence graph and RNG hand-over are integrated and validated; it is a separate capability extension.
