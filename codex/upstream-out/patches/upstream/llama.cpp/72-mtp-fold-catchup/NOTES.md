# Private package facts: 72-mtp-fold-catchup

Verdict: FEATURE (performance).

## Relation to existing packages

Separate from old package 20: this appends draft K/V writes to target verification, rather than shifting draft semantics. Prerequisites 51 and 73 cover host-backed Meta views for CPU verification. Meta fix is independently reviewable; do not hide it inside the folding feature.

## Design

Enable when the existing safety predicates accept: single sequence, one MTP head, shared model/device layout, separate draft cache, no adapters or pipeline parallelism, consecutive tokens in one bounded ubatch. Other paths retain catch-up. An armed pending_h pointer remains stable until the next target decode. The folded cache mutation must roll back if target decode fails.

## Changes beyond original commits

Resolve master conflicts by keeping only fold state: omit stack-only TP constants, deferred-logit resets and sampling fields. Remove ggml_round_trace_count. Replace LLAMA_MTP_FOLD_C opt-in with automatic supported-condition initialization. Preserve the same shape and backend safety predicates. No debug mutants/dump from cd4a9d49a ship.

## Upstream design intent (2026-10-08, offline)

FEATURE (performance): The existing implementation is correct; this is a capability/performance extension, not a documented bug.

Dupcheck pending (no network). Offline references are inherited from the named prerequisite packages; no new remote state was checked. No claim of maintainer endorsement. Features need an owner-filed issue first. These are private technical notes, not PR, issue, commit or reviewer text. The owner reviews the code and writes any public submission.

General-default safety adaptation: fold builds the draft K/V using the target graph's cparams. Require matching resolved RoPE/YaRN parameters, flash-attention cache layout and op/KQV offload policy; reject training contexts. The original opt-in form did not check these settings. This gate preserves fallback for differing custom draft contexts; it needs a fresh runtime refusal test.

local/ retains cd4a9d49a draft-distribution dumps and the shifted-hidden-row negative control. The opt-in constructor context is ported onto the general automatic default; diagnostics are absent from root.
