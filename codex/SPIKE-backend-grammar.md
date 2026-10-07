# SPIKE backend-grammar: backend sampling under tool-call grammar + reasoning budget

Tree `wt/backend-grammar-spike` (branch `topic/backend-grammar-spike`, base ddedffacb = stack7 48b82b0d1 + item commits).
Labels: M measured, D derived, G guess. Line numbers are at base ddedffacb unless marked.

## Verdict

**The premise is stale for our stack. Backend (device) sampling already stays on for agent traffic** in stack4 → stack7
(serving) through package 16 `backend-verify-sampling` (e849b7504, ancestor of `frozen/stack6` ac7465885 and stack7
48b82b0d1, checked with `git merge-base --is-ancestor`). The device runs `chain(top_k(k+1))`. The host keeps the reasoning
budget, the lazy GBNF, the rest of the chain and the RNG, and it works on the k candidates. It reads the full row only on a
rare fallback, through a deferred D2H. What is left to win is **≤ ~0.06 ms/round (D, 95 % upper bound; point estimate ≈ 0)**.
That is below the 0.3 ms/round threshold, so **no new prototype and no GPU A/B.**

Fix the docs:
- `docs/levers-status.md:51` "Backend sampling for agent traffic ❌ by design" → banked (package 16, in stack4+).
- `docs/ledger/llama-cpp.md:17` (❌ for agent traffic) and row 18 (D1 "❌ blocked"): mark them superseded by package 16.
- Package 22 KNOWN LIMITATION holds **for upstream master only**. Package 16 is the upstream form of the fix (see Design).

## 1. Disable paths and what each sampler needs

Disable paths (stack7):

| where | condition | agent traffic (Swift serving: `--backend-sampling --temp 1 --top-k 20 --top-p 0.95 --min-p 0.05`, codex tools) |
|---|---|---|
| `common/sampling.cpp:562-575` `common_sampler_init` | grammar or rbudget present **and** `!use_candidates`. `use_candidates = backend_sampling && 0 < prefilter_k < 256 && prefilter_bias.empty()` (`:563`). `prefilter_k` (`:272-299`) is 0 if mirostat is set, or if any sampler ahead of top-k is active (penalties, DRY, top-n-sigma) or reads the distribution | not hit: top_k 20 comes first among the active samplers; the bias is empty (codex sends no `logit_bias`; Swift IQ4/J5q GGUFs have no `tokenizer.ggml.suppress_tokens` (M, header grep)). Hit by `ignore_eos` / `logit_bias` (harness depth items; `cand-neginf` covers -inf) |
| `common/sampling.cpp:824-825` asserts | a token sampled on the device together with a grammar or rbudget | cannot fire: the candidate chain has no `dist` |
| `tools/server/server-context.cpp:1962-1972` | `n_probs > 0 && !post_sampling_probs` (pre-sampling logprobs) → `llama_set_sampler(nullptr)` | not hit (codex asks for no logprobs, G) |
| `src/llama-context.cpp:1367-1372` | draft vocabulary smaller than the leading top-k | n/a (target context; k+1 = 21 ≪ 25 879) |
| `src/llama-context.cpp:1378-1394` (`-sm tensor`) | the chain's first active sampler must be top-k with k ≤ 256, the LM head must be split by vocab, and each shard must be ≥ k | `top_k(21)` passes (package 04) |
| upstream master `common/sampling.cpp` (`:415-424` in the 10-01 reading) | **unconditional** disable when grammar or rbudget is present | the KNOWN LIMITATION (package 22) |

Per-row fallbacks inside the candidate path (exact; they read the full row through `logits_deferred_fetch`, `src/llama-context.cpp:928-947`):
- tie or non-finite value among the top k+1 (`common/sampling.cpp:155-166`);
- reasoning budget FORCING, or `grammar_first` with an active grammar (`:811-814`) (the target always runs with `grammar_first=false`; the draft samplers that pass `true` have no grammar);
- grammar reject of the drawn token → full-row refill, full-vocab GBNF mask, chain (`:866-880`).

What each sampler in the agent chain reads:

| sampler | reads | candidate path |
|---|---|---|
| reasoning budget (IDLE/COUNTING/DONE) | nothing (`common/reasoning-budget.cpp:169-172`, pass-through) | host, accept-side state only |
| reasoning budget FORCING | full row (masks everything but the forced token) | full-row fallback; happens only with a finite budget, and we never change reasoning effort |
| lazy GBNF before its trigger | nothing (accepts all tokens) | host, trigger matched on accepted text |
| GBNF after trigger, `grammar_first=false` | the one drawn token; on reject the full vocab | 1-token check on the host; reject → full row |
| logit bias | the whole vocab by id | must be empty (or -inf only: `cand-neginf`) |
| penalties / DRY / top-n-sigma | history + whole row | must be no-ops (they are, at the serving params) |
| top-k 20 | whole row | **device**: `top_k(21)`, then the strict-order certificate on the host |
| typical-p (1.0), top-p 0.95, min-p 0.05, XTC (off), temp 1, dist | only the k kept | host over 20 entries, same RNG stream |

## 2. Prize (D)

- Host sampling with candidates: "a few µs per row" (D, `work/verify-sampling/REPORT.md` Q2). The full-row host scan that the depth harness pays because of `ignore_eos` costs 0.64/0.31/0.17 ms/round at 8k/32k/90k (M, same report). Serving does not pay it.
- Package 16 GPU A/B (stack era 09-30): seed identity held; think −0.51, **tool −0.26 ms/round** (M, `patches/upstream/llama.cpp/16-backend-verify-sampling/EVIDENCE.md`). This gain is already banked.
- What is left is the grammar-reject fallback: a full-row D2H (~1 MB, ~0.1–0.3 ms), plus a full-vocab GBNF mask of 24–32 ms at structural positions and 51 ms p50 in free text (M, CPU, r4), plus the full chain (~0.24 ms).
  How often it fires on Swift 1.5: **0 of 2,043 grammar-active positions** (M, r4 teacher-forcing on CPU, 8 responses;
  `SP/grammar-committee/r4-fastpath.txt` §3). On the 4B proxy: 17 of 27,693 (0.06 %).
  95 % upper bound for Swift: 3/2043 = 0.15 % × ~40 ms × 3.8 sampled positions per active round × 20 % active rounds
  ≈ **0.045 ms/round** (D). At the 4B rate: ≈ 0.02 ms/round (D).
- Ties: 2 in 53,240 real rows (Codex review, ledger row 51), × ~0.5 ms → < 0.001 ms/round (D).

**Total prize ≤ ~0.06 ms/round (D), < 0.3 → stop.**

## 3. Design (the upstreamable form) and exactness conditions

The design is package 16 as built (e849b7504). It is general: it does not depend on the backend, the model or TP, and it
needs no special case for the meta device.
1. `common_sampler_init`: the user asks for backend sampling, and the chain cannot run fully on the device (grammar,
   reasoning budget, or host-only samplers after top-k). If the chain is top-k-first, send `chain(top_k(k+1))` to the
   device (`common_sampler_get_backend`, `:768`) instead of disabling. The host keeps the full chain, the grammar, the
   budget and the RNG.
2. Host `set_logits` from the k+1 candidates, with a certificate: the k+1 logits are finite and strictly decreasing.
   Then drop the (k+1)-th. If the certificate fails, take the full row.
3. `llama_context` keeps the full logits row on the device (`logits_deferred`). A reader (`llama_get_logits_ith`), the
   next compute or a reserve copies it lazily, so the fallback is exact and there is no eager per-step D2H.

Exactness (token-identical to the host path at the same seed, for `common_sampler_sample_row` and for the MTP rejection
verify) holds iff all of these hold:
- (E1) Every sampler ahead of top-k is the identity at its parameters (bias empty, penalties/DRY/top-n-sigma off,
  mirostat 0), and 0 < k < 256. This is `common_sampler_prefilter_k`.
- (E2) The device top-(k+1) returns the bit-exact logits of the k+1 largest entries of the same logits tensor that the
  host would read. Under TP this needs the exact meta gather (package 04).
- (E3) The certificate holds: values finite and strictly ordered, so the top-k set and its descending order are unique.
  The host chain's own top-k (partial sort) then produces the same array, and everything downstream (typical/top-p/
  min-p/XTC/temp/dist) reads only those k entries. So the outputs are bit-equal and the RNG draws are equal.
- (E4) Every consumer that reads more than k entries takes the full row: FORCING, `grammar_first` with an active
  grammar, a grammar reject (resample over the full vocab with grammar + chain), and pre-sampling logprobs (the server
  already turns backend sampling off for those).
- (E5) No token is sampled on the device (no `dist` in the device chain), so grammar/budget accept order is unchanged.
- MTP rejection verify (`:938+`): the target p is the chain output over the same k entries. The grammar mask and the
  renormalisation run on the host; the residual draws come from `gsmpl->rng`. Both paths do identical arithmetic.

A law note that holds for both paths (not an identity issue): with `grammar_first=false`, the law is upstream's
"draw, check, resample masked" mixture, not the masked law (PR #12643). On Swift the deviation is 0 on 2,043 positions
(r4). The exact masked law needs a certificate on the post-chain support plus a top-p check (r4 §5; about 15 lines). It
fires on 0.24 % of positions.

Not recommended (prize ≈ 0): a candidate-first resample on grammar reject. It is exact only if ≥ k grammar-valid tokens
lie in a larger device top-K (K ≫ k). Without that it needs a robustness certificate on top-p/min-p against the unseen
mass of at most (k−m)·e^{L_{K+1}}. That is per-sampler, bound-aware code that upstream would not take for a 0.05 ms gain.

Upstream risks for package 16:
- `llama_get_logits_ith` used to return the sampled (candidate) logits when a backend sampler was set. Now it returns
  the full row (e849b7504 `src/llama-context.cpp` hunk). That is an API behaviour change; the PR has to state it, or add a
  separate getter.
- Async lifetime: `logits_deferred` must be flushed before any graph reuse, reserve or memory update. The commit hooks
  `process_ubatch`, `graph_compute`, `graph_reserve`, `memory_update` and `sched_reserve`. A new compute entry point that
  skips the flush would read stale logits. Package 16 EVIDENCE notes that this needs tests.
- Package 16 EVIDENCE asks for feature discussion with upstream first (it waits for #27694). Overlap: #25919 (backend
  sampling with reasoning budget), #29797.

## 3a. Owner rule check (default-on, exact fallback, multi-seq, no env)

- No env var or gate in package 16 (e849b7504): `git show | grep getenv` is empty. The candidate path is automatic
  whenever backend sampling is on and E1 holds; otherwise it falls back exactly to the full row, per row, with no flag.
  The two `LOG_DBG` lines in b61577f3c are test-only and are not part of the package.
- Default-on: upstream `common_params_sampling::backend_sampling` defaults to `false` for the target (`common/common.h:299`;
  the draft default is `true`, `:335`). Given E1–E5, the candidate path is token-identical to host sampling. So a general
  follow-up patch can turn target backend sampling on by default **whenever the chain is top-k-first** (candidate chain,
  exact) and keep host sampling otherwise. That replaces today's opt-in flag. It is a one-line policy change after
  package 16 lands. It is not proposed in the same patch, because upstream's current backend chain (full chain on the
  device, device RNG) gives the same law but not the same tokens.
- Multi-sequence: per-slot samplers, and `logits_deferred` covers every output row of the last ubatch
  (`row0`, `n`). Nothing is np1-specific.
- Patch chain for upstream: 04 (TP backend sampling, only for `-sm tensor`) → 16 (candidates + deferred row; this fixes
  "grammar/budget disable backend sampling" in general) → optional default-on policy. Each one is backend-portable.

## 4. What I implemented (commits on `topic/backend-grammar-spike`)

- b61577f3c `sampling : debug-log candidate rows and full-row grammar resamples`: two `LOG_DBG` lines, only to prove
  engagement in the CPU check. Not for any stack.
- 92d6d0687 `codex: backend-grammar CPU server check`: `codex/bg-check.sh` (bw-mac payload) and
  `codex/bg_server_check.py`. 86168070f: debug level 5, RAM wait bounded to 10 min (rc 124 `skipped: RAM`), 3 seeds.

## 5. Mac CPU evidence (M)

Run `codex/mac-runs/backend-grammar-mac-a2/` (HEAD 86168070f, `bin/bw-mac`, M4, CPU, 4 threads,
Qwen3.5-4B-UD-Q4_K_XL, same 248k tokenizer family). One `llama-server` boot with the serving sampler flags
(`--backend-sampling --temp 1 --top-k 20 --top-p 0.95 --min-p 0.05`, `--jinja`, `--cache-ram 0`, `-lv 5`).
Each request uses `/v1/chat/completions` with 2 tools, `cache_prompt: false` and a per-request seed. The candidate arm
sets `backend_sampling: true`, the host arm `false`. Result: **PASS** (`server/summary.json`, `check.log`):
- **Identity:** 6/6 pairs (seeds 1, 7, 1729 × `tool_choice` auto (lazy grammar) / required) are equal in reasoning text,
  content, tool calls and token count (160/119/118 tokens). Every response thinks first, then makes one parsed
  `exec_command` call. The reasoning budget is active: `common_sampler_init: reasoning-budget accepted prefill token` (M).
- **Engagement:** the candidate arm logs a `candidate row` on every sampled token (794 rows = 794 completion tokens). The
  host arm logs 0. Neither logs a disable warning. The context logs `setting backend sampler for seq_id 0 (n = 1)`, which
  is the `top_k(21)` chain.
- **Negative control:** a request with `top_k: 0` logs `backend sampling is not compatible with ...` and reads 0 candidate
  rows. So the warning detector and the disable path both fire when the condition holds.
- **Comparator control:** cand seed 1 vs host seed 7 differ, so the comparison can see a difference.
- **Not covered:** 0 grammar resamples in all 13 requests (`required` did not cause rejects either). The full-row reject
  fallback is covered only by package 16's unit tests (grammar forcing / off-support residual, a corrupt-candidate
  mutant changes 243/256 seeds). A server-level reject case would need a grammar the model fights. That is G-level
  value; skipped.
- a1 (92d6d0687) was aborted by me: at `-lv 4`, `LOG_DBG` (level 5) is silent, so engagement read 0. That was a test
  bug, fixed in 86168070f. Its one finished pair (seed 1 auto) was already identical.
- Mac rule (coordinator): the RAM wait is bounded to 10 min; then the run exits 124 with `skipped: RAM`. a2 did not
  hit it.

Not shown on the Mac: TP (`-sm tensor`) and multi-sequence. Both run in serving today: stack6/7 `-duo` (np2) uses the
same per-slot `llama_set_sampler(common_sampler_get_backend)`. Package 16 GPU seed identity was np1 (M); np2 has no
separate identity check (D).

## 6. GPU item / build window

None prepared. The prize is under the threshold. The engagement and exactness on the serving path are already covered by
package 16's GPU seed-identity A/B and by the CPU check above. If a direct GPU confirmation is ever wanted, use the harness
agent shape (`harness/agents.py`, real codex runs → lazy tool grammar). The existing `stack6-two-agent-iq4.toml` form,
with arms that differ only in `--backend-sampling`, measures the same thing as package 16's "tool" row. Expected result:
≤ 0.3 ms/round (G), inside the boot spread.

## 7. Next

- Parent: apply the doc fixes in the Verdict. Nothing to submit.
- If grammar work comes back: the exact masked-law certificate (r4 §5) is the only grammar-sampling item that changes
  output quality, and it costs ≈ 0. Upstreaming package 16 is a feature discussion (see risks), not a perf emergency.
