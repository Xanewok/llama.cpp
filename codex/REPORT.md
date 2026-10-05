# Streaming UTF-8 token accounting

Fixed on master: `352725cc5`, parent `a7b94df2c`, branch `topic/stream-tokens-utf8`.
Clean cherry-pick on stack: `f80185077`, parent `98c695111`, branch `topic/stream-tokens-utf8-stack`.

Cause: `process_token` suppresses partial responses until UTF-8 is complete, but the next response previously returned only its newest ID. Probability recording was also inside that gate. Nightly evidence: M: 507 IDs versus 512 predictions; five split `\u2554` characters each lost one ID.

Design: collect IDs for streaming or `return_tokens`; reset `n_sent_tokens` with the slot; send the unsent suffix with each text delta. Save probability records before text-delta mutation and serialize the same suffix as a vector. A token limit inside UTF-8 flushes IDs/probs in an empty-text partial. Sampling/RNG code is unchanged; original sampled IDs are retained.

Evidence (master / stack): `tools/server/server-context.cpp:278 / :281` cursor, `:1912 / :2013` recording, `:2133 / :2234` suffix, `:2159 / :2260` final flush; `tools/server/server-task.cpp:1067` native serialization and `:1149` OAI chat. Tests: `tools/server/tests/unit/test_completion.py:103` native and `:146` chat.

CPU validation, GPUs hidden, nice 19, inference/build limit eight threads:
- M: master 18 passed; stack 18 passed. Ten new and eight existing cases. Cached `stories260K-f32.gguf`; grammar forces split UTF-8. IDs equal non-stream IDs and `predicted_n`; probability IDs/bytes and OAI chat content are checked.
- Negative control: frozen base executable AND shared libraries via `LD_LIBRARY_PATH`. M: five failed, five passed (missing IDs and chat probabilities).
- M: five harness parser groups passed on recorded artifacts; shell syntax passed. No GPU execution.

Commands: CPU configure flags are in `validation.json`; `CCACHE_DISABLE=1 nice -n 19 cmake --build build-cpu --target llama-server -j8`. Stack uses `build-stack-cpu`. Run `$SP/venv/bin/python codex/run_cpu_tests.py <renamed-binary> -q -m 'not slow' -k 'buffered_tokens or completion_stream_vs_non_stream or test_n_probs or test_completion_stream'` under nice 19. Runner uses the cached model, disables downloads, and sets `CUDA_VISIBLE_DEVICES=""`. See `validation.json` and logs.

Other fields/risks: partial stop strings already emit empty-text chunks with IDs; no loss there. Existing non-stream stop-word probability trimming, incomplete-final-text handling, and chat logprobs without message deltas remain. Speculative probabilities remain an existing TODO; MTP/GPU execution is untested. No speed claim; fake server timings are not performance evidence.

Parent A/B: freeze at the fix commits; rename to `stream-tokens-srv`, hash binary/all libraries in `SHA256SUMS`, identity in `../SOURCE`; set `UP_BIN_DIR`, `PATCH_BIN_DIR`, GPU devices. Queue `gpu_ab.sh --canary` first, then full interleaved boots. Requires fresh output; checks seed/count identity, expected base loss, budget, kernel hazards, exact-PID cleanup.
