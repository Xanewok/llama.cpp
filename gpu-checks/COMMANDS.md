# T-8 parent GPU gates

Run these only in a parent-owned gpuq item after committing, building, and freezing `frozen/t8-gdn`. The worker did not submit or execute GPU work. Use the frozen libraries with the frozen binaries, and retain every log and exit status. Affinity is `taskset -c 6-11,18-23 nice -n 19` throughout.

Build the freeze with the fs2 recipe and include `llama-server`, `test-backend-ops`, `test-recurrent-state-rollback`, and `test-llama-archs`. Copy `llama-server` to `round-trace` as in the fs2 freeze. Keep `SOURCE` and `RECIPE.txt` consistent with the new commit. Freeze the complete test executables and shared libraries.

## Operator identity

For each UUID below, run both graph settings. `-o 'GATED_DELTA_NET.*'` includes the existing tests, all new accept-length tests, fused cache writes, and the changing-input same-address graph replay check. All comparisons between snapshot and replay paths use `memcmp` within that GPU, including signed zero; CPU-to-CUDA tolerance is not the identity gate.

```sh
T8_ROOT=/home/xanewok/repos/qwen38-eval-worktrees/frozen/t8-gdn
for T8_GPU in GPU-ee9a3956-297c-604b-b4c1-f07930d27e46 GPU-3fe72007-d2f3-11c0-4d57-cc6fbde1451f; do
    for T8_DISABLE_GRAPHS in 0 1; do
        taskset -c 6-11,18-23 nice -n 19 env CUDA_VISIBLE_DEVICES="$T8_GPU" LD_LIBRARY_PATH="$T8_ROOT/build/bin" GGML_CUDA_DISABLE_GRAPHS="$T8_DISABLE_GRAPHS" "$T8_ROOT/build/bin/test-backend-ops" test -b CUDA0 -o 'GATED_DELTA_NET.*'
    done
done
```

Require zero failures and a positive `changing-input graph replay byte identity PASS cases=224 rounds=4` line in every run. The new regular cases cover S=16/32/64/128, scalar/KDA gates, K=1..6, np1/np2, every independent acceptance pair, reversed state indices and retention slots, and cache fusion. The changing-input check also requires its one-bit negative control to fail comparison. The frozen fs2 arm is checked separately by the full-model token-identity experiments below; the new operator test alone cannot prove that recompilation leaves the frozen CUDA arithmetic unchanged.

## Model state, restore, and graph switching

Generate a tiny qwen35 fixture with the frozen `test-llama-archs -a '^qwen35$' -o FIXTURE_DIR`, then use the generated gguf below as `T8_MODEL`. In a gpuq item, run `test-recurrent-state-rollback --gdn-state-movers` with `LLAMA_GDN_REPLAY=0` and `1`, for `-np 1` and `-np 2`, with both devices visible, `-ngl 99 -sm tensor --tensor-split 50,50 -t 1 -tb 1 -lv 5`. Run this matrix with CUDA graphs enabled and disabled. Set `LLAMA_GDN_REPLAY_LOG=1` only for the enabled proof run.

Compare every `STATE_MOVERS` stdout line byte-for-byte between the two switch settings. Require both exit statuses zero, `negative_controls=PASS`, and positive `gdn_replay: enabled` and `gdn_replay: restored` lines with the switch on. This exercises the actual model builder, changed data and graph plans, short continuation batches, immediate selected-state serialization, restore, sequence removal, and full accept. It complements the model-free two-layer CPU tests for copy and unequal np2 accepts.

## Served-stack identity and timing

Two harness window specs preserve the requested fs2 templates, including flags, model and prompt hashes, depths, request seed, output length, steady minimum, ABBA order, and real np2 joint start. Both use `identity = "cross-arm"`. Both arms receive the same environment. The new retention and partial-replay proof lines are required only for `t8-gdn` canaries; measurement boots do not enable replay logging.

After the parent freezes `t8-gdn`, expand from `/home/xanewok/repos/qwen38-eval`:

```sh
python3 -m harness.window /home/xanewok/repos/qwen38-eval-worktrees/wt/t8-gdn/gpu-checks/t8-gdn-fs2-np1-depth-iq4-window.toml
python3 -m harness.window /home/xanewok/repos/qwen38-eval-worktrees/wt/t8-gdn/gpu-checks/t8-gdn-fs2-np2-real-iq4-window.toml
```

The expansion writes ordinary experiment TOMLs beside each input and resolves immutable arm commits from each frozen `SOURCE`. Then the parent runs `python3 -m harness.mkitem EXPERIMENT --check` and submits each item through its normal gpuq workflow. No future commit hash is fabricated in these inputs. Finish and pin `REPORT-t8-gdn.md` before rendering an item.

Require both self-identity and cross-arm token identity, canary proof, matched acceptance counters, at least 40 steady intervals and 256 predicted tokens, and np2 overlap >= 0.8. Inspect min free VRAM and first-use replay overhead as well as steady round time. Any byte or token difference blocks the optimization even if timing improves.
