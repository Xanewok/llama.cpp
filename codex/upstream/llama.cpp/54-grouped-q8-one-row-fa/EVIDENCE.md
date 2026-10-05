# Evidence draft: grouped one-row q8_0 attention

This is private engineering evidence, not PR text. Root BASE is a7b94df2c; root source is e6d897de6. Local BASE is 6ef137376; local source is 3eab968dd. Each directory contains one cumulative patch and a series file. Only CUDA attention, its generated instances, and native backend-operation cases are packaged; codex test/harness files remain private.

## Source contract

The former one-row path could select direct q8 MMA only at DKQ=DV=256, ncols2=8, GQA>4. The template inventory had three q8 cases: ncols1=1,4,8. D128 and D256/GQA4 therefore retained independent per-query-head VEC reads. GQA1 has no KV rereading to remove and already reaches the measured byte floor at long KV. The new direct path groups up to eight query heads and preserves that VEC choice. Wider/asymmetric heads use existing legal f16 tile configurations, including width 16/32 when required.

Q8 widths must be multiples of 32. Consequently 80 and 112 cannot be tested as q8 widths without changing the requested geometry. They are explicitly listed as unsupported by the probe, while their f16 controls run. Valid pairs match f16 coverage: 64/64,96/96,128/128,192/128,256/256,320/256,512/512,576/512. The selector retains existing mask, ALiBi, softcap, layout, capability and KV-padding fallbacks.

The staged converter preserves the integer-to-half value exactly, then rounds q*d once. Pair conversion remains for even block counts; odd block counts use half-aligned scales. Large heads dequantize the existing dimension batches directly into shared tiles. Raw q8 staging is omitted for their single-stage pipeline. Kernel strides use raw bytes for q8, avoiding truncation of 102-byte D96 rows. Padded q8 columns do not read nonexistent sinks. No sampler code changes.

## Reproduction

CPU: CUDA_VISIBLE_DEVICES= nice -n 19 bash codex/run_q8_general_cpu.sh. Dispatch/allocation harnesses compile extracted production functions. Loader harnesses execute production loaders under CPU half/copy shims and compare every output half. Mutants disable dispatch, force global f16 scratch, remove a partial-load dimension offset, or remove the sink bound. All five fail, including an integer-cast half-scale control. The CPU shim uses a CUDA-layout block with an actual _Float16 scale; the former integer-bit shim hid the CUDA half-to-integer conversion defect. Native test-backend-ops adds D 23 cases; D eight small-head cases include sinks.

CPU evidence is in codex/logs/general-{stack,master}-{dispatch,loader}.log, codex/q8-general-attn-cpu.txt and codex/logs/general-probe-test.log. Root and stack loader bodies and the sink expression are identical. CUDA compile logs and configure logs record the sm_120, CUDA 13.2, NCCL OFF, offline configuration. Base source snapshots are exact git archives; attention files in fix snapshots reproduce topic-branch blobs. Explicit commit metadata identifies the master snapshot. No CUDA executable ran.

D instance deltas: root +10, stack +9 (3 -> 12). M root CUDA library bytes: 71,018,120 -> 72,411,504; M stack bytes: 71,724,192 -> 72,937,992. D deltas are +1,393,384 (+1.96%) and +1,213,800 (+1.69%). These compare the same compiler/architecture/configuration in each lane; stack frozen sizes include equal RUNPATH treatment. Root builds are not frozen or queued.

Both patch files were applied to exact archived bases, then every resulting changed file was compared byte-for-byte with its topic commit. Package patches exclude private codex artifacts.

## GPU item contract

codex/q8-probe4 copies probe3, retaining commit identity, deterministic tensors, full ERR reporting, strict q8 tolerance, f16 A/B ERR integrity, library/hash gates, word-bounded hazard checks, memory-release waiting, raw command/output/exit records and exact-PID termination. Real SQL fixtures come unmodified from q8-probe3/full-a1/01-gpu0-patch-{test,perf}.sql; other real fixtures retain their SOURCE ledger. New parsers were tested against these recordings and negative controls.

D requested combinations: 180; D valid cases: 150; D unsupported q8 widths: 30. D controls: 16. Full run uses both GPUs and up/patch/patch/up on each. D perf window: 250 ms per case. D canary ceiling: 300 s; D run ceiling: 1500 s; D manifest ceiling: 1800 s; G estimate: 1000 s. The kernel canary needs one paired A/B cycle to exercise the f16 integrity gate. It uses ten cases plus the same controls, before the full run; no server is started.

Freeze paths are inside the worktree, using wt freeze with QWEN38_WT_ROOT=$PWD/codex to satisfy the write boundary. SOURCE is 6ef137376 for base and 3eab968dd for fix. Binary hashes, SOURCE and all harness inputs are pinned. CPU preflight passed; gpuq submit --check passed. No queue submission occurred.

## Outstanding evidence

GPU correctness, actual bandwidth and regressions are pending. The queued item covers the requested small heads; large/asymmetric GPU coverage remains in native backend-operation cases for a later queue item. Shorter KV and untested architectures can change the crossover. Half accumulation can change results relative to VEC within the existing FLASH_ATTN_EXT tolerance; CPU tests prove loader values and split algebra, not GPU NMSE. Inherited f16 partial-group sink reads are outside this q8 change. Reported predictions are G guesses; traffic accounting is D logical loads, not measured DRAM bytes. Upstream issue/duplicate checks were not performed because network access was forbidden.
