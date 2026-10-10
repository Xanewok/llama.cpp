# spec-tf-j5q: CPU recovery of full-a1

Labels: M observed in recorded artifacts or this run; D calculated from observations; G teacher-forced predictions whose transfer to served rounds is not validated.

## Conclusion

Teacher-forced acceptance predictions are recoverable from the completed real-head dumps without another GPU extraction. Both sliced-head controls pass; Q3_K reproduces the registered negative direction. The parked control failed because the full head's saved top-128 cannot reconstruct every subset top-10. Exact served-law claims remain limited by start-selection bias and the target-stream support mismatch below.

Results: `/mnt/scratch/work/spec-tf/reanalysis-a1/analysis/results.json`; detailed estimator outputs: `analysis/estimate/summary.json` and `analysis/estimate/*.npz`. Source dumps remain at `/mnt/scratch/work/spec-tf/full-a1/`.

## Validity and coverage

M: `target`, `base`, `q3_k`, `iq3_s`, `q8_0`, `full`, and `cov99` each recorded `returncode=0`, `fault_rc=0`. Each extraction log ends with completion of stream 35. M: 36 streams, prompt depths 1500 through 19000, 1536 continuation tokens per stream. The extractor used k=5, not k=3.

M: target has 252 binary files, 8,842,328,640 bytes. Each draft arm has 180 binary files, 305,256,240 bytes. `codex/reanalyse_spec_tf.py:28` (`inventory`) checks every binary size, recorded tokens, starts, exact target headers, and stream completion logs. Hidden files are checked by size only, never loaded; the largest is 420,577,280 bytes. Probability arrays use read-only memory maps. All source file sizes and mtimes remained unchanged during reanalysis (M).

M: `base` offline reconstruction has one insufficient row: stream `depth-2500` (index 2), start 3369, step 3, target/draft position 3371. Only 9 eligible subset IDs are retained in the saved top-128, against draft `top_k=10`; the actual subset contains 25,879 IDs. Retained IDs: `17, 653, 19, 18, 23, 220, 24, 16, 15`. The missing subset logit cannot be recovered from these dumps.

## Control contract and fix

`codex/spec_tf.py:22` (`draft_candidates`) exposes separate insufficient-count, unproven-boundary, and boundary-tie masks. `draft_probs` still refuses any unproven offline row. `codex/check_vocab.py:11` (`compare_rows`) compares only provably covered rows against real sampler probabilities, including missing real probability mass. `codex/check_vocab.py:26` (`main`) writes exclusions, their positions and reasons, and complete-chain coverage. It returns failure on any compared discrepancy above 1e-3, or when no rows can be compared.

An offline acceptance delta uses only complete covered chains, with the same starts and real-arm weights on both sides (`codex/spec_tf.py:75`, `load_stream`). Its output explicitly marks this scope; it is not a full-population estimate. `codex/gpu_payload.py:163` estimates acceptance from the real `base` and `cov99` slices already extracted, instead of trying to estimate those populations again from sparse full-head logits. Sampling code and RNG behavior are unchanged.

## Acceptance results

M: each arm retains all 55,116 complete starts across 36 streams, including the start excluded only from the offline control. All estimates below use stored real sampler probabilities. SE is D; no sampled GPU acceptance rate is claimed.

| Arm | Step-1 overlap mean +/- SE (D) | Harmonic tokens/round (G) +/- SE (D) | Paired delta vs base (G) +/- SE (D) | Arithmetic mean T (G) +/- SE (D) |
|---|---:|---:|---:|---:|
| `base` | 0.904788 +/- 0.009718 | 3.718556 +/- 0.138988 | +0.000000 +/- 0.000000 | 4.695280 +/- 0.120192 |
| `q3_k` | 0.902015 +/- 0.009996 | 3.671106 +/- 0.139889 | -0.030811 +/- 0.002716 | 4.644480 +/- 0.122742 |
| `iq3_s` | 0.899701 +/- 0.010438 | 3.619335 +/- 0.142721 | -0.056566 +/- 0.007774 | 4.608175 +/- 0.128221 |
| `q8_0` | 0.906511 +/- 0.009578 | 3.745015 +/- 0.138974 | +0.028585 +/- 0.002618 | 4.720501 +/- 0.118947 |
| `cov99` | 0.935665 +/- 0.010608 | 4.320939 +/- 0.205460 | +0.791170 +/- 0.108062 | 5.014253 +/- 0.133246 |
| `full` | 0.941578 +/- 0.009318 | 4.459460 +/- 0.190396 | +0.909035 +/- 0.097981 | 5.066163 +/- 0.127166 |

The paired delta uses fixed base weights and therefore is not the difference of the two harmonic point estimates.

Per-step probability of reaching an accepted draft (G), with stream SE (D):

| Arm | Step 1 | Step 2 | Step 3 | Step 4 | Step 5 |
|---|---:|---:|---:|---:|---:|
| `base` | 0.904788 +/- 0.009718 | 0.812382 +/- 0.019621 | 0.728883 +/- 0.026839 | 0.656042 +/- 0.031110 | 0.593185 +/- 0.033370 |
| `q3_k` | 0.902015 +/- 0.009996 | 0.806508 +/- 0.020089 | 0.719662 +/- 0.027495 | 0.642383 +/- 0.031828 | 0.573912 +/- 0.033851 |
| `iq3_s` | 0.899701 +/- 0.010438 | 0.800445 +/- 0.021084 | 0.709290 +/- 0.028716 | 0.632032 +/- 0.033170 | 0.566706 +/- 0.035353 |
| `q8_0` | 0.906511 +/- 0.009578 | 0.815830 +/- 0.019322 | 0.734153 +/- 0.026564 | 0.662772 +/- 0.030831 | 0.601235 +/- 0.033088 |
| `cov99` | 0.935665 +/- 0.010608 | 0.862984 +/- 0.021256 | 0.795960 +/- 0.029352 | 0.735904 +/- 0.034587 | 0.683740 +/- 0.037897 |
| `full` | 0.941578 +/- 0.009318 | 0.872280 +/- 0.019636 | 0.807160 +/- 0.027931 | 0.748228 +/- 0.033474 | 0.696918 +/- 0.037094 |

Both real sliced-head controls PASS against the registered 1e-3 probability tolerance:

| Slice | Compared / eligible rows (M) | Excluded fraction (D) | Covered / complete starts (M) | Maximum probability error (M) | Offline minus real delta on covered chains (D) |
|---|---:|---:|---:|---:|---:|
| `base` | 275,939 / 275,940 | 3.623976227e-06 | 55,115 / 55,116 | 4.17464e-08 | -4.697946633e-09 |
| `cov99` | 275,940 / 275,940 | 0 | 55,116 / 55,116 | 4.17464e-08 | -5.838919508e-09 |

D: the base row exclusion fraction is 0.0003623976%; cov99 has zero exclusions. The offline control deltas are numerical equivalence diagnostics over covered chains, not new arm efficacy estimates.

Pre-registered Q3_K direction: PASS (negative paired delta). Other arm improvements are descriptive; the design supplies no separate decision threshold.

## Per-depth estimates

Each cell is paired delta (G) +/- per-start bootstrap SE (D), with 2000 draws and seed 1729. This SE ignores dependence between neighboring starts; there is one stream per depth and no independent within-depth stream SE. Base delta and its SE are exactly zero. Per-depth overlap, harmonic and arithmetic estimates, and all step probabilities are in `analysis/results.json`.

| Prompt tokens (M) | q3_k | iq3_s | q8_0 | cov99 | full |
|---:|---:|---:|---:|---:|---:|
| 1500 | -0.033315 +/- 0.008339 | -0.045880 +/- 0.009888 | +0.029227 +/- 0.006663 | +0.580616 +/- 0.074083 | +0.625057 +/- 0.075821 |
| 2000 | -0.023915 +/- 0.005949 | -0.061386 +/- 0.009201 | +0.018317 +/- 0.003817 | +0.005663 +/- 0.002798 | +0.397510 +/- 0.031759 |
| 2500 | -0.029257 +/- 0.010918 | -0.086818 +/- 0.013784 | +0.050282 +/- 0.005884 | -0.001202 +/- 0.000601 | +0.252079 +/- 0.041977 |
| 3000 | -0.034906 +/- 0.011778 | -0.098975 +/- 0.013990 | +0.039705 +/- 0.006105 | +0.361104 +/- 0.058938 | +0.580337 +/- 0.066878 |
| 3500 | -0.057237 +/- 0.007591 | -0.054021 +/- 0.010782 | +0.020798 +/- 0.005908 | +0.034600 +/- 0.021452 | +0.085464 +/- 0.030940 |
| 4000 | -0.046134 +/- 0.007776 | -0.087079 +/- 0.011001 | +0.035625 +/- 0.004694 | +0.016051 +/- 0.006437 | +0.517360 +/- 0.041847 |
| 4500 | -0.066223 +/- 0.010337 | -0.130973 +/- 0.013312 | +0.029489 +/- 0.006838 | +0.118416 +/- 0.026641 | +0.492949 +/- 0.049552 |
| 5000 | -0.020585 +/- 0.012026 | -0.109874 +/- 0.015371 | +0.055316 +/- 0.007002 | +0.059789 +/- 0.023617 | +0.190783 +/- 0.040438 |
| 5500 | -0.029716 +/- 0.008697 | -0.104478 +/- 0.014433 | +0.037466 +/- 0.005863 | +0.458634 +/- 0.055283 | +0.592813 +/- 0.059757 |
| 6000 | -0.023471 +/- 0.006921 | -0.039820 +/- 0.009034 | +0.028573 +/- 0.004192 | +0.421490 +/- 0.058060 | +1.120717 +/- 0.079167 |
| 6500 | -0.003404 +/- 0.004819 | +0.059971 +/- 0.007836 | +0.035444 +/- 0.005067 | -0.000048 +/- 0.000013 | -0.000120 +/- 0.000032 |
| 7000 | -0.005479 +/- 0.003358 | +0.012990 +/- 0.004527 | +0.006360 +/- 0.001634 | +1.500974 +/- 0.092544 | +1.500898 +/- 0.091099 |
| 7500 | -0.001767 +/- 0.003370 | +0.002752 +/- 0.004349 | +0.004945 +/- 0.001498 | +1.775882 +/- 0.089141 | +1.775807 +/- 0.091087 |
| 8000 | -0.030113 +/- 0.005154 | -0.010023 +/- 0.005740 | +0.016147 +/- 0.003211 | +1.400098 +/- 0.090686 | +1.525698 +/- 0.089793 |
| 8500 | -0.049324 +/- 0.006505 | -0.035709 +/- 0.008423 | +0.021615 +/- 0.003266 | +0.790423 +/- 0.072426 | +1.471949 +/- 0.082098 |
| 9000 | -0.028366 +/- 0.008572 | -0.030813 +/- 0.009780 | +0.018429 +/- 0.003728 | +0.071771 +/- 0.028495 | +0.603308 +/- 0.073565 |
| 9500 | -0.010223 +/- 0.008400 | -0.047467 +/- 0.011860 | +0.019464 +/- 0.005281 | +0.977543 +/- 0.076592 | +0.977138 +/- 0.074720 |
| 10000 | -0.030039 +/- 0.005152 | -0.039137 +/- 0.007012 | +0.005857 +/- 0.002381 | +2.031677 +/- 0.078196 | +2.031307 +/- 0.078020 |
| 10500 | -0.017275 +/- 0.004488 | -0.136917 +/- 0.013114 | +0.005526 +/- 0.002516 | +1.433274 +/- 0.087115 | +1.433234 +/- 0.085579 |
| 11000 | -0.034084 +/- 0.010471 | -0.123211 +/- 0.012294 | +0.051695 +/- 0.005926 | +0.549788 +/- 0.049534 | +0.560374 +/- 0.049354 |
| 11500 | -0.034035 +/- 0.009431 | -0.080550 +/- 0.011322 | +0.060403 +/- 0.005954 | +0.496168 +/- 0.044369 | +0.497681 +/- 0.044203 |
| 12000 | -0.051050 +/- 0.009415 | -0.090594 +/- 0.012084 | +0.038821 +/- 0.005769 | +0.702336 +/- 0.058365 | +0.701048 +/- 0.055842 |
| 12500 | -0.042887 +/- 0.009767 | -0.097515 +/- 0.013591 | +0.043789 +/- 0.005752 | +0.537380 +/- 0.049454 | +0.567045 +/- 0.047504 |
| 13000 | -0.025584 +/- 0.009557 | -0.073548 +/- 0.011981 | +0.040444 +/- 0.005596 | +1.111863 +/- 0.071903 | +1.111176 +/- 0.073877 |
| 13500 | -0.025465 +/- 0.011587 | -0.140141 +/- 0.013799 | +0.056894 +/- 0.006419 | +0.518305 +/- 0.050193 | +0.529206 +/- 0.049799 |
| 14000 | -0.024183 +/- 0.008012 | -0.031839 +/- 0.009700 | +0.031016 +/- 0.004171 | +0.598172 +/- 0.068376 | +0.608240 +/- 0.071135 |
| 14500 | -0.046060 +/- 0.009951 | -0.032979 +/- 0.013428 | +0.047643 +/- 0.005648 | +0.148895 +/- 0.043024 | +0.148667 +/- 0.042275 |
| 15000 | -0.060728 +/- 0.008945 | -0.042920 +/- 0.010137 | +0.030484 +/- 0.005248 | +0.071039 +/- 0.024737 | +0.102694 +/- 0.030198 |
| 15500 | -0.061851 +/- 0.009205 | -0.101139 +/- 0.012054 | +0.026814 +/- 0.004989 | +0.957393 +/- 0.067378 | +0.972966 +/- 0.066966 |
| 16000 | -0.036895 +/- 0.007488 | -0.041749 +/- 0.010374 | +0.025324 +/- 0.005053 | +1.236515 +/- 0.076101 | +1.235834 +/- 0.075927 |
| 16500 | -0.019832 +/- 0.003412 | +0.006650 +/- 0.004525 | +0.008266 +/- 0.001625 | +2.018152 +/- 0.089549 | +2.017972 +/- 0.086498 |
| 17000 | -0.009937 +/- 0.006590 | -0.041958 +/- 0.006459 | +0.018279 +/- 0.003124 | +1.364569 +/- 0.087687 | +1.364260 +/- 0.088535 |
| 17500 | -0.020487 +/- 0.007378 | -0.060674 +/- 0.009231 | +0.030501 +/- 0.004617 | +1.424709 +/- 0.085031 | +1.424425 +/- 0.082785 |
| 18000 | -0.033919 +/- 0.004093 | +0.010355 +/- 0.005399 | +0.008888 +/- 0.002395 | +1.945282 +/- 0.092179 | +1.945036 +/- 0.087609 |
| 18500 | -0.020702 +/- 0.006770 | -0.021388 +/- 0.007964 | +0.019602 +/- 0.003683 | +1.412461 +/- 0.088051 | +1.412088 +/- 0.087671 |
| 19000 | -0.020765 +/- 0.006941 | -0.029522 +/- 0.008579 | +0.011608 +/- 0.003321 | +1.352339 +/- 0.085907 | +1.352270 +/- 0.086609 |

Per-depth harmonic tokens/round (G). Per-depth stream SE is not identifiable from one stream.

| Prompt tokens (M) | base | q3_k | iq3_s | q8_0 | cov99 | full |
|---:|---:|---:|---:|---:|---:|---:|
| 1500 | 4.638081 | 4.590846 | 4.546960 | 4.665741 | 5.173649 | 5.223703 |
| 2000 | 2.126970 | 2.098747 | 2.054162 | 2.139960 | 2.132160 | 2.390081 |
| 2500 | 3.915010 | 3.855629 | 3.763618 | 3.964649 | 3.913421 | 4.164540 |
| 3000 | 4.373056 | 4.312708 | 4.223245 | 4.410626 | 4.680037 | 4.889635 |
| 3500 | 4.903845 | 4.817207 | 4.793443 | 4.924769 | 4.936255 | 4.984042 |
| 4000 | 2.473155 | 2.417154 | 2.369990 | 2.500732 | 2.486290 | 2.831178 |
| 4500 | 3.054777 | 2.962410 | 2.879415 | 3.068780 | 3.129295 | 3.390354 |
| 5000 | 4.265740 | 4.216768 | 4.084430 | 4.323010 | 4.318282 | 4.443509 |
| 5500 | 3.765809 | 3.707381 | 3.591970 | 3.796783 | 4.093579 | 4.229047 |
| 6000 | 4.053037 | 4.024769 | 3.989018 | 4.083000 | 4.370321 | 5.061048 |
| 6500 | 5.540596 | 5.527737 | 5.599659 | 5.580011 | 5.540534 | 5.540455 |
| 7000 | 4.174705 | 4.166599 | 4.179394 | 4.180713 | 5.594606 | 5.594522 |
| 7500 | 3.968737 | 3.965286 | 3.969103 | 3.973862 | 5.668261 | 5.668168 |
| 8000 | 4.034593 | 4.003102 | 4.021765 | 4.049238 | 5.311059 | 5.466892 |
| 8500 | 3.925188 | 3.874436 | 3.882495 | 3.944719 | 4.562176 | 5.298790 |
| 9000 | 4.912738 | 4.877130 | 4.869926 | 4.937689 | 4.985938 | 5.481157 |
| 9500 | 3.991338 | 3.969652 | 3.909751 | 4.006502 | 4.839405 | 4.839183 |
| 10000 | 3.478509 | 3.451061 | 3.433121 | 3.483138 | 5.359006 | 5.358533 |
| 10500 | 4.171959 | 4.152324 | 3.996913 | 4.176962 | 5.509884 | 5.509846 |
| 11000 | 2.976750 | 2.917712 | 2.816029 | 3.016779 | 3.383318 | 3.389612 |
| 11500 | 2.767467 | 2.711249 | 2.654060 | 2.813008 | 3.113231 | 3.114816 |
| 12000 | 3.198474 | 3.122263 | 3.073641 | 3.226626 | 3.722067 | 3.721296 |
| 12500 | 2.974196 | 2.906406 | 2.838535 | 3.008060 | 3.355998 | 3.389901 |
| 13000 | 3.440540 | 3.402318 | 3.330860 | 3.473760 | 4.327109 | 4.326461 |
| 13500 | 3.027033 | 2.976675 | 2.844294 | 3.072820 | 3.383650 | 3.393912 |
| 14000 | 4.675426 | 4.638210 | 4.612322 | 4.707971 | 5.224526 | 5.234206 |
| 14500 | 5.031526 | 4.954739 | 4.943305 | 5.084383 | 5.160484 | 5.160247 |
| 15000 | 5.078134 | 4.998676 | 4.997778 | 5.113617 | 5.157114 | 5.187276 |
| 15500 | 3.490301 | 3.414050 | 3.351482 | 3.508666 | 4.287025 | 4.303051 |
| 16000 | 3.840710 | 3.794173 | 3.778620 | 3.863271 | 4.948899 | 4.947548 |
| 16500 | 3.724195 | 3.707102 | 3.721952 | 3.731621 | 5.628684 | 5.628426 |
| 17000 | 4.176424 | 4.165410 | 4.128944 | 4.194013 | 5.462704 | 5.462333 |
| 17500 | 3.597369 | 3.572572 | 3.508958 | 3.624152 | 4.793249 | 4.792753 |
| 18000 | 3.749937 | 3.721681 | 3.758973 | 3.757680 | 5.550030 | 5.549657 |
| 18500 | 4.088495 | 4.064156 | 4.061431 | 4.106472 | 5.433315 | 5.433078 |
| 19000 | 3.963143 | 3.939987 | 3.916578 | 3.975790 | 5.140963 | 5.140887 |

## Statistical scope

For each saved start, `beta_j = sum_x min(p_j(x), q_j(x))`, `r_j = min(1, q_j(x_j)/p_j(x_j))`, and `T = 1 + sum_j product_{i<j}(r_i) beta_j`. Mean beta is a calculated overlap statistic (D). Harmonic tokens/round and base-weighted paired deltas retain G, as in the existing estimator. Arithmetic mean T and per-step reach estimates are also retained in `results.json`.

SE uses streams as the units, not adjacent token starts. The harmonic SE uses the delta method on stream means of `1/T`; paired-delta SE is the SE of stream paired deltas. Each prompt depth has one recorded stream. There is no independent within-depth stream SE; `per_start_bootstrap_se_D` is descriptive and ignores serial dependence. Stream SE also ignores dependence between related prefixes and cross-stream carry. No population confidence or speed claim follows from these SEs.

## Pre-registered validation

The design registers a Q3_K direction check and probability controls, not an efficacy threshold for every arm. Its predicted SE range is a guess, not a pass criterion.

Existing CPU evidence, read without rerunning models: `codex/cpu-validation/rows.json` contains M 32 streams, no errors, maximum p error 4.999313463205368e-10 and q error 4.999695102370083e-10, and zero support changes, against the 1e-3 tolerance. `codex/cpu-validation/controls/negative-controls.json` records D misaligned-hidden acceptance loss 69.9767% and permuted-stream loss 95.7804%, both above the registered 50% minimum. The build-window `spec-tf-a5/deterministic-rerun.log` records the identical-arm and seeded-server deterministic checks passing. Its `slice-negative-control.log` records the wrong-vocab control discrepancy M 0.9999996621190892, exceeding 1e-3.

The CPU aggregate measured 3.1390065460146324 tokens/round (M) versus harmonic prediction 2.856469133962012 (G), a prediction-minus-measurement gap of -0.2825374120526204 (D). Served GPU probability-row equivalence and GPU start-selection bias were not measured by this dump run. The optional requantized-4B dose-response validation is absent. These limits keep tokens/round and paired deltas at G; a recovery extraction rerun is unnecessary, while stronger served-law claims need separate validation.

Target-stream support check: FAIL for exact stream-law equivalence. M: 3 of 55,260 checked next-token rows have true tokens outside the extracted final target support, all at `depth-4000`: position 5061 -> token 1876 (raw rank 6), position 5083 -> token 11 (raw rank 3), and position 5488 -> token 1124 (raw rank 2). These positions refer to p rows; the true token is at position + 1. Their raw logit ratios to the top candidate are below the configured min_p=0.05 (D). The original server's p rows are absent, so whether the cause is numerical conditioning or another serving/extraction difference is unverified. `codex/spec_tf.py:17` computes r=0 when the saved p gives the true token zero mass. No rows were discarded because of this mismatch, and the SE does not include its systematic effect. Another identical extraction would not establish equivalence to the original server; that requires a paired served-probability trace, beyond CPU-only dump recovery.

## Reproduction and limits

Full analysis command:

`CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 ~/repos/qwen38-eval-worktrees/bin/cpugate taskset -c 6-11,18-23 python3 codex/reanalyse_spec_tf.py --input /mnt/scratch/work/spec-tf/full-a1 --output /mnt/scratch/work/spec-tf/reanalysis-a1/analysis`

`cpugate` supplies `nice -n 19`. The analysis uses one NumPy/BLAS thread, no models, builds, CUDA programs, or network. Existing item files, `frozen/`, and other trees were not edited. This local task explicitly authorizes correcting the analysis files in the parked item's named worktree; pins must be regenerated before any future queue execution. No item was submitted, retried, cancelled, or reprioritized.

M: full reanalysis completed in 28.046661 seconds after the gate opened. Peak self RSS was 370,584 KiB; maximum child RSS was 91,644 KiB. Their conservative sum is 462,228 KiB (D), below the 8 GiB limit. This is analysis elapsed time, not model or GPU throughput.

M: 15 explicit tests pass in 1.530 seconds, recorded in `codex/reanalysis-tests.log`. Command: `CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 ~/repos/qwen38-eval-worktrees/bin/cpugate taskset -c 6-11,18-23 env PYTHONPATH="$PWD/codex" python3 -m unittest test_spec_tf test_estimator_io test_import_streams test_gpu_payload`. New `test_estimator_io.py` cases cover a mixed low-coverage control, exact exclusion accounting, a covered-row mismatch that returns 1, zero comparable rows that return 1, and ambiguous boundaries. The estimator's broken plain-product negative control differs from the Monte Carlo result by 92.193 SE (D). Recorded parser tests use the real depth and Xid fixtures; the full analysis additionally exercises parsers on the task-named GPU dumps. `git diff --check` passes.

The four background SP reports named in the brief are absent at their supplied paths and were not found by filename search under the supplied scratch root. The task's design, recorded extraction, and build-window validation evidence are available and were used. No host-time or GPU-speed inference depends on the missing background reports.
