# Offline Swift draft-vocabulary coverage

M: 187 input files across 27 campaigns; 0 matching model-calls.jsonl files. Llama inputs use response_item records from Codex rollouts, including failed/retried attempts. Only assistant reasoning, prose, and tool arguments are counted; input history, event_msg copies, tool results, and tool names are excluded.

M: tokenizer SHA256 `d73c2c5f7aa0ed522c8d96ef3524739eb61e3c78e74839a2ce4a1c56ea340a20`; subset SHA256 `e30af79520ade5b7ddb851d75cf271f07d407dc768bf646e4517607d562a65a6`; 25879 unique IDs.

M: all 248077 tokenizer IDs match IQ4_XS GGUF token spellings; the other 243 GGUF entries are [PAD...] placeholders absent from tokenizer.json.

| Region | Tokens (M) | Outside subset (M) | Outside fraction (D) | Empirical acceptance ceiling (D) |
|---|---:|---:|---:|---:|
| prose | 109929 | 2529 | 2.300576% | 97.699424% |
| reasoning | 1212685 | 26244 | 2.164123% | 97.835877% |
| tool-call args | 530853 | 35064 | 6.605218% | 93.394782% |
| all | 1853467 | 63837 | 3.444194% | 96.555806% |

At any history, subset acceptance is at most p(S), since sum min(p,q) <= p(S). The outside-output fraction estimates the target mass forced into residual verification, so 1-f is an empirical ceiling, not a measured draft acceptance. Relative to a perfect full-vocabulary drafter, support alone forces loss f. Relative to the actual full drafter, incremental loss is unknown: renormalization also changes q within S. It is incorrect to claim f is an upper bound on total acceptance loss.

Each recorded region is tokenized independently with tokenizer.json (NFC + byte BPE), without special-token insertion. These are reconstructed content tokens, not a capture of wire token IDs; message boundaries, template delimiters, parser stripping, and normalization can change counts. The sample is agent trajectories rather than independent draws. Full provenance is in coverage.json.

| Outside token ID | Token spelling | Decoded piece | Count (M) |
|---:|---|---|---:|
| 27238 | `"\u0120{\\"` | `" {\\"` | 7628 |
| 49575 | `"\u0120}\\"` | `" }\\"` | 5377 |
| 17347 | `";\\"` | `";\\"` | 4084 |
| 25427 | `",\\"` | `",\\"` | 3101 |
| 48147 | `"orc"` | `"orc"` | 2655 |
| 13485 | `"unge"` | `"unge"` | 1900 |
| 68568 | `".imag"` | `".imag"` | 1342 |
| 12160 | `"_callback"` | `"_callback"` | 1240 |
| 28817 | `".real"` | `".real"` | 1219 |
| 12911 | `"\u0120jest"` | `" jest"` | 1077 |
| 60607 | `"()\\"` | `"()\\"` | 695 |
| 89248 | `"conj"` | `"conj"` | 682 |
| 7515 | `"\u0120cells"` | `" cells"` | 612 |
| 6329 | `"\u0120imag"` | `" imag"` | 569 |
| 34176 | `".compute"` | `".compute"` | 562 |
| 27110 | `"imag"` | `"imag"` | 534 |
| 40980 | `".inputs"` | `".inputs"` | 458 |
| 7197 | `"\u0120exerc"` | `" exerc"` | 456 |
| 34821 | `"(\\\""` | `"(\\\""` | 443 |
| 30174 | `"-num"` | `"-num"` | 440 |
| 1837 | `"bers"` | `"bers"` | 439 |
| 47373 | `"jest"` | `"jest"` | 426 |
| 49038 | `"Equality"` | `"Equality"` | 398 |
| 36383 | `"\u0120reactor"` | `" reactor"` | 363 |
| 16007 | `"\u0120conj"` | `" conj"` | 307 |
| 19941 | `"attack"` | `"attack"` | 293 |
| 87245 | `"-groups"` | `"-groups"` | 293 |
| 9834 | `"\u0120python"` | `" python"` | 288 |
| 17359 | `"_im"` | `"_im"` | 280 |
| 4210 | `"-group"` | `"-group"` | 250 |

| Campaign | Tokens (M) | Outside (M) | Fraction (D) |
|---|---:|---:|---:|
| campaign-swift15-iq4-xs-j1-longtail-20260930-stack-resync-f16kv-pair1 | 60580 | 1983 | 3.273358% |
| campaign-swift15-iq4-xs-j1-longtail-20260930-stack-resync-fmq-pair1 | 70834 | 2170 | 3.063501% |
| campaign-swift15-iq4-xs-longtail-20260928 | 79060 | 2463 | 3.115355% |
| campaign-swift15-iq4-xs-longtail-20260929-combo-pair1 | 80257 | 2654 | 3.306877% |
| campaign-swift15-iq4-xs-longtail-20260929-combo-pair2 | 70928 | 2312 | 3.259644% |
| campaign-swift15-iq4-xs-longtail-20260929-combo-pair3 | 77544 | 2390 | 3.082121% |
| campaign-swift15-iq4-xs-longtail-20260929-gs2-pair1 | 69658 | 2238 | 3.212840% |
| campaign-swift15-iq4-xs-longtail-20260929-gs2-pair2 | 62754 | 2065 | 3.290627% |
| campaign-swift15-iq4-xs-longtail-20260929-gs2-pair3 | 114815 | 3890 | 3.388059% |
| campaign-swift15-iq4-xs-longtail-20260930-acclog | 69420 | 2433 | 3.504754% |
| campaign-swift15-iq4-xs-longtail-20260930-stack-resync-f16kv-pair1 | 76029 | 2582 | 3.396073% |
| campaign-swift15-iq4-xs-longtail-20260930-stack-resync-fmq-pair1 | 78214 | 2624 | 3.354898% |
| campaign-swift15-iq4-xs-longtail-20260930-stack-resync-pair1 | 92497 | 3425 | 3.702823% |
| campaign-swift15-iq4-xs-longtail-mtpfast-bdb199-rep1-20260929-1213 | 59065 | 1959 | 3.316685% |
| campaign-swift15-iq4-xs-longtail-mtpfast-bdb199-rep2-20260929-1213 | 71685 | 2472 | 3.448420% |
| campaign-swift15-iq4-xs-longtail-mtpfast-bdb199-rep3-20260929-1213 | 59937 | 2071 | 3.455295% |
| campaign-swift15-iq4-xs-longtail-upstream-ctl-rep1-20260929-1213 | 69063 | 2622 | 3.796534% |
| campaign-swift15-iq4-xs-longtail-upstream-ctl-rep2-20260929-1213 | 75101 | 3021 | 4.022583% |
| campaign-swift15-iq4-xs-longtail-upstream-ctl-rep3-20260929-1213 | 69162 | 2039 | 2.948151% |
| campaign-swift15-iq4-xs-lost4-rep1-20260929 | 28521 | 910 | 3.190631% |
| campaign-swift15-iq4-xs-lost4-rep2-20260929 | 53329 | 2273 | 4.262221% |
| campaign-swift15-iq4-xs-lost4-rep3-20260929 | 35116 | 1009 | 2.873334% |
| campaign-swift15-q5-k-s-longtail-20260930-stack-resync-fmq-pair1 | 91371 | 3021 | 3.306301% |
| campaign-swift15-q6-k-longtail-20260928 | 70972 | 2485 | 3.501381% |
| campaign-swift15-q6-k-lost4-rep1-20260929 | 28999 | 765 | 2.638022% |
| campaign-swift15-q6-k-lost4-rep2-20260929 | 84061 | 3804 | 4.525285% |
| campaign-swift15-q6-k-lost4-rep3-20260929 | 54495 | 2157 | 3.958161% |

M: CPU elapsed 28.163 s; no inference or GPU access.
