Linux CPU build (-DGGML_CUDA=OFF, OpenMP), wt/stack10b 2d3511bf7, 10-09 17:3x.
fixed rc=0 (both subgraph_capacity cases PASSED); mutant rc=139 SEGV in-process; old forking test (997c8e417^^) rc=124: hung at test_meta_subgraph_capacity(compute) until timeout 90 s.
