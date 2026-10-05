# draft-vocab

Stack4 already serves the reduced MTP vocabulary. This branch adds optional GGUF-carried IDs and validates the existing compact-head/rejection path; it does not introduce a new speed optimization over the served subset.

CPU checks passed, including separate stack4 bit identity, real embedded-GGUF decoding and both rejection-law negative controls. A CUDA build is frozen at `5e24a96c5`, never run. Both J5q and IQ4_XS GPU items pass `gpuq submit --check` and remain unsubmitted. The winning k and GPU performance remain unknown here.

Design, proof, labeled CPU numbers, evidence, commands, prepared item paths and risks: [REPORT-draft-vocab.md](REPORT-draft-vocab.md).
