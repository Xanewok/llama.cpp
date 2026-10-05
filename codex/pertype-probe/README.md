# Per-type MMVQ probe

Parent only: queue `codex/pertype-probe/item.toml`. Do not run GPU modes from the coding session. `codex/gpu_ab.sh --preflight` hides GPUs and checks frozen hashes, source identity, library resolution and recorded-artifact parser controls.

Full order: up/patch/patch/up on each RTX 5060 Ti, where up is the frozen general form and patch is the per-type table. No server is launched. Each child PID is recorded and terminated exactly. The one-patch-boot canary uses aligned correctness, partial rows and body/head performance. Full partial tests cover rows 1 through 13; columns 5 through 8 cover all selected warp/row pairs. Performance also covers unchanged columns 1 through 4 and q8_0.

The harness reuses `codex/w2r4-general-probe/probe.py` and its recorded GPU fixtures. Parser tests use actual GPU SQL, stderr, inventory and kernel-log artifacts; head validation changes only the named geometry in a recorded SQL row. Bad correctness, timing, commit, device, shape, missing and duplicate records are rejected. Results are saved before boot gates fail. GPU-vs-CPU numerical agreement does not prove token identity or serving speed.

Budget: 8 boots x (3 stages x 60 s + 60 s memory wait) = D: 1920 s, within D: 70% of the 3600 s run budget. Canary: one boot, same parsers/gates, D: 240 s stage/wait envelope within the 540 s budget. GPU modes have not run here.
