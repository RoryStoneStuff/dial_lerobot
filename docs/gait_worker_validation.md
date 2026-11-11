# Gait inner-loop reliability

## Diagnosis (2026-11-03)

The recorded failures were not all crashes. Several runs previously reported
as stopped have complete results, including `gait_subprocess_smoke_20261103`
and `bayesian_gait_inner_short_20261102_worker`. Host processes are not visible
in the agent's sandbox process namespace. A missing sandbox PID or an
unfinished checkpoint is insufficient evidence that a host job has exited.

There were also genuine native faults. The kernel journal records an
`llvm-worker-4` SIGSEGV in `libjax_common.so`. The saved core from the earlier
PID 3148 failure has an unresolved stack ending in an LLVM callback inside
`mlir::sdy::walkShardings`, also in `libjax_common.so`. This locates that fault
in the JAX compiler; it does not establish the underlying upstream defect.
It is not evidence that a gait fall killed CUDA initialization.

The 2026-11-03 cadence-only outer run later recorded a gait worker SIGABRT
(`terminate called without an active exception`) during JAX
`backend_compile_and_load` warmup. The isolated outer process continued to the
next candidate. Its output then stops during the following candidate, and no
process remained after the reported forced reboot. The prior boot journal has
no entries after 03:17 MST, while the worker abort is timestamped 03:36 MST.
There is no kernel evidence for OOM, NVIDIA Xid, or a thermal event in that
missing interval, so the compiler abort is confirmed but the host freeze cause
is not. A worker SIGABRT alone should not reboot or freeze the OS.

A controlled standalone CUDA diagnostic with the existing implementation,
512 samples, solver 5/2, and two 10-second gait evaluations completed
successfully. Its results are under
`/home/rory/mpc_results/gait_cuda_diagnostic_20261103`.

## Changes

- Duty, cadence, and gait start delay are dynamic inputs to `MorphologyKernel`.
  Each inner search retains one kernel across its gait candidates. Other cost,
  robot, and MPC settings must still match for reuse.
- The outer gait-enabled Bayesian search keeps one isolated Python worker alive
  across morphology candidates. The worker retains its JAX context and shared
  morphology kernel, reducing repeated CUDA startup/teardown while keeping
  native faults contained outside the outer optimizer. A worker crash is
  recorded, and the next request starts a fresh worker.
- Standalone gait searches still use a one-shot isolated interpreter. Worker
  logs and `worker_status.json` preserve errors, return codes/signals, timeouts,
  completion state, and the persistent worker PID.
- The outer Bayesian optimizer catches the dedicated worker failure type and
  records an unsuccessful morphology instead of aborting the whole search.
- The outer optimizer now atomically checkpoints after every candidate,
  including the Bayesian RNG state and partial generation. `--resume` continues
  from that checkpoint; older runs without checkpoints are reconstructed from
  `optimization.json` and `current_generation.json` where possible.
- Nested runs no longer query JAX devices in the parent process. The worker
  records the actual device, avoiding an unnecessary parent CUDA context while
  the isolated worker compiles and evaluates.
- The selected gait's evaluation is reused for outer scoring, with its actual
  trajectory path and gait configuration saved in the winning design. This
  avoids a redundant unisolated simulation in the parent. It is a search score,
  not an independent validation on new seeds.
- Automatic motor resizing is rejected in nested mode rather than silently
  changing morphology after gait tuning.

## Verification

`tests/test_gait_bayesian.py` exercises a real child SIGSEGV, a real child
timeout, continuation after worker failures, all-failed completion, and saved
gait configuration. `tests/test_shared_morphology.py` compares shared and fresh
evaluators, verifies that changed gait targets take effect, and checks that
gait changes do not compile another kernel.

For a full nested GPU check (one baseline and two proposed morphologies, each
with the default gait plus six additional gait candidates):

```bash
bash run_mpc.sh optimize --optimizer bayesian --gait-inner-loop \
  --gait-candidates 6 \
  --config configs/codesign_draft_mass_calibrated_15gen.toml \
  --generations 1 --popsize 2 --episodes 1 --steps 1000 \
  --samples 512 --solver-iterations 5 --ls-iterations 2 \
  --output results/gait_nested_check
```

Use an empty, unique output directory. The inner loop uses two random proposals
then GP-guided proposals. To continue an interrupted run, pass the same
settings and output folder with `--resume`:

```bash
bash run_mpc.sh optimize --optimizer bayesian --gait-inner-loop \
  --gait-candidates 4 --gait-parameters cadence \
  --config configs/codesign_draft_mass_calibrated_15gen.toml \
  --generations 21 --popsize 6 --episodes 1 --steps 1000 \
  --samples 512 --solver-iterations 5 --ls-iterations 2 \
  --output /home/rory/mpc_results/gait_cadence_only_6outer_20261103 \
  --resume
```

For SSH disconnections, launch under `nohup` or `tmux`; a detached process
survives the SSH session but not a host reboot. After a reboot, rerun the
command with `--resume`. Automatic post-reboot restart needs a persistent
systemd service. Check worker completion and the outer `optimization.json`
`stop` record before reporting completion. Longer locomotion studies should
use multiple episodes/seeds and independently validate the selected design.

## 2026-11-07 kernel incident

The resumed cadence-only search had completed generations 0–9 and checkpointed
one evaluation in generation 10 before the host stopped. The prior boot journal
shows three kernel oopses: two in `psi_avgs_work` on CPUs 8 and 9, followed by
one while the Python process was exiting. The bad instruction-fetch addresses
were invalid/NX-protected. No OOM-killer, NVIDIA Xid, or machine-check record was
found in the captured journal. This is kernel-level memory corruption, not an
exception the Python optimizer can catch. The log alone does not identify
whether its trigger was the proprietary NVIDIA module, the kernel, or hardware.

The inner search differs materially from outer Bayesian candidate selection:
each morphology adds a baseline plus the requested gait evaluations, and the
older implementation started and stopped a JAX/CUDA process for every
morphology. The persistent isolated worker removes most of that context churn,
but is a mitigation rather than proof of a kernel fix. Before another long GPU
run, investigate kernel/driver updates and system memory stability; use a short
monitored smoke test after any system change. Keep the checkpointed run intact
and resume only after the host-level instability is addressed.
