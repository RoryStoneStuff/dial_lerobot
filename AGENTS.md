# Instructions for agents working on this repository

Read `README_HPC_AGENT.md` first when setting up or running this project on a
new machine. Read `docs/hpc.md` for installation and `docs/codesign.md` for
the morphology search. This is LeRobot locomotion using HydraX DIAL MPC and
CMA-ES; it does not require an RL training environment.

## Preserve the experiment

- The reference configuration is `configs/codesign_fixed_mass_moderate_15gen.toml`.
- Preserve controller, costs, physics, seeds, and morphology bounds unless the
  user's requested experiment changes them. Put experiments in new configs.
- Original body masses are fixed. The free design variables are thigh length,
  shin length, and hip width. The legacy density slot is fixed at 2.0.
- Longer links still change COM locations and rotational inertia. Do not
  accidentally freeze the geometry or reuse nominal cost height references.
- Keep the shared morphology kernel in the outer optimization. Model arrays
  and morphology-dependent references are inputs; do not recompile or clear
  JAX caches after every candidate.
- Rank designs by completion, then survival, then lower running cost per
  simulated second. The legacy distance-minus-cost fitness is diagnostic.
- A short smoke test establishes plumbing, not successful locomotion.
  Compare walking over at least 10 simulated seconds and retain fall stopping.

## Execution and reporting

- Install the project into a new local `.venv`; do not copy another machine's
  environment or rely on `/home/rory` paths.
- Follow the cluster's scheduler and resource policies. GPU simulations run
  in a GPU allocation, not on login nodes. Identify the scheduler/account/
  partition requirements from local documentation; do not invent them.
- Request one GPU and run one optimization process at a time initially.
- Confirm JAX sees a GPU before GPU experiments. Never silently substitute CPU
  execution for a GPU performance measurement.
- Use unique, initially empty output folders. Keep logs outside that folder
  until the launcher creates it; it rejects nonempty output folders.
- Record Git revision, environment versions, GPU model, config, scheduler job
  ID, output path, and elapsed execution/compilation time when reporting runs.
- Distinguish submitted, queued, running, failed, and completed jobs. Check
  scheduler state and saved progress rather than assuming a job completed.
- The baseline has shown variation between runs even at the same seed. Do not
  promise identical trajectories across hardware or claim robust improvement
  from search seeds alone; use additional seeds for validation.

## Git workflow

Use `git status` before changes and preserve unrelated work. Use a separate
branch for HPC changes, and push code/docs/configs when authorized by the
user. Environments, downloaded assets, caches, and result directories are
excluded from Git; transfer experimental artifacts separately. Never put
credentials in source files or commits.
