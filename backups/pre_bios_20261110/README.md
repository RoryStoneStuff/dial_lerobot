# Pre-BIOS project backup — 2026-11-10

The main repository contains the current DRAFT, Bayesian, gait inner loop, controller, and figure-generation source. These supporting snapshots preserve changes in other repositories without pushing to third-party upstreams.

For each repository in manifest.json, clone its origin and check out base_commit, apply working-tree.patch with `git apply`, then copy the contents of untracked/ into that checkout. lerobot_mpc/ is an older standalone source snapshot.

Experiment trajectories, environments, caches, crash dumps, and downloaded dependencies are not in this Git backup. Experiment results must be copied separately from /home/rory/mpc_results. No simulation or stability validation was performed for this backup.
