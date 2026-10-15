# Maximum-Hip Cascade Benchmark

`configs/morphology_plans/max_hip_cascade_720.json` contains 720 morphology
points. It retains the earlier 432-point proposal and adds 48 surface points
on the `+100%` hip-width cap, 96 surface points descending over the outer
faces, and 144 matching two-thirds radial points.

The plan is a proposed experiment. Its points have not yet been evaluated.

## Local GPU run

From the repository root, activate the project environment and confirm that
JAX sees the intended GPU:

```bash
python -c 'import jax; print(jax.devices())'
```

Run the established 20-second, four-seed-equivalent benchmark settings at
`0.85 m/s` with 512 samples and 5/2 solver iterations:

```bash
python -u -m scripts.multi_morphology_benchmark \
  --config configs/codesign_fixed_mass_moderate_15gen.toml \
  --morphology-plan configs/morphology_plans/max_hip_cascade_720.json \
  --output results/max_hip_cascade_720_vx085_n512_solver5_ls2_4seed \
  --speeds 0.85 \
  --episodes 4 \
  --seconds 20 \
  --seed 1045 \
  --samples 512 \
  --solver-iterations 5 \
  --ls-iterations 2 \
  2>&1 | tee max_hip_cascade_720_vx085_n512_solver5_ls2_4seed.log
```

The launcher resumes by skipping conditions whose `evaluation.json` already
exists. Use a scheduler job on an HPC system and keep the same command inside
the GPU allocation. The complete run contains 2,880 episodes and is expected
to take roughly 60--84 hours on the reference RTX 4090.

Check the plan and command without allocating a GPU:

```bash
python -m scripts.multi_morphology_benchmark \
  --config configs/codesign_fixed_mass_moderate_15gen.toml \
  --morphology-plan configs/morphology_plans/max_hip_cascade_720.json \
  --output /tmp/max_hip_cascade_dry_run \
  --speeds 0.85 --episodes 4 --seconds 20 \
  --samples 512 --solver-iterations 5 --ls-iterations 2 --dry-run
```
