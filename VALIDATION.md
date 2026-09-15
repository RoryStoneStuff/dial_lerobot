# Validation of the LeRobot MPC port

Completed September 8, 2026. These results validate the implementation and
model transfer. They do not establish stable walking or an improved morphology.

## Model identity and code checks

`scripts/verify_ppo_model.py` passed against the zoo's compiled PPO articulation.
It compares body masses and inertial tensors, COMs/orientations, joint axes and
ranges, armature/friction/damping, actuator gains and torque limits, collision
geometry/contact settings, neutral world poses, and actuator forces under the
same bounded command. The additional floor is handled separately.

The model has 19 qpos, 18 qvel, 12 actuators, total mass 14.3817786 kg, and
neutral base height 0.750000015 m. Its source MJCF SHA-256 is
`2f80b27c41f8770177b945125f5b0e8d6e279346bc43cedb4f922904946ffea6`.
The offscreen image in `results/nominal_model.png` was rendered and visually
inspected: it shows the same red lower-body mechanism and fixed torso geometry.

Ten automated tests passed as `rory` (`python -m pytest -q -W error::UserWarning`,
34.97 s), covering morphology changes, bounded alternating gait references,
velocity frames, fall/terminal costs, config validation, seeded DIAL evaluation,
saved logs, replay loading, cache selection, and migration of an initialized
compilation cache. There were no warnings.
Static checks passed with `python -m ruff check dial_lerobot tests`.
GPU repeatability is checked with floating-point tolerance, not bitwise equality.

## Cache permission regression

Earlier validation ran as root and left HydraX's shared `/tmp/jax_cache`
root-owned. Launching the live viewer as `rory` then failed in DIAL warmup while
XLA tried to write per-fusion autotuning files. The application now selects
`/home/rory/.cache/lerobot_mpc-1000/jax`, checks the nested cache directories,
and resets JAX's initialized file-cache handle before switching directories.
This override runs after HydraX imports and before controller construction.

The full baseline was rerun as `rory` (UID 1000), with cache errors made fatal:

```bash
JAX_RAISE_PERSISTENT_CACHE_ERRORS=true ./run_mpc.sh test --config configs/baseline.toml --steps 2 --episodes 1 --output results/cache_fix_validation
```

The same 1,024-sample warmup used by the viewer passed in 36.13 s. Two subsequent
MPC updates passed in 3.99 s with finite outputs and no permission warnings or
errors. Saved metadata points to the new cache; compiled files and per-fusion
autotuning records were confirmed to be owned by `rory`. This checked the
reported warmup failure through the headless entry point; desktop interaction
was not exercised.

## Full DIAL settings

Command:

```bash
bash run_mpc.sh test --config configs/baseline.toml --steps 25 --episodes 1 --output results/baseline_validation
bash run_mpc.sh visualize results/baseline_validation/episode_000.npz
```

This used 1,024 samples, 4 knots, 1 DIAL iteration, 0.5 s planning horizon,
0.005 s physics, and the original cost weights with the documented robot
adaptations. Runtime: MuJoCo/MJX 3.10.0, JAX/JAXlib 0.10.2, HydraX 0.0.2, CUDA.

| Measurement | Result |
|---|---:|
| JAX warmup | 36.00 s |
| Simulated duration | 0.125 s |
| Evaluation wall time | 52.90 s |
| Median MPC update time | 2.124 s |
| Final base height | 0.69349 m |
| Forward displacement | 0.002982 m |
| Mean body-frame forward velocity | 0.02323 m/s |
| Healthy fraction during this short test | 1.0 |
| Sum of running costs | 50.09026 |
| Co-design fitness | -50.08728 |

All state, control and cost outputs were finite. Full state and cost logs, the
generated MJCF, metadata and diagnostic PNG are saved in that folder. The
0.125-second duration is much too short for a gait claim. Most running cost in
this brief test came from joint-reference tracking. The current JAX backend at
these settings is far slower than real time; it also makes the default outer
search expensive. No long locomotion benchmark was run.

## Complete outer-loop check

The initial `results/cma_validation` attempt was interrupted by the disconnect
before it completed a candidate. The completed run is separate:

```bash
bash run_mpc.sh optimize --config configs/smoke.toml --output results/cma_validation_completed
```

One CMA generation evaluated two distinct morphologies using actual DIAL,
with 8 samples, a 0.02-second horizon, and 3 updates in one episode per candidate.
Candidate fitness values were -6.28677994 and -6.33389588. The chosen design was
approximately `[0.3648888844, 0.2538546415, 2.0323855064]`. It is only the winner
of this tiny plumbing test, not a recommended or optimized robot design.

`optimization.json`, `current_generation.json`, and `best_design.json` contain
the completed generation, per-candidate scores/episode metrics, effective
configuration and source-model metadata. The test exercises model changes,
controller construction, simulation, scoring, CMA updates and saved output.
The saved design also reloaded successfully through `inspect --design`, and
the `visualize` command generated `optimization.png` from the completed log.

## Remaining performance work

The native GUI code and saved replay use the validated scene/controller, and
offscreen rendering plus replay loading passed. Desktop interaction itself
was not exercised during this session. Stable locomotion, long episodes, and
a meaningful morphology search remain experiments to run after tuning. A full
80-generation search was not started.
