# HydraX DIAL LeRobot lower-body locomotion

For a fresh clone on another machine or an HPC server, follow
[the HPC setup and Git workflow](docs/hpc.md).

For morphology optimization around the recent walking controller, see
[the LeRobot co-design workflow](docs/codesign.md) and
`configs/codesign_walking.toml`.

This is the humanoid workflow from `../dial-mpc`, ported to
the same zoo lower body used by the PPO gait experiment. DIAL plans lower-body control trajectories online. The baseline uses torque control
with the official Unitree H1 lower-body limits. CMA-ES optionally searches robot morphology around
that controller. Running MPC does not require a PPO checkpoint or PPO training.

The model builder reads the existing file
`../lerobot-legged-zoo/models/bipedal_plateform_no_arms/mjcf/robot.xml` and its
meshes. Nominal geometry, masses, inertias and joint limits come from LeRobot.
The torque-mode controller uses H1-style PD gains and joint-family torque limits;
it is not identical to the PPO controller. No source repository
is modified by these commands. Changes to a design exist in the generated scene.

Current stance work and the exact cost changes are documented in
[the stance-contact report](docs/stance_contact_audit.md). Use
`configs/stance_contact.toml` for this investigation. Walking is not established;
the historical `baseline.toml` uses a 20 ms physics step that was unstable in the
zero-target sanity check with this robot/controller.

The controlled comparison with the original H1 walking reward weights uses
`configs/paper_weights.toml`; see [weights and results](docs/paper_weight_trials.md).
This preset disables our additional posture/contact/terminal objectives while
retaining LeRobot geometry and fall protection. It is an experiment, not a
validated walking controller. Both current presets use normalized torque effort.

## Quadratic interpolation options

Set `spline_type` in the `[mpc]` section of your TOML config:

- `quadratic` retains the existing local three-point Lagrange interpolator,
  clipped to normalized action bounds `[-1, 1]`.
- `quadratic_dial` uses `jax_cosmo.scipy.interpolate.InterpolatedUnivariateSpline`
  with `k=2`, the same library and spline order used by DIAL-MPC's `node2u`.
  It preserves that library's overshoot and extrapolation behavior rather than
  clipping the interpolated values. Both options require at least three knots.

```toml
[mpc]
spline_type = "quadratic_dial"
```

The new option applies to rollout controls, action evaluation, and warm-start
resampling. Warm-start query times still use this controller's existing endpoint
clamping; selecting the spline does not change the rest of the planning algorithm.
Existing configs and defaults are unchanged. The MPC environment needs
`jax-cosmo==0.1.0` and `setuptools>=69,<81` (for jax-cosmo's `pkg_resources`
import), now declared in `pyproject.toml`.

## Start here

On this machine the launcher uses the working DIAL environment at
`/home/rory/co_design/.venv/bin/python`. It adds this package to `PYTHONPATH`;
it does not install into or change that environment or the PPO environment.

JAX compilation and GPU autotuning use a cache owned by the launching account:
`~/.cache/dial_lerobot-1000/jax` for `rory` (UID 1000). The package overrides
HydraX's shared `/tmp/jax_cache` default after importing HydraX. This prevents
root-run validation from leaving a cache that your normal GUI launch cannot
write. Set `DIAL_LEROBOT_CACHE_DIR` to choose another user-owned cache directory.
The first launch with a fresh cache recompiles the controller.

```bash
cd /home/rory/coDesign/dial_lerobot

# Model identity, mass, motor gains, target limits and nominal design vector.
bash run_mpc.sh inspect

# Very small plumbing test; deliberately insufficient for gait assessment.
bash run_mpc.sh test --config configs/smoke.toml

# Stance investigation: 5 ms physics, 10 ms replanning, up to 10 simulated seconds.
bash run_mpc.sh test --config configs/stance_contact.toml --steps 1000 --episodes 1 --output results/walk_01

# Plot velocity, foot clearance, path, individual costs and motor targets.
bash run_mpc.sh visualize results/walk_01/episode_000.npz

# Watch the saved motion without paying for MPC computation during playback.
bash run_mpc.sh replay results/walk_01/episode_000.npz --loop

# Live MPC in the native MuJoCo GUI (run from the remote desktop's terminal).
bash run_mpc.sh render --config configs/stance_contact.toml --seconds 10

```

Use a new output folder for each run. Omitting `--output` creates a timestamped
folder. Initial JAX compilation can take tens of seconds or longer. The live
viewer opens after compilation. A requested playback rate is a maximum;
live MPC can run slower than real time. Saved replay avoids this computation.
The current 128-sample stance comparisons take about 0.36 wall seconds per
10 ms MPC update on the RTX 4090, plus roughly 32 seconds of compilation/warmup.
These timings apply to the configuration in the stance report, not every preset.

Viewer keys: **Space** pauses/resumes; **R** resets/restarts; close the window
to exit. Mouse controls are MuJoCo's normal camera controls. Velocity is set
through `--vx`, `--vy` or the TOML file; there are no WASD velocity bindings.
The live viewer pauses at `--seconds` or on a detected fall. `--seconds 0` removes
the duration limit; the configured fall criterion still applies.

## Change the knobs

All commands use the same configuration code. Edit a copy of
`configs/stance_contact.toml`, or override common values:

```bash
bash run_mpc.sh test --config configs/stance_contact.toml --vx 0.4 --samples 512 --horizon 0.5 --knots 8 --iterations 2 --noise 0.2 --temperature 0.3 --steps 1000 --episodes 1

# Same 200 Hz physics, replan at 50 Hz: 500 updates = 10 simulated seconds.
bash run_mpc.sh test --config configs/stance_contact.toml --decimation 4 --steps 500 --episodes 1

bash run_mpc.sh test --help
```

Historical `configs/baseline.toml` values (not the current stance preset):

| Setting | Baseline | Meaning |
|---|---:|---|
| Physics timestep | 0.02 s | DIAL-style 50 Hz control/physics baseline |
| Control decimation | 1 | Source evaluator replans each physics step |
| Planning horizon | 0.4 s | 20 simulated steps per candidate |
| Samples / knots / iterations | 2048 / 5 / 1 | DIAL H1-style baseline |
| Noise / temperature | 0.3 / 0.05 | DIAL-style trajectory sampling temperature |
| Annealing beta, iteration / horizon | 1.0 / 0.9 | DIAL trajectory annealing parameters |
| Spline | zero | Zero-order hold; linear is also supported |
| Episode updates / episodes | 300 / 2 | 6 s per episode at baseline decimation |
| Initial height jitter | ±0.02 m | Independently seeded episodes |
| Target forward / lateral speed | 0.5 / 0 m/s | Body-frame running-cost target |
| Target ramp | 2 s | Linear ramp from zero |
| Gait cadence / duty / clearance | 1 Hz / 0.50 / 0.05 m | Left/right phases separated by half a cycle |

Always compute duration from the effective configuration: `steps * timestep * decimation`
gives the episode's simulation duration. By default, a fall terminates the episode immediately. The saved trajectory is
still written, together with `fell`, `failure_time_s`, and the prefix cost totals.
Set `terminate_on_fall = false` only when reproducing a fixed-length diagnostic.

## Cost function

The running objective contains foot-height tracking, hip/knee reference tracking,
body-frame velocity, upright orientation, yaw/rate, base height, lateral-joint
posture, stance/swing contact, normalized-action magnitude, optional stance slip,
and a fall penalty. The terminal objective is
`5 * height_error² + 2 * (body_vx - target_vx)² + 10 * up_vector_error²`
with an un-ramped velocity target. These coefficients are configurable.

See [the cost-change table](docs/stance_contact_audit.md#exact-cost-changes) for
formulas and normalization. The contact objective uses all six collision
capsules per foot; it does not measure ground reaction force. The gait-height
objective uses the lowest capsule surface. A new stance-slip term penalizes
horizontal sole-site velocity while a scheduled stance foot is near the floor.

Joint and actuator indices are resolved by name because the model's tree order
and actuator order differ. Normalized actions in [-1,1] map piecewise to joint
targets, with action zero at the CAD zero pose and positive/negative extremes
at soft joint limits. Each physics step computes clipped PD torques. The hip
references have opposite left/right signs, and both knee references flex toward
negative angles. Default hip/knee amplitudes are 0.25/0.35 rad. Legacy
`qstand_hip`/`qstand_knee` fields currently do not alter this zero-centered reference.

Grounded neutral height is derived from the collision geometry, nominally
about 0.720 m. The default healthy-height threshold is that height times
0.60/0.85; upright dot must also exceed 0.8. Initial clearance is separate from
this target. A configurable startup delay holds a double-stance reference;
duty determines the subsequent per-leg stance fraction.

HydraX integrates the planning running cost with `dt` and adds terminal cost.
The outer co-design fitness copies the source: forward displacement minus the
**unscaled sum** of running costs, averaged across episodes. Higher fitness is
better; negative values are normal. Costs are accumulated at every physics
step even when control decimation is greater than one. Compare designs using
the same episode duration, timestep and scoring settings.

## Morphology co-design

```bash
# Validate the outer search machinery cheaply (not an optimized walking design).
bash run_mpc.sh optimize --config configs/smoke.toml --output results/search_smoke

# Original CMA settings: 80 generations, 12 candidates per generation.
bash run_mpc.sh optimize --config configs/baseline.toml --output results/search_01

bash run_mpc.sh visualize results/search_01/optimization.json
bash run_mpc.sh inspect --design results/search_01/best_design.json
bash run_mpc.sh render --design results/search_01/best_design.json

# Evaluate an explicit design without changing the zoo files.
bash run_mpc.sh test --theta 0.34 0.26 2.0 --steps 2000 --episodes 1

# Optional design sweep at fixed density proxy. Each point runs full DIAL evaluation.
bash run_mpc.sh heatmap --points 5 --density 2.0 --output results/sweep_01
bash run_mpc.sh visualize results/sweep_01/heatmap.json
```

The nominal design is `[0.3413655864, 0.2607679717, 2.0]`: thigh and shin
anchor-to-anchor distances in metres, then a mass-density proxy. The tiny CAD
lateral offsets are included, so these differ slightly from the usual
0.34/0.26 m segment labels. Both sides use the same length scale factors and
retain the CAD's mirrored geometry. The Berkeley initial vector `[0.15, 0.16, 2]`
is not used because it would substantially shrink the trained robot.

The source model template contains no substitutions for its design variables;
this port implements actual changes. A segment's direct geometry and visual
meshes scale uniformly about its joint; its child joint anchor and COM move
with that scale. Its mass becomes `CAD_mass * length_scale * density/2`,
preserving the source's linear mass-versus-length assumption while reproducing
the zoo mass at the nominal design. Its inertia scales by that mass factor
times `length_scale²`. The torso and feet retain their own geometry and mass.

This is a simulation design family: uniform link scaling also changes link
width and decorative motor meshes. The density proxy is not a physical
material density in kg/m³, and mass changes include the complete thigh/shin
assemblies. Motor gains, armature and torque limits remain fixed. Manufacturing
constraints and motor selection are outside the original search and this port.

CMA keeps the source's sigma 0.05, bounds `[0.05,0.05,0.1]` to `[0.4,0.4,8]`,
seed 42, and generation/candidate evaluation seed schedule. Use
`--generations` and `--popsize` to reduce the search. Geometry changes compile
separate MJX controllers; executables are cleared between candidates to bound
GPU memory. A full search is expensive. Progress and best design are written
after candidates/generations; Ctrl-C stops the job and preserves completed logs.
Exact interrupted CMA-state resume is not implemented. A saved best design
can seed a fresh search using `--design`.
The full default search contains 960 candidates and 576,000 MPC updates,
so it can take days at the measured baseline speed. Start with the small
validation settings and tune the controller before spending that compute.

## Files and results

| File | Purpose |
|---|---|
| `dial_lerobot/assets/model_builder.py` | Zoo scene, PPO physics, morphology |
| `dial_lerobot/tasks/lerobot_task.py` | Adapted humanoid cost and gait references |
| `dial_lerobot/evaluator.py` | Shared DIAL controller, simulation and metrics |
| `dial_lerobot/optimize.py` | CMA-ES and design sweep |
| `dial_lerobot/visualize_lerobot.py` | Live GUI, replay, snapshot and plots |
| `dial_lerobot/run_experiment.py` | Command interface |
| `configs/baseline.toml` | Source defaults with robot adaptations |
| `configs/smoke.toml` | Small integration settings |
| `configs/stance_contact.toml` | Current stance investigation settings |
| `dial_lerobot/assets/foot_geometry.py` | Geometry-derived flat-floor capsule distances |
| `scripts/stance_trials.py` | Reproducible direct-seed stance comparisons |
| `scripts/analyze_stance.py` | Saved-state clearance and slip diagnostics |
| `scripts/verify_ppo_model.py` | Independent comparison with compiled PPO model |

Each evaluation writes the generated `model.xml`, `metadata.json`,
`evaluation.json`, and per-episode `.npz`/`.csv` files. Metadata records the
source MJCF path/hash, effective configuration, robot properties, runtime
versions and device. Saved XML references the original mesh files by absolute
path. NPZ includes full qpos/qvel, normalized actions, PD torque commands, actuator forces, foot
positions, capsule heights, sole-site velocities, scheduled stance masks, body velocity, target velocity, cost components and timing. CSV
contains the main motion and cost metrics. The `visualize` command produces PNGs.

## Environment and checks

The existing environment tested here has Python 3.12, MuJoCo/MJX 3.10.0,
JAX/JAXlib 0.10.2, HydraX 0.0.2, and CMA 4.4.4 on CUDA. Its editable HydraX
source is `/home/rory/co_design/hydrax`. `pyproject.toml` also pins a compatible
HydraX revision for a separate installation. To use a different existing
environment, set `LEROBOT_MPC_PYTHON=/absolute/path/to/python`. A local `.venv`
takes precedence over the default shared environment. On a fresh machine,
install this package and its dependencies into that local environment.
Set `LEROBOT_ZOO_ROOT` if the zoo is elsewhere.

```bash
cd /home/rory/coDesign/dial_lerobot
/home/rory/co_design/.venv/bin/python -m pytest -q
/home/rory/coDesign/lerobot-legged-zoo/.venv/bin/python scripts/verify_ppo_model.py

# Offscreen PNG: does not open the desktop GUI.
MUJOCO_GL=egl bash run_mpc.sh snapshot --output results/nominal_model.png
```

The automated checks cover physical model parity, morphology effects, cost
frames and penalties, alternating reachable gait references, seeded DIAL
evaluation, command bounds, logging and replay loading. These are software
checks, not evidence that the copied cost produces a stable walking gait.
See `VALIDATION.md` for the runs performed during this port.
