# LeRobot morphology search

Use `configs/codesign_walking.toml`. It freezes the recent forward-walking
controller: 0.85 m/s target, 10 ms physics, 2/2 solver, 256 DIAL samples,
0.4 s horizon, six quadratic DIAL knots with shift reconstruction, noise 0.05,
temperature 0.02, and no joint-reference or forward-placement cost.
Each design gets four 10-second episodes, stopping immediately on failure.
This is a starting evaluation protocol, not a guarantee that modified designs
will remain stable with the nominal robot's sample count and solver settings.

The design vector is `[thigh_length_m, shin_length_m, density_proxy, hip_width_m]`.
Nominal values are approximately `[0.34137, 0.26077, 2.0, 0.266319]`. Leg-length bounds are
about +/-10%; the proxy is in [1.8, 2.2]. For each thigh/shin, mass scales as
`length_ratio * proxy / 2`, and inertia scales as `mass_ratio * length_ratio**2`.
The builder scales that segment's geometry and child anchors. Feet, torso,
joint limits, actuator gains and torque limits stay fixed. This is a synthetic
parameterized design family, not a CAD structural-strength model or literal
material-density optimization. Initial floor placement, reference root height,
and fall-height thresholds are derived from each generated model.
Hip width is the lateral distance between the left/right hip-yaw anchors in
the torso frame, with bounds [0.240, 0.293] m. Complete leg assemblies are
translated symmetrically about the original anchor midpoint; their rotations,
internal joint positions and local masses/inertias are preserved. The torso
mesh and its local inertia are not stretched, so this is a hip mounting-spacing
design parameter, not a generated wider pelvis structure. Whole-body COM and
inertia include the translated legs automatically. Legacy three-value designs
still build with nominal hip spacing, and old saved trajectories remain usable.
CMA's initial per-variable sampling spread is scaled by each variable's
allowed range, so metres and the dimensionless mass proxy are not treated as
the same unit. `sigma0` gives the initial thigh-length spread in metres.

## Ranking

For geometry exploration at original robot weight, use
`configs/codesign_fixed_mass_quick.toml`. Its robot setting
`preserve_link_masses = true` keeps every original body mass, including when
link lengths change. Segment COM locations scale with geometry and segment
inertias scale with length ratio squared. The mass proxy is excluded from CMA
through `fixed_density_proxy = 2.0`; the legacy four-value saved design vector
remains compatible. The remaining free parameters are thigh length, shin
length, and hip width. Older presets retain their original mass-scaling policy.

The outer objective compares these quantities lexicographically:

1. Fraction of episodes completed without failure (higher is better).
2. Mean fraction of requested duration survived (higher is better).
3. Mean running cost per simulated second (lower is better).

Cost is integrated using the physics timestep before dividing by episode time.
The MPC costs are unchanged. This separates design selection from control
planning and prevents short failed episodes winning by accumulating less cost.
The existing distance-minus-cost `fitness` is retained as a legacy diagnostic;
it is NOT the selection objective. Existing fitness plots show this diagnostic.
CMA-ES receives within-generation ranks, with ties preserved. This retains its
ranking-based selection without arbitrarily huge scalar penalty constants.

Every design uses the same evaluation seed and therefore the same episode
seed set. Those are search seeds: verify any promising design on additional
held-out seeds before claiming improvement. Numerical invalidity during model
construction or warmup ranks below completed evaluable designs.
The starting morphology is evaluated first and retained as the best design
unless a candidate beats its objective. Its trajectories are in `baseline/`.

## Commands

From `/home/rory/coDesign/dial_lerobot`:

```bash
# Nominal reference evaluation.
./run_mpc.sh test --config configs/codesign_walking.toml --output results/codesign_nominal

# Small validation: initial morphology plus two candidates, two episodes each.
./run_mpc.sh optimize --config configs/codesign_walking.toml --generations 1 --popsize 2 --episodes 2 --output results/codesign_smoke

# Initial search: up to five generations of six designs, four episodes each.
./run_mpc.sh optimize --config configs/codesign_walking.toml --output results/codesign_search

# Evaluate the selected morphology and saved controller configuration.
./run_mpc.sh test --design results/codesign_search/best_design.json --seed 91 --output results/codesign_validation
```

Use fresh output directories. Search writes `optimization.json`,
`current_generation.json`, and `best_design.json`. Each candidate also has a
directory `generation_000/candidate_000/` containing `evaluation.json`,
`metadata.json`, `model.xml`, and episode NPZ/CSV files. Replay those NPZ files
with `./run_mpc.sh replay PATH/episode_000.npz --rate 1.0`.

Each different morphology currently builds its own evaluator and compiles JAX
code. Compile time is recorded separately from episode execution. Executables
are released between designs to avoid accumulating GPU memory. The nominal
controller's execution time alone therefore understates co-design cost.
This search uses CMA-ES, not RL; an RL outer optimizer could reuse the model
builder, evaluator, and objective later.

## Initial smoke validation, 2026-10-01

Two changed designs and the nominal robot were evaluated on the same two seeds,
for up to 10 seconds each. Results are in
`/home/rory/mpc_results/codesign_smoke_20261001/comparison.json`.

| Design (thigh, shin, proxy) | Completed | Mean cost / simulated second |
| --- | --- | --- |
| Nominal (0.34137, 0.26077, 2.0) | 2/2 | 0.6519 |
| Candidate 1 (0.35060, 0.25844, 2.01295) | 1/2 | 1.3948 |
| Candidate 2 (0.36516, 0.25689, 1.99532) | 2/2 | 0.9644 |

Neither candidate beat nominal. Candidate 1 fell at 6.75 seconds in one
episode. Candidate compilation took 67.5 and 70.2 seconds; full 10-second
episodes took about 39--40 seconds to compute. The two-candidate search took
286 seconds. The nominal evaluation was run separately after that search.
The automatic nominal incumbent and range-scaled CMA spreads were added
after launching that smoke search and are covered by the unit tests; future
searches use them. This verifies integration, not optimization convergence.

## Four-parameter search launched 2026-10-01

Hip-anchor width was added and a full configured search was launched with
five generations, six candidates per generation, and four 10-second episodes
per design plus the nominal incumbent (up to 31 designs / 124 episodes).
Search results: `/home/rory/mpc_results/codesign_hipwidth_full_20261001/`.
The immutable TOML snapshot, launch manifest and stdout log are beside that
folder with suffixes `.toml`, `.launch.json` and `.log`.
The run starts from nominal `[0.3413655864, 0.2607679717, 2.0, 0.266319]`.
Six candidates is the retained initial population setting, not a claim that
five generations are enough to establish CMA-ES convergence in four dimensions.

## Broad exploration with 512 samples

`configs/codesign_extreme_512.toml` widens thigh bounds to [0.18, 0.52] m,
shin bounds to [0.13, 0.40] m, the mass proxy to [0.8, 4.0], and hip-anchor
width to [0.14, 0.42] m. It raises CMA sigma0 to 0.07 and DIAL samples to 512;
the remaining controller settings, evaluation seeds and 5x6 search budget
are retained. All 16 corners build grounded models with positive inertias
and unchanged actuator/joint limits. That geometry check does not assert
closed-loop stability at those limits; failures are handled by the evaluator.
The run launched on 2026-10-01 is at
`/home/rory/mpc_results/codesign_extreme512_20261001_180330/`, with its frozen
config and log beside it. It starts from the original nominal morphology and
keeps that baseline as incumbent unless a candidate improves the objective.
## Reusing compilation across morphologies

CMA-ES and the design heatmap now retain one `MorphologyKernel` for the search.
The MJX model arrays and morphology-dependent standing/fall heights and foot
references are runtime arguments. Changing thigh length, shin length, mass
proxy, or hip width therefore does not require a new executable. Sample count,
horizon, solver settings, costs, and robot topology still define the compiled
configuration and must stay fixed within one shared kernel.

The planner preserves geom ordering but replaces disabled visual mesh geoms
with disabled sphere placeholders, removing mesh assets from its model. All
active collision geoms, explicit body inertials, joints, and actuators remain
physical inputs. Saved XMLs and replay retain the original detailed visual
meshes. Height-field terrain is unsupported by this shared path; the existing
flat ground is supported. The original evaluator path remains available for
comparison through `Evaluator(cfg, theta)` without a shared kernel.

The wide search at 256 samples is configured in
`configs/codesign_extreme_256.toml`. Launch it with:

```bash
./run_mpc.sh optimize --config configs/codesign_extreme_256.toml --output /home/rory/mpc_results/codesign_extreme256_shared
```

To repeat the single-episode GPU profile, from the repository directory:

```bash
PYTHONPATH=. XLA_PYTHON_CLIENT_PREALLOCATE=false /home/rory/co_design/.venv/bin/python scripts/profile_shared_morphology.py --output /home/rory/mpc_results/shared256_profile_repeat
```

That script checks compiled-cache reuse across several morphologies before
recording GPU utilization and memory at 200 ms intervals during one episode.
Compilation is excluded from its GPU episode measurements.

Measured on the RTX 4090 (2026-10-02): the 256-sample profile completed
10 simulated seconds in 44.95 seconds with no fall. GPU utilization averaged
80.2% (95th percentile 87%), with 570 MiB peak device-wide memory use out of
24,564 MiB. Nominal and both wide-bound morphologies reused the two existing
startup/continued-step JIT variants, each taking about 0.44 seconds for its
first planning update after model creation. The initial cold warmup observed
during implementation took 133 seconds; a later launch using the disk cache
took 31.45 seconds. These measurements support memory capacity for concurrent
runs, but do not establish a throughput improvement; that requires a paired
concurrency benchmark. Results are saved in
`/home/rory/mpc_results/shared_morphology256_profile_20261002/`.
