# HPC handoff for a Codex agent

Your task is to make this project runnable on the user's HPC system, preserve
the established experiment, and perform the specific run the user requests.
For an initial setup request, install and validate first; do not automatically
start the four-hour search described below. User instructions define which
experiments to launch.

## 1. Clone and read the project

The private repository is `https://github.com/RoryStoneStuff/dial_lerobot`.
Authenticate as an account with access using the HPC's approved GitHub method.
Authentication on the original workstation does not authenticate this machine.
Do not request that tokens be pasted into chat or committed into files.

From a writable project directory:

```bash
git clone https://github.com/RoryStoneStuff/dial_lerobot.git
cd dial_lerobot
git status --short --branch
git rev-parse HEAD
```

Read `AGENTS.md`, `docs/hpc.md`, `docs/codesign.md`, and
`configs/codesign_fixed_mass_moderate_15gen.toml`. If the repository already
exists, inspect it and preserve local changes before updating it. Do not clone
over or reset an existing checkout.

## 2. Inspect the cluster and install the environment

Identify writable persistent storage, Python 3.12+, available modules, the
job scheduler, GPU allocation procedure, and any required account/partition.
Use cluster documentation and existing job examples. Ask the user only for
required details that cannot be discovered. Do not install a system NVIDIA
driver or change cluster-wide software.

On a node where installation/network access is allowed:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[cuda12,dev]'
.venv/bin/python scripts/fetch_robot_assets.py
```

Select `cuda12` or `cuda13` based on the allocated GPU and driver capabilities;
CUDA 13 was used on the original workstation, but is not required by the
experiment. Follow cluster module guidance and avoid mixing conflicting CUDA
libraries. If Python is available through a module or Conda, use that Python
to create the environment rather than assuming `python3.12` is on PATH.

The asset fetcher downloads only the 55 required files, verifies their hashes,
and applies the original local joint-limit patch. The full zoo, PPO, Brax,
and other neighboring workstation repositories are not required for this
MPC/co-design workflow. HydraX is installed from its pinned Git revision.

The launcher automatically uses `.venv` and the fetched assets. Set
`LEROBOT_MPC_PYTHON` or `LEROBOT_ZOO_ROOT` only for alternate locations.
Set `DIAL_LEROBOT_CACHE_DIR` to writable user-owned persistent storage if the
default home cache is unsuitable. Each machine compiles for its own hardware.

## 3. Validate before a long run

Lightweight checks, on a node where CPU checks are allowed:

```bash
JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/test_codesign.py tests/test_runtime.py tests/test_failure_logging.py -q
./run_mpc.sh inspect --config configs/codesign_fixed_mass_moderate_15gen.toml
```

Inside an actual GPU allocation:

```bash
nvidia-smi
.venv/bin/python -c 'import jax; print(jax.devices()); assert any(d.platform == "gpu" for d in jax.devices()), "No JAX GPU available"'
./run_mpc.sh test --config configs/codesign_fixed_mass_moderate_15gen.toml --episodes 1 --output results/hpc_baseline_01
```

Use a new output name if one already exists. The baseline runs for at most
10 simulated seconds and stops early on a fall. A fall is an experimental
result; report it rather than secretly changing gains, costs, timestep,
solver iterations, or sample count to make setup look successful. Report
non-finite states or installation errors separately from ordinary falls.

## 4. Run the requested morphology optimization

Reference settings:

| Setting | Value |
|---|---|
| Physics timestep / solver / line search | 0.01 s / 2 / 2 |
| DIAL samples / horizon / knots | 256 / 0.4 s / 6 |
| Spline / warm-start shift | Quadratic DIAL / reconstruction enabled |
| Noise / temperature | 0.05 / 0.02 |
| Forward target | 0.85 m/s with a 3-second ramp |
| Search | 15 generations, 6 candidates, 4 episodes each |
| Episode duration | 10 simulated seconds, stop on fall |
| Free morphology parameters | Thigh length, shin length, hip width |
| Bounds | 0.256–0.427 m / 0.196–0.326 m / 0.226–0.306 m |
| Initial thigh sampling SD | 0.035 m; other free dimensions range-scaled |
| Mass policy | Each original body mass fixed; total 14.3817786 kg |
| Search / evaluation seed | 42 / 45 |

After the user requests the full search, launch this within the GPU job:

```bash
./run_mpc.sh optimize --config configs/codesign_fixed_mass_moderate_15gen.toml --output results/hpc_15gen_01
```

Generate a cluster-appropriate job script after discovering its scheduler.
Set the working directory, Python/modules if required, one GPU, appropriate
CPU/RAM, a job time limit, and stdout/stderr files. Report the submitted job
ID and monitor its state. On a scheduled cluster, a `nohup` process on the
login node is not a substitute for a GPU allocation.

The original RTX 4090 search took about 3 hours 56 minutes. This is a reference
measurement, not an HPC runtime guarantee. There are up to 364 episodes
including the nominal baseline; early falls shorten many of them. Initial
compilation is separate from episode execution. Reuse the shared kernel
throughout the search; candidate warmups locally took roughly 0.5 seconds.

Inspect `optimization.json`, `current_generation.json`, `best_design.json`,
and candidate `evaluation.json` files to report progress and final outcomes.
Selection favors completion first, then survival, then cost. A failed design
with a small accumulated cost must not beat a completed design.

## 5. Reference result and subsequent validation

The workstation's best 15-generation design was:

```text
theta = [0.29729056757593436, 0.23983019494331373, 2.0, 0.23611481653778235]
# thigh metres, shin metres, fixed legacy mass slot, hip width metres
```

It completed four 10-second search episodes, traveled 5.60–6.06 m per episode,
and had mean running cost per simulated second 0.5584. It appeared in
generation 2, and subsequent generations did not improve it. Those results
are from the search seeds, not held-out validation. Raw workstation result
folders are not in GitHub. Expect numerical differences on another GPU.

For a requested comparison, pass the vector through `--theta` and retain the
same reference config. Use additional seeds and longer episodes when the
user asks to validate reliability. Do not claim this geometry is globally
optimal or guaranteed to walk on new hardware.

## 6. Save the handoff

Summarize the installation, exact commands, Git revision, selected CUDA extra,
Python/dependency versions, GPU, scheduler script/job ID, config, output paths,
and observed results. Keep environment snapshots and full trajectory artifacts
in run storage. Push useful code/config/docs changes on an HPC-specific branch
when the user requests it; do not commit environments, assets, or secrets.

GUI replay generally belongs on a workstation with a display. Export result
folders from the cluster separately and account for absolute mesh paths in
their saved XMLs before replaying on another machine.
