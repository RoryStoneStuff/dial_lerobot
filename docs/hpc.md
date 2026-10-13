# Running and collaborating from an HPC server

Clone this repository into writable storage on the HPC login node. Use Python
3.12 or newer. Create the environment and fetch assets while internet access
is available; the compute job itself does not need network access.

```bash
cd dial_lerobot
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[cuda12,dev]'
.venv/bin/python scripts/fetch_robot_assets.py
.venv/bin/python -c 'import jax; print(jax.devices())'
```

The `cuda12` extra installs NVIDIA CUDA runtime dependencies through pip.
Use `cuda13` instead if the server supports that version. Run the device
check inside a GPU allocation, not on a login node without GPUs. The NVIDIA
driver comes from the cluster; follow its GPU environment/module instructions.
For CPU-only checks, use `pip install -e '.[dev]'` instead.

HydraX is pinned to the exact clean revision used in the local experiment
environment. The asset manifest pins the zoo revision and checks SHA-256
hashes. Its patch restores the twelve joint ranges that were present in our
local zoo checkout but absent from upstream. Fetching these assets reproduces
the robot rather than silently using upstream joint limits.

The launcher automatically uses this repository's `.venv`. Assets fetched to
`external/lerobot-legged-zoo` are discovered automatically. To store assets
elsewhere, pass `--destination` to the fetch script and set `LEROBOT_ZOO_ROOT`
to that directory. For a different Python environment, set
`LEROBOT_MPC_PYTHON` to its Python executable.

Within a GPU allocation, run:

```bash
./run_mpc.sh inspect --config configs/codesign_fixed_mass_moderate_15gen.toml
./run_mpc.sh optimize --config configs/codesign_fixed_mass_moderate_15gen.toml --output results/hpc_15gen_01
```

Allocate one GPU. The current search evaluates morphologies sequentially and
reuses the compiled MPC function within the process. Set
`DIAL_LEROBOT_CACHE_DIR` to user-owned persistent storage if the default home
cache is unsuitable. Cache files compiled locally need not be transferred;
the HPC may use different hardware and compile its own variants.

For Git changes on either machine:

```bash
git pull --ff-only
git switch -c my-experiment
# Edit and review changes.
git add configs docs dial_lerobot scripts tests
git commit -m "Describe the experiment change"
git push -u origin my-experiment
```

Authenticate to GitHub separately on the HPC, using its approved HTTPS or
SSH method. Each machine uses its own environment, credentials, and JAX cache.
Do not work on the same branch simultaneously without pulling and resolving
changes. Generated results, downloaded meshes, environments, and caches are
excluded from Git; transfer result folders separately when needed. Saved
trajectory XMLs contain absolute mesh paths, so replay on another machine
needs those paths updated to its asset location.
