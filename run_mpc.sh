#!/usr/bin/env bash
set -euo pipefail
mpc_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mpc_python="${LEROBOT_MPC_PYTHON:-}"
if [[ -z "$mpc_python" && -x "$mpc_dir/.venv/bin/python" ]]; then
  mpc_python="$mpc_dir/.venv/bin/python"
elif [[ -z "$mpc_python" ]]; then
  mpc_python="$mpc_dir/../../co_design/.venv/bin/python"
fi
if [[ ! -x "$mpc_python" ]]; then
  echo "Set LEROBOT_MPC_PYTHON to a Python environment with HydraX, MJX, JAX and CMA installed." >&2
  exit 1
fi
export PYTHONPATH="$mpc_dir${PYTHONPATH:+:$PYTHONPATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
cd "$mpc_dir"
exec "$mpc_python" -m dial_lerobot.run_experiment "$@"
