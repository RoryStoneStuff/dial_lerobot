"""User-owned JAX caches for the standalone LeRobot application."""

import os
from pathlib import Path
import pwd
import tempfile

import jax
from jax.experimental.compilation_cache import compilation_cache


def configure_jax_cache() -> Path:
    """Override HydraX's shared /tmp cache before compiling the controller.

    Call after importing HydraX: its package initializer overwrites JAX's
    environment-based cache setting. Separate accounts must not share writes.
    """
    account_home = Path(pwd.getpwuid(os.geteuid()).pw_dir)
    cache_base = Path(os.environ.get("XDG_CACHE_HOME", account_home / ".cache"))
    default = cache_base / f"lerobot_mpc-{os.geteuid()}" / "jax"
    cache = Path(os.environ.get("LEROBOT_MPC_CACHE_DIR", default)).expanduser().resolve()
    # Check the actual nested location used by XLA too: a writable parent can
    # still contain root-owned directories left by an earlier privileged run.
    autotune = cache / "xla_gpu_per_fusion_autotune_cache_dir"
    for directory in (cache, autotune, autotune / "tmp"):
        try:
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            if directory.stat().st_uid != os.geteuid():
                raise PermissionError("directory belongs to another account")
            with tempfile.TemporaryFile(dir=directory):
                pass
        except OSError as exc:
            raise PermissionError(
                f"Cannot write the MPC compilation cache at {directory}: {exc}. "
                "Set LEROBOT_MPC_CACHE_DIR to a directory owned by your account."
            ) from exc
    if jax.config.jax_compilation_cache_dir != str(cache):
        # Imports can initialize JAX's file cache already. Updating its config
        # alone changes XLA autotuning paths but leaves that file cache in /tmp.
        compilation_cache.reset_cache()
        compilation_cache.set_cache_dir(str(cache))
    return cache
