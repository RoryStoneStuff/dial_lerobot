from pathlib import Path
import os

import jax
from jax.experimental.compilation_cache import compilation_cache
import pytest

from dial_lerobot.runtime import configure_jax_cache


def test_cache_overrides_shared_hydrax_default(monkeypatch, tmp_path):
    monkeypatch.delenv("LEROBOT_MPC_CACHE_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    previous = jax.config.jax_compilation_cache_dir
    try:
        jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
        actual = configure_jax_cache()
        assert actual.is_relative_to(tmp_path)
        assert str(os.geteuid()) in str(actual.relative_to(tmp_path))
        assert Path(jax.config.jax_compilation_cache_dir) == actual
        probe = actual / "xla_gpu_per_fusion_autotune_cache_dir" / "tmp" / "probe"
        probe.write_text("writable")
    finally:
        jax.config.update("jax_compilation_cache_dir", previous)


def test_explicit_cache_with_spaces(monkeypatch, tmp_path):
    override = tmp_path / "custom cache"
    monkeypatch.setenv("LEROBOT_MPC_CACHE_DIR", str(override))
    previous = jax.config.jax_compilation_cache_dir
    try:
        assert configure_jax_cache() == override
    finally:
        jax.config.update("jax_compilation_cache_dir", previous)


def test_unusable_nested_cache_fails_before_compilation(monkeypatch, tmp_path):
    monkeypatch.setenv("LEROBOT_MPC_CACHE_DIR", str(tmp_path))
    (tmp_path / "xla_gpu_per_fusion_autotune_cache_dir").write_text("not a directory")
    with pytest.raises(PermissionError, match="DIAL_LEROBOT_CACHE_DIR"):
        configure_jax_cache()


def test_initialized_file_cache_moves_to_new_directory(monkeypatch, tmp_path):
    old = tmp_path / "old"
    new = tmp_path / "new"
    monkeypatch.setenv("LEROBOT_MPC_CACHE_DIR", str(new))
    settings = {
        "jax_compilation_cache_dir": str(old),
        "jax_persistent_cache_min_compile_time_secs": 0,
        "jax_persistent_cache_min_entry_size_bytes": -1,
        "jax_raise_persistent_cache_errors": True,
    }
    previous = {name: getattr(jax.config, name) for name in settings}
    try:
        compilation_cache.reset_cache()
        for name, value in settings.items():
            jax.config.update(name, value)
        jax.jit(lambda value: value + 1)(1).block_until_ready()
        old_entries = set(old.glob("*-cache"))
        assert old_entries

        configure_jax_cache()
        jax.jit(lambda value: value + 2)(1).block_until_ready()
        assert list(new.glob("*-cache"))
        assert set(old.glob("*-cache")) == old_entries
    finally:
        compilation_cache.reset_cache()
        for name, value in previous.items():
            jax.config.update(name, value)
