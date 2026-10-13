from dataclasses import replace

import numpy as np
import mujoco
from dial_lerobot.assets.foot_geometry import foot_capsules, capsule_bottoms
import pytest

from dial_lerobot.config import load_config
from dial_lerobot.evaluator import Evaluator
from dial_lerobot.visualize_lerobot import load_trajectory


@pytest.mark.integration
def test_dial_seed_bounds_logs_and_replay(tmp_path):
    cfg = load_config("configs/smoke.toml")
    cfg = replace(cfg, evaluation=replace(cfg.evaluation, episode_steps=2))
    evaluator = Evaluator(cfg)
    evaluator.warmup()
    result = evaluator.evaluate(seed=9, output=tmp_path)
    repeated = evaluator.evaluate(seed=9)
    # GPU float32 contact solves need numerical repeatability, not bit identity.
    assert result["fitness"] == pytest.approx(repeated["fitness"], abs=1e-5)
    assert np.isfinite(result["fitness"])
    model, trajectory = load_trajectory(tmp_path / "episode_000.npz")
    assert model.nu == 12
    assert trajectory["qpos"].shape == (2, 19)
    assert np.all(trajectory["ctrl"] >= model.actuator_ctrlrange[:, 0] - 1e-7)
    assert np.all(trajectory["ctrl"] <= model.actuator_ctrlrange[:, 1] + 1e-7)
    assert trajectory["time"][-1] == pytest.approx(cfg.robot.timestep * cfg.mpc.control_decimation * 2, abs=1e-6)
    assert (tmp_path / "episode_000.csv").is_file()
    with np.load(tmp_path / "episode_000.npz") as archive:
        d = mujoco.MjData(model)
        d.qpos[:] = archive["qpos"][-1]
        d.qvel[:] = archive["qvel"][-1]
        mujoco.mj_forward(model, d)
        np.testing.assert_allclose(archive["foot_bottoms"][-1],
                                   capsule_bottoms(model, d, foot_capsules(model)), atol=1e-6)
