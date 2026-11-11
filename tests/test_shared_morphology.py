import jax
import mujoco
import numpy as np
from dataclasses import replace

from dial_lerobot.assets.model_builder import build_model, initial_data
from dial_lerobot.config import load_config
from dial_lerobot.evaluator import Evaluator


def test_planning_model_preserves_physics_and_shared_structure():
    cfg = load_config("configs/codesign_walking.toml")
    kernel = None
    for theta in (None, [.337859605, .27841552, 1.26401876, .20143171],
                  cfg.design.lower_bounds, cfg.design.upper_bounds):
        visual = build_model(theta, cfg.robot)
        physics = build_model(theta, cfg.robot, include_visual=False)
        assert physics.nmesh == 0
        active = (visual.geom_contype != 0) | (visual.geom_conaffinity != 0)
        for name in ("geom_type", "geom_size", "geom_pos", "geom_quat",
                     "geom_contype", "geom_conaffinity", "geom_solref", "geom_solimp"):
            np.testing.assert_array_equal(getattr(visual, name)[active], getattr(physics, name)[active])
        for name in ("body_mass", "body_inertia", "body_pos", "body_ipos", "qpos0"):
            np.testing.assert_array_equal(getattr(visual, name), getattr(physics, name))
        a, b = initial_data(visual), initial_data(physics)
        for _ in range(20):
            a.ctrl[:] = b.ctrl[:] = np.linspace(-1, 1, visual.nu)
            mujoco.mj_step(visual, a)
            mujoco.mj_step(physics, b)
        np.testing.assert_allclose(a.qpos, b.qpos, rtol=0, atol=1e-12)
        np.testing.assert_allclose(a.qvel, b.qvel, rtol=0, atol=1e-12)
        evaluator = Evaluator(cfg, theta, shared_kernel=kernel, reuse_compilation=True)
        kernel = evaluator.shared_kernel
        assert jax.tree.structure(kernel.inputs(evaluator.task)) == kernel.structure
        assert float(kernel.inputs(evaluator.task)[1][0]) == np.float32(evaluator.task.rest_height)


def test_shared_step_matches_original_and_does_not_recompile():
    cfg = load_config("configs/codesign_walking.toml")
    cfg = replace(cfg, mpc=replace(cfg.mpc, num_samples=8, num_knots=3,
                                  plan_horizon=0.04, initial_diffusion_iterations=1))
    theta = [.337859605, .27841552, 1.26401876, .20143171]
    original = Evaluator(cfg, theta)
    shared = Evaluator(cfg, theta, reuse_compilation=True)
    a, pa = original.reset(seed=123, height_offset=0)
    b, pb = shared.reset(seed=123, height_offset=0)
    a, pa, ra = original.step(a, pa)
    b, pb, rb = shared.step(b, pb)
    jax.block_until_ready(b.qpos)
    for name in ("qpos", "qvel", "action", "cost_terms", "foot_bottoms", "plan_cost_min"):
        np.testing.assert_allclose(ra[name], rb[name], rtol=2e-4, atol=2e-4)
    kernel = shared.shared_kernel
    assert kernel.step._cache_size() == 1
    other = Evaluator(cfg, shared_kernel=kernel)
    x, px = other.reset(seed=123, height_offset=0)
    x, px, row = other.step(x, px)
    jax.block_until_ready(x.qpos)
    assert kernel.step._cache_size() == 1
    assert not np.isclose(other.task.rest_height, shared.task.rest_height)
    assert np.isfinite(np.asarray(row["cost_terms"])).all()


def test_gait_changes_reuse_kernel_and_match_fresh_evaluator():
    cfg = load_config("configs/codesign_walking.toml")
    cfg = replace(cfg, mpc=replace(cfg.mpc, num_samples=8, num_knots=3,
                                  plan_horizon=0.04, initial_diffusion_iterations=1))
    template = Evaluator(cfg, reuse_compilation=True)
    state, params = template.reset(seed=123, height_offset=0)
    jax.block_until_ready(template.step(state, params))
    kernel = template.shared_kernel
    tuned = replace(cfg, cost=replace(cfg.cost, duty=0.65, cadence=1.3, gait_start_delay=0.1))
    shared = Evaluator(tuned, shared_kernel=kernel)
    fresh = Evaluator(tuned, reuse_compilation=True)
    for time in (0.05, 0.45, 0.8):
        state, params = shared.reset(seed=123, height_offset=0)
        state = state.replace(time=state.time * 0 + time)
        baseline_row = template.step(state, params)[2]
        shared_row = shared.step(state, params)[2]
        fresh_row = fresh.step(state, params)[2]
        jax.block_until_ready(shared_row)
        for name in ("qpos", "qvel", "action", "cost_terms", "foot_target_z"):
            np.testing.assert_allclose(shared_row[name], fresh_row[name], rtol=2e-4, atol=2e-4)
        if time == 0.45:
            assert not np.allclose(shared_row["foot_target_z"], baseline_row["foot_target_z"])
    assert kernel.step._cache_size() == 1
