from pathlib import Path

import jax.numpy as jnp
import numpy as np

from dial_lerobot.assets.model_builder import build_model
from dial_lerobot.config import load_config
from dial_lerobot.controllers import LeRobotDIAL
from dial_lerobot.tasks.lerobot_task import LeRobotLocomotionTask


def make_h1_controller():
    config = load_config(Path("configs/h1_baseline_lerobot.toml"))
    model = build_model(cfg=config.robot)
    task = LeRobotLocomotionTask(model, config.cost)
    mpc = config.mpc
    controller = LeRobotDIAL(
        task,
        num_samples=mpc.num_samples,
        noise_level=mpc.noise_level,
        beta_opt_iter=mpc.beta_opt_iter,
        beta_horizon=mpc.beta_horizon,
        temperature=mpc.temperature,
        plan_horizon=mpc.plan_horizon,
        num_knots=mpc.num_knots,
        iterations=mpc.iterations,
        spline_type=mpc.spline_type,
        initial_diffusion_iterations=mpc.initial_diffusion_iterations,
        trajectory_diffuse_factor=mpc.trajectory_diffuse_factor,
        normalize_costs_for_temperature=mpc.normalize_costs_for_temperature,
        append_mean_sample=mpc.append_mean_sample,
        fix_initial_knot=mpc.fix_initial_knot,
    )
    return config, controller


def test_h1_baseline_dial_resolution_and_noise_schedule():
    config, controller = make_h1_controller()
    assert config.mpc.num_knots == 6  # H1 Hnode=5 means Hnode + 1 values.
    assert config.mpc.plan_horizon == 0.4
    assert config.mpc.num_samples == 2048
    assert config.mpc.temperature == 0.05

    expected_h1_horizon_sigma = 0.9 ** np.arange(5, -1, -1)
    params = controller.init_params(
        initial_knots=jnp.zeros((controller.num_knots, controller.task.model.nu)), seed=7
    )
    assert int(params.opt_iteration) == -10
    np.testing.assert_allclose(
        controller.noise_std_by_knot(params.opt_iteration), expected_h1_horizon_sigma,
        rtol=1e-6, atol=1e-7,
    )
    np.testing.assert_allclose(
        controller.noise_std_by_knot(-9), expected_h1_horizon_sigma * 0.2,
        rtol=1e-6, atol=1e-7,
    )
    np.testing.assert_allclose(
        controller.noise_std_by_knot(0), expected_h1_horizon_sigma,
        rtol=1e-6, atol=1e-7,
    )

    candidates, next_params = controller.sample_knots(params)
    assert candidates.shape == (2049, 6, 12)  # Nsample plus current mean.
    assert np.all(np.asarray(candidates[:-1, 0]) == 0.0)  # Current action knot is fixed.
    np.testing.assert_array_equal(candidates[-1], params.mean)
    assert int(next_params.opt_iteration) == -9


def test_h1_temperature_is_invariant_to_cost_units():
    _, controller = make_h1_controller()
    costs = jnp.array([1.0, 1.1, 1.4, 2.0])
    weights = controller.cost_weights(costs)
    rescaled_weights = controller.cost_weights(17.0 * costs + 53.0)
    np.testing.assert_allclose(weights, rescaled_weights, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(jnp.sum(weights), 1.0, atol=1e-7)


def test_h1_temperature_normalization_survives_extreme_finite_rollout_costs():
    _, controller = make_h1_controller()
    costs = jnp.array([0.079, 0.2, 2.26e27])
    weights = controller.cost_weights(costs)
    assert np.isfinite(np.asarray(weights)).all()
    np.testing.assert_allclose(jnp.sum(weights), 1.0, atol=1e-7)
    assert float(weights[2]) < 1e-6
    np.testing.assert_allclose(weights[:2], jnp.array([0.5, 0.5]), atol=1e-7)
