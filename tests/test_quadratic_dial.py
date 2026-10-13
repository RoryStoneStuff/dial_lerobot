import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hydrax.tasks.pendulum import Pendulum
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline

from dial_lerobot.config import Config, MPCConfig
from dial_lerobot.controllers import LeRobotDIAL, interp_quadratic, interp_quadratic_dial


@pytest.mark.parametrize("query", [jnp.array(0.23), jnp.array([-0.1, 0.0, 0.13, 0.23, 0.5, 0.7])])
def test_dial_matches_library_batched_jit(query):
    times = jnp.array([0.0, 0.08, 0.2, 0.31, 0.5])
    knots = jnp.array([
        [[0.1, 0.3], [-0.5, 0.2], [0.9, -0.8], [0.2, 0.5], [-0.4, 0.9]],
        [[0.4, -0.1], [0.3, 0.7], [-0.6, -0.2], [-0.1, 0.8], [0.2, -0.4]],
    ])
    expected = np.stack([
        np.stack([
            np.asarray(InterpolatedUnivariateSpline(times, values[:, actuator], k=2)(query))
            for actuator in range(values.shape[1])
        ], axis=-1)
        for values in knots
    ])
    actual = jax.jit(interp_quadratic_dial)(query, times, knots)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_dial_preserves_knots_and_overshoot():
    times = jnp.arange(4.0)
    knots = jnp.array([[[1.0], [-1.0], [1.0], [-1.0]]])
    np.testing.assert_allclose(interp_quadratic_dial(times, times, knots), knots, atol=1e-6)
    query = jnp.linspace(0.0, 3.0, 101)
    actual = interp_quadratic_dial(query, times, knots)
    assert float(jnp.max(jnp.abs(actual))) > 1.0
    assert not np.allclose(actual, interp_quadratic(query, times, knots))


def test_dial_minimum_knots_and_polynomial():
    times = jnp.array([0.0, 0.2, 0.7])
    query = jnp.linspace(-0.1, 0.9, 21)
    actual = interp_quadratic_dial(query, times, (times**2)[None, :, None])
    np.testing.assert_allclose(actual[0, :, 0], query**2, atol=1e-6)
    Config(mpc=MPCConfig(num_knots=3, spline_type="quadratic_dial")).validate()
    with pytest.raises(ValueError, match="at least 3"):
        Config(mpc=MPCConfig(num_knots=2, spline_type="quadratic_dial")).validate()
    with pytest.raises(ValueError, match="at least 3"):
        interp_quadratic_dial(query, times[:2], jnp.zeros((1, 2, 1)))


@pytest.mark.parametrize("spline_type, expected", [
    ("quadratic", interp_quadratic), ("quadratic_dial", interp_quadratic_dial),
])
def test_controller_selects_spline(spline_type, expected):
    controller = LeRobotDIAL(
        Pendulum(), num_samples=2, plan_horizon=0.4, num_knots=4,
        spline_type=spline_type, noise_level=0.3, beta_opt_iter=1.0,
        beta_horizon=0.9, temperature=0.05,
    )
    assert controller.spline_type == spline_type
    assert controller.interp_func is expected
    params = controller.init_params()
    actual = controller.get_action(params, jnp.array(0.1))
    assert np.isfinite(np.asarray(actual)).all()
