import jax.numpy as jnp
import numpy as np

from hydrax.tasks.pendulum import Pendulum

from dial_lerobot.controllers import LeRobotDIAL


def make_controller(shift_reconstruction):
    return LeRobotDIAL(
        Pendulum(),
        num_samples=2,
        plan_horizon=0.4,
        num_knots=4,
        spline_type="zero",
        noise_level=0.3,
        beta_opt_iter=1.0,
        beta_horizon=0.9,
        temperature=0.05,
        shift_reconstruction=shift_reconstruction,
        shift_steps=1,
    )


def test_shift_reconstruction_is_finite_and_preserves_disabled_path():
    initial_knots = jnp.arange(4.0)[:, None]
    state_time = jnp.asarray(0.01)
    new_tk = jnp.linspace(0.0, 0.4, 4) + state_time

    disabled = make_controller(False)
    params = disabled.init_params(initial_knots=initial_knots, seed=1)
    expected = disabled.interp_func(
        jnp.clip(new_tk, params.tk[0], params.tk[-1]),
        params.tk,
        params.mean[None, ...],
    )[0]
    actual = disabled._warm_start_mean(state_time, params, new_tk)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    enabled = make_controller(True)
    params = enabled.init_params(initial_knots=initial_knots, seed=1)
    shifted = enabled._warm_start_mean(state_time, params, new_tk)
    assert shifted.shape == params.mean.shape
    assert np.isfinite(np.asarray(shifted)).all()
    assert not np.allclose(np.asarray(shifted), np.asarray(initial_knots))
