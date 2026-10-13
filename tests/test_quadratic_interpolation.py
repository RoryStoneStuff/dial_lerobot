import jax
import jax.numpy as jnp
import numpy as np

from dial_lerobot.config import Config, MPCConfig
from dial_lerobot.controllers import interp_quadratic


def test_quadratic_interpolator_matches_quadratic_polynomial():
    tk = jnp.arange(4.0)
    knots = (0.1 * tk**2)[None, :, None]
    tq = jnp.array([0.0, 0.25, 0.75, 1.5, 2.25, 3.0])
    actual = interp_quadratic(tq, tk, knots)[0, :, 0]
    expected = 0.1 * tq**2
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_quadratic_interpolator_is_jittable_and_bounded():
    tk = jnp.linspace(0.0, 0.4, 6)
    knots = jnp.array([[[1.0], [-1.0], [1.0], [-1.0], [1.0], [-1.0]]])
    tq = jnp.linspace(0.0, 0.4, 41)
    actual = jax.jit(interp_quadratic)(tq, tk, knots)
    assert np.isfinite(np.asarray(actual)).all()
    assert np.max(np.asarray(actual)) <= 1.0
    assert np.min(np.asarray(actual)) >= -1.0


def test_quadratic_spline_requires_three_knots():
    cfg = Config(mpc=MPCConfig(num_knots=2, spline_type="quadratic"))
    try:
        cfg.validate()
    except ValueError as exc:
        assert "at least 3 knots" in str(exc)
    else:
        raise AssertionError("quadratic interpolation accepted only two knots")
