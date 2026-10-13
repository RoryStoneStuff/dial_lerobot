from pathlib import Path

import jax.numpy as jnp
import numpy as np

from test_dial_h1_profile import make_h1_controller


def test_beta_opt_iter_controls_regular_diffusion_passes():
    _, controller = make_h1_controller()
    controller.iterations = 4
    horizon_sigma = 0.9 ** np.arange(5, -1, -1)

    for iteration in range(4):
        expected = horizon_sigma * np.exp(-iteration / (controller.beta_opt_iter * 4))
        np.testing.assert_allclose(
            controller.noise_std_by_knot(iteration), expected, rtol=1e-6, atol=1e-7
        )


def test_regular_diffusion_iteration_wraps_after_configured_passes():
    _, controller = make_h1_controller()
    controller.iterations = 4
    params = controller.init_params(
        initial_knots=jnp.zeros((controller.num_knots, controller.task.model.nu)), seed=7
    ).replace(opt_iteration=jnp.asarray(3, dtype=jnp.int32))

    _, next_params = controller.sample_knots(params)
    assert int(next_params.opt_iteration) == 0
