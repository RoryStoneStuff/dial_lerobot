"""HydraX DIAL with the official normalized-action biped interface."""

from functools import partial
import jax
import jax.numpy as jnp
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline
from hydrax.alg_base import Trajectory
from hydrax.algs import DIAL
from mujoco import mjx


# A failed sampled rollout should be treated as a very bad candidate, rather
# than allowing one NaN/Inf to poison the whole DIAL softmax update. This is
# deliberately finite so the other samples can still be weighted and used.
INVALID_ROLLOUT_COST = 1.0e6


@partial(jax.vmap, in_axes=(None, None, 0))
def interp_quadratic(tq, tk, knots):
    """Bounded local quadratic interpolation, matching H1's k=2 intent.

    Each query uses the nearest three knots and a quadratic Lagrange
    polynomial.  Clipping preserves DIAL's normalized action bounds when a
    quadratic segment overshoots between knots.
    """
    num_knots = knots.shape[0]
    if num_knots < 3:
        raise ValueError("quadratic interpolation requires at least 3 knots")
    left = jnp.clip(jnp.searchsorted(tk, tq, side="right") - 2, 0, num_knots - 3)
    i0, i1, i2 = left, left + 1, left + 2
    t0, t1, t2 = tk[i0], tk[i1], tk[i2]
    y0, y1, y2 = knots[i0], knots[i1], knots[i2]
    w0 = (tq - t1) * (tq - t2) / ((t0 - t1) * (t0 - t2))
    w1 = (tq - t0) * (tq - t2) / ((t1 - t0) * (t1 - t2))
    w2 = (tq - t0) * (tq - t1) / ((t2 - t0) * (t2 - t1))
    return jnp.clip(
        w0[..., None] * y0 + w1[..., None] * y1 + w2[..., None] * y2,
        -1.0, 1.0,
    )


@partial(jax.vmap, in_axes=(None, None, 0))
def interp_quadratic_dial(tq, tk, knots):
    """DIAL-MPC's jax-cosmo k=2 spline, including overshoot and extrapolation."""
    if knots.shape[0] < 3:
        raise ValueError("quadratic interpolation requires at least 3 knots")
    return jax.vmap(
        lambda values: InterpolatedUnivariateSpline(tk, values, k=2)(tq),
        in_axes=1,
        out_axes=-1,
    )(knots)


def _finite_state(state):
    """Return a scalar mask indicating whether every MJX data leaf is finite."""
    leaves = jax.tree_util.tree_leaves(state)
    return jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in leaves]))


def _freeze_state(new_state, old_state, keep_new):
    """Keep the last finite MJX state after a rollout becomes invalid."""
    return jax.tree_util.tree_map(
        lambda new, old: jnp.where(keep_new, new, old), new_state, old_state
    )


def _unit_cost_deltas(costs):
    """Return finite costs shifted and scaled to [0, 1], plus their validity mask."""
    finite = jnp.isfinite(costs)
    safe_costs = jnp.where(finite, costs, 0.0)
    offset = jnp.min(jnp.where(finite, safe_costs, jnp.inf))
    deltas = jnp.where(finite, safe_costs - offset, 0.0)
    span = jnp.max(deltas)
    return deltas / jnp.maximum(span, 1e-8), finite


class LeRobotDIAL(DIAL):
    """HydraX DIAL with the paper-repo H1 sampling/update conventions."""

    def __init__(self, *args, initial_diffusion_iterations=1,
                 trajectory_diffuse_factor=0.2,
                 normalize_costs_for_temperature=False,
                 append_mean_sample=False, fix_initial_knot=False,
                 shift_reconstruction=False, shift_steps=1, **kwargs):
        self.initial_diffusion_iterations = int(initial_diffusion_iterations)
        self.trajectory_diffuse_factor = float(trajectory_diffuse_factor)
        self.normalize_costs_for_temperature = bool(normalize_costs_for_temperature)
        self.append_mean_sample = bool(append_mean_sample)
        self.fix_initial_knot = bool(fix_initial_knot)
        self.shift_reconstruction = bool(shift_reconstruction)
        self.shift_steps = int(shift_steps)
        if self.shift_steps < 1:
            raise ValueError("shift_steps must be positive")
        spline_type = kwargs.get("spline_type", "zero")
        # Hydrax does not expose quadratic interpolation; install the local
        # JAX implementation after constructing its standard controller.
        if spline_type in ("quadratic", "quadratic_dial"):
            kwargs["spline_type"] = "linear"
        super().__init__(*args, **kwargs)
        if spline_type in ("quadratic", "quadratic_dial"):
            self.spline_type = spline_type
            self.interp_func = (
                interp_quadratic if spline_type == "quadratic" else interp_quadratic_dial
            )

    @partial(jax.vmap, in_axes=(None, None, None, 0, 0))
    def eval_rollouts(self, model, state, controls, knots):
        def _scan_fn(x, u):
            # Carry a validity bit through the scan. Once a sample becomes
            # invalid, keep its last finite state so subsequent MJX steps do
            # not repeatedly propagate NaNs through the trajectory.
            x, valid = x
            x_trial = x.replace(ctrl=self.task.action_to_control(x, u))
            x_trial = mjx.step(model, x_trial)
            x_trial = self.task.refresh_kinematics(model, x_trial)
            raw_cost = self.dt * self.task.running_cost(x_trial, u)
            next_valid = valid & _finite_state(x_trial) & jnp.isfinite(raw_cost)
            x_safe = _freeze_state(x_trial, x, next_valid)
            cost = jnp.where(next_valid, raw_cost, INVALID_ROLLOUT_COST)
            sites = self.task.get_trace_sites(x_safe)
            sites = jnp.where(jnp.isfinite(sites), sites, 0.0)
            return (x_safe, next_valid), (x_safe, cost, sites)

        (final_state, valid), (states, costs, trace_sites) = jax.lax.scan(
            _scan_fn, (state, jnp.array(True)), controls
        )
        terminal_cost = self.task.terminal_cost(final_state)
        final_cost = jnp.where(
            valid & jnp.isfinite(terminal_cost), terminal_cost, INVALID_ROLLOUT_COST
        )
        final_trace_sites = self.task.get_trace_sites(final_state)
        costs = jnp.append(costs, final_cost)
        trace_sites = jnp.append(trace_sites, final_trace_sites[None], axis=0)
        return states, Trajectory(controls=controls, knots=knots, costs=costs, trace_sites=trace_sites)

    def init_params(self, initial_knots=None, seed=0):
        params = super().init_params(initial_knots, seed)
        # Negative values mark the one-time H1 Ndiffuse_init annealing pass.
        return params.replace(opt_iteration=jnp.asarray(
            -self.initial_diffusion_iterations, dtype=jnp.int32
        ))

    def noise_std_by_knot(self, opt_iteration):
        """H1's geometric horizon profile, with its startup trajectory anneal."""
        knot = jnp.arange(self.num_knots)
        horizon_sigma = self.noise_level * jnp.exp(
            -(self.num_knots - 1 - knot)
            / (self.beta_horizon * self.num_knots)
        )
        in_initial_diffusion = opt_iteration < 0
        diffusion_index = jnp.clip(
            opt_iteration + self.initial_diffusion_iterations,
            0,
            self.initial_diffusion_iterations - 1,
        )
        trajectory_sigma = jnp.where(
            in_initial_diffusion,
            self.trajectory_diffuse_factor ** diffusion_index,
            jnp.exp(-jnp.maximum(opt_iteration, 0) / (self.beta_opt_iter * self.iterations)),
        )
        return horizon_sigma * trajectory_sigma

    def sample_knots(self, params):
        rng, sample_rng = jax.random.split(params.rng)
        noise = jax.random.normal(
            sample_rng,
            (self.num_samples, self.num_knots, self.task.model.nu),
        )
        knots = params.mean + self.noise_std_by_knot(params.opt_iteration)[None, :, None] * noise
        if self.fix_initial_knot:
            # H1 holds the already-current action knot fixed in every sample.
            knots = knots.at[:, 0, :].set(params.mean[0, :])
        if self.append_mean_sample:
            # The paper implementation evaluates Ybar alongside Nsample noisy candidates.
            knots = jnp.concatenate((knots, params.mean[None, ...]), axis=0)
        next_iteration = jnp.where(
            params.opt_iteration < 0, params.opt_iteration + 1,
            (params.opt_iteration + 1) % self.iterations,
        )
        return knots, params.replace(rng=rng, opt_iteration=next_iteration)

    def cost_weights(self, costs):
        """Convert candidate costs to MPPI weights using H1's optional scaling."""
        if self.normalize_costs_for_temperature:
            unit_deltas, finite = _unit_cost_deltas(costs)
            any_finite = jnp.any(finite)
            spread = jnp.maximum(jnp.std(unit_deltas), 1e-8)
            logits = jnp.where(
                finite, -unit_deltas / (spread * self.temperature), -jnp.inf
            )
            logits = jnp.where(any_finite, logits, jnp.zeros_like(logits))
        else:
            logits = -costs / self.temperature
        return jax.nn.softmax(logits, axis=0)

    def normalized_cost_spread(self, costs):
        """Finite, scale-free cost spread for telemetry when costs are huge."""
        unit_deltas, _ = _unit_cost_deltas(costs)
        return jnp.std(unit_deltas)

    def update_params(self, params, rollouts):
        costs = jnp.sum(rollouts.costs, axis=1)
        weights = self.cost_weights(costs)
        mean = jnp.sum(weights[:, None, None] * rollouts.knots, axis=0)
        return params.replace(mean=mean)

    def _warm_start_mean(self, state_time, params, new_tk):
        """Warm-start knots, optionally using H1's action-shift reconstruction."""
        if not self.shift_reconstruction:
            clamped_tk = jnp.clip(new_tk, params.tk[0], params.tk[-1])
            return self.interp_func(clamped_tk, params.tk, params.mean[None, ...])[0]

        # H1 shifts the already planned action sequence, zeros its tail, then
        # fits that shifted sequence back into knot values. Use Hydrax's same
        # query grid so this remains valid at any LeRobot physics timestep.
        old_tq = jnp.linspace(params.tk[0], params.tk[-1], self.ctrl_steps)
        old_u = self.interp_func(old_tq, params.tk, params.mean[None, ...])[0]
        if self.shift_steps >= self.ctrl_steps:
            shifted_u = jnp.zeros_like(old_u)
        else:
            shifted_u = jnp.roll(old_u, -self.shift_steps, axis=0)
            shifted_u = shifted_u.at[-self.shift_steps:].set(0.0)
        new_tq = jnp.linspace(state_time, state_time + self.plan_horizon, self.ctrl_steps)
        return self.interp_func(new_tk, new_tq, shifted_u[None, ...])[0]

    def optimize(self, state, params):
        """Run H1's many-pass initialization, then one diffusion pass per MPC tick."""
        new_tk = jnp.linspace(0.0, self.plan_horizon, self.num_knots) + state.time
        mean = self._warm_start_mean(state.time, params, new_tk)
        params = params.replace(tk=new_tk, mean=mean)

        def one_update(current_params):
            knots, current_params = self.sample_knots(current_params)
            knots = jnp.clip(knots, self.task.u_min, self.task.u_max)
            rng, dr_rng = jax.random.split(current_params.rng)
            rollouts = self.rollout_with_randomizations(state, new_tk, knots, dr_rng)
            current_params = current_params.replace(rng=rng)
            return self.update_params(current_params, rollouts), rollouts

        params, rollouts = one_update(params)
        extra_updates = max(self.initial_diffusion_iterations, self.iterations) - 1

        def scan_update(carry, index):
            current_params, current_rollouts = carry
            should_update = (current_params.opt_iteration < 0) | (index < self.iterations - 1)
            current_params, current_rollouts = jax.lax.cond(
                should_update,
                lambda p: one_update(p),
                lambda p: (p, current_rollouts),
                current_params,
            )
            return (current_params, current_rollouts), None

        (params, rollouts), _ = jax.lax.scan(
            scan_update, (params, rollouts), jnp.arange(extra_updates)
        )
        return params, jax.tree.map(lambda x: x, rollouts)
