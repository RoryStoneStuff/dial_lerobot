"""HydraX DIAL with the official normalized-action biped interface."""

from functools import partial
import jax
import jax.numpy as jnp
from hydrax.alg_base import Trajectory
from hydrax.algs import DIAL
from mujoco import mjx

class LeRobotDIAL(DIAL):
    """DIAL whose normalized actions are converted to PD torques by the task."""
    @partial(jax.vmap, in_axes=(None, None, None, 0, 0))
    def eval_rollouts(self, model, state, controls, knots):
        def _scan_fn(x, u):
            x = x.replace(ctrl=self.task.action_to_control(x, u))
            x = mjx.step(model, x)
            cost = self.dt * self.task.running_cost(x, u)
            sites = self.task.get_trace_sites(x)
            return x, (x, cost, sites)
        final_state, (states, costs, trace_sites) = jax.lax.scan(_scan_fn, state, controls)
        final_cost = self.task.terminal_cost(final_state)
        final_trace_sites = self.task.get_trace_sites(final_state)
        costs = jnp.append(costs, final_cost)
        trace_sites = jnp.append(trace_sites, final_trace_sites[None], axis=0)
        return states, Trajectory(controls=controls, knots=knots, costs=costs, trace_sites=trace_sites)
