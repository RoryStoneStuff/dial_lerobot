"""Feasibility-first morphology ranking, independent of the MPC objective."""

import math


def design_objective(result, cfg):
    """Compare completion, survival, then time-averaged cost, in that order.

    CMA-ES consumes within-generation ranks; arbitrary large penalty constants
    are unnecessary. All candidates must use the same duration and seed set.
    """
    episodes = result["episodes"]
    if len(episodes) != cfg.evaluation.num_episodes:
        raise ValueError("Design evaluation must include every configured episode")
    duration = (cfg.evaluation.episode_steps * cfg.robot.timestep
                * cfg.mpc.control_decimation)
    completed, survival, costs = [], [], []
    for episode in episodes:
        seconds = float(episode["sim_seconds"])
        cost = float(episode["integrated_running_cost"])
        if not math.isfinite(seconds) or not math.isfinite(cost) or seconds < 0:
            raise FloatingPointError("Non-finite or invalid design evaluation")
        completed.append(not episode["fell"] and episode.get("failure_reason") is None
                         and episode["updates"] >= cfg.evaluation.episode_steps)
        survival.append(min(seconds / duration, 1.0))
        costs.append(cost / max(seconds, cfg.robot.timestep))
    count = len(episodes)
    metrics = {
        "completion_fraction": sum(completed) / count,
        "survival_fraction": sum(survival) / count,
        "mean_cost_per_second": sum(costs) / count,
    }
    # Larger keys are better. Fully completed episodes have equal survival;
    # tiny float32 clock differences must not outrank their walking quality.
    if all(completed):
        metrics["survival_fraction"] = 1.0
    key = (metrics["completion_fraction"], metrics["survival_fraction"],
           -metrics["mean_cost_per_second"])
    return key, metrics


def minimization_ranks(keys):
    """Rank zero is best; identical objective keys receive identical ranks."""
    ordered = sorted(set(keys), reverse=True)
    ranks = {key: rank for rank, key in enumerate(ordered)}
    return [ranks[key] for key in keys]
