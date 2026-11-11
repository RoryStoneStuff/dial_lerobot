"""Feasibility-first morphology ranking, independent of the MPC objective."""

import math


def design_objective(result, cfg):
    """Compare completion, survival, positive mechanical CoT, then MPC cost.

    CMA-ES consumes within-generation ranks; arbitrary large penalty constants
    are unnecessary. All candidates must use the same duration and seed set.
    """
    episodes = result["episodes"]
    if len(episodes) != cfg.evaluation.num_episodes:
        raise ValueError("Design evaluation must include every configured episode")
    duration = (cfg.evaluation.episode_steps * cfg.robot.timestep
                * cfg.mpc.control_decimation)
    completed, survival, costs, transport_costs, motor_margins, speeds = [], [], [], [], [], []
    for episode in episodes:
        seconds = float(episode["sim_seconds"])
        cost = float(episode["integrated_running_cost"])
        transport_cost = episode.get("positive_mechanical_cot")
        if not math.isfinite(seconds) or not math.isfinite(cost) or seconds < 0:
            raise FloatingPointError("Non-finite or invalid design evaluation")
        if transport_cost is not None:
            transport_cost = float(transport_cost)
            if not math.isfinite(transport_cost) or transport_cost < 0:
                raise FloatingPointError("Non-finite or invalid mechanical cost of transport")
        completed.append(not episode["fell"] and episode.get("failure_reason") is None
                         and episode["updates"] >= cfg.evaluation.episode_steps)
        survival.append(min(seconds / duration, 1.0))
        costs.append(cost / max(seconds, cfg.robot.timestep))
        transport_costs.append(transport_cost)
        speed = episode.get("mean_vx_m_s")
        if speed is None or not math.isfinite(float(speed)):
            raise FloatingPointError("Design evaluation has no valid mean forward speed")
        speeds.append(float(speed))
        max_torque_fraction = episode.get("max_torque_fraction")
        if cfg.robot.model_backend == "draft":
            if max_torque_fraction is None or not math.isfinite(float(max_torque_fraction)):
                raise FloatingPointError("DRAFT evaluation has no valid motor torque margin")
            motor_margins.append(1.0 - float(max_torque_fraction))
    count = len(episodes)
    mean_transport_cost = (sum(transport_costs) / count
                           if all(value is not None for value in transport_costs)
                           else None)
    metrics = {
        "completion_fraction": sum(completed) / count,
        "survival_fraction": sum(survival) / count,
        "mean_positive_mechanical_cot": mean_transport_cost,
        "mean_cost_per_second": sum(costs) / count,
        "mean_vx_m_s": sum(speeds) / count,
    }
    minimum_mean_vx = cfg.design.minimum_mean_vx
    metrics["speed_feasible"] = float(
        minimum_mean_vx is None or metrics["mean_vx_m_s"] >= minimum_mean_vx
    )
    if cfg.robot.model_backend == "draft":
        metrics["minimum_motor_margin"] = min(motor_margins)
        metrics["motor_feasible"] = float(metrics["minimum_motor_margin"] >= 0.05)
    # Larger keys are better. Fully completed episodes have equal survival;
    # tiny float32 clock differences must not outrank their walking quality.
    if all(completed):
        metrics["survival_fraction"] = 1.0
    transport_rank = (-mean_transport_cost if mean_transport_cost is not None
                      else -float("inf"))
    if cfg.robot.model_backend == "draft":
        key = (metrics["completion_fraction"], metrics["survival_fraction"],
               metrics["motor_feasible"], metrics["speed_feasible"], transport_rank,
               -metrics["mean_cost_per_second"])
    else:
        key = (metrics["completion_fraction"], metrics["survival_fraction"],
               metrics["speed_feasible"], transport_rank, -metrics["mean_cost_per_second"])
    return key, metrics


def minimization_ranks(keys):
    """Rank zero is best; identical objective keys receive identical ranks."""
    ordered = sorted(set(keys), reverse=True)
    ranks = {key: rank for rank, key in enumerate(ordered)}
    return [ranks[key] for key in keys]
