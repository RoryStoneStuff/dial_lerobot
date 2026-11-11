"""Gaussian-process Bayesian optimization for morphology design."""

from datetime import datetime, timezone
from pathlib import Path
import json
import math
import time

import jax
import numpy as np

from dial_lerobot.assets.draft_adapter import motor_feedback_theta
from dial_lerobot.assets.model_builder import nominal_design
from dial_lerobot.config import Config
from dial_lerobot.design_objective import design_objective, minimization_ranks
from dial_lerobot.evaluator import Evaluator, runtime_versions, write_json


def _initial_theta(cfg, theta0):
    dimensions = len(cfg.design.lower_bounds)
    motor_design = dimensions == 10
    theta = (nominal_design(cfg.robot, include_hip_width=dimensions >= 4,
                            include_motor_design=motor_design)
             if theta0 is None else np.asarray(theta0, dtype=float))
    if dimensions == 4 and theta.shape == (3,):
        theta = np.append(theta, nominal_design(cfg.robot, include_hip_width=True)[3])
    if dimensions == 10 and theta.shape == (4,):
        theta = np.concatenate((theta, np.ones(6)))
    if theta.shape != (dimensions,) or not np.isfinite(theta).all():
        raise ValueError("Initial design must match the configured bounds and contain finite values")
    if np.any(theta < cfg.design.lower_bounds) or np.any(theta > cfg.design.upper_bounds):
        raise ValueError("Initial design is outside Bayesian optimization bounds")
    return theta


def _utility(metrics, cfg):
    cot = metrics.get("mean_positive_mechanical_cot")
    cot = float(cot) if cot is not None and math.isfinite(float(cot)) else math.inf
    cost = float(metrics.get("mean_cost_per_second", math.inf))
    cot_score = 0.0 if not math.isfinite(cot) else 1.0 / (1.0 + max(cot, 0.0))
    cost_score = 0.0 if not math.isfinite(cost) else 1.0 / (1.0 + max(cost, 0.0))
    utility = 1_000_000.0 * float(metrics["completion_fraction"])
    utility += 10_000.0 * float(metrics["survival_fraction"])
    if cfg.robot.model_backend == "draft":
        utility += 100.0 * float(metrics["motor_feasible"])
    utility += 10.0 * float(metrics["speed_feasible"])
    return utility + cot_score + 0.01 * cost_score


def _kernel(left, right, length_scale=0.18):
    delta = left[:, None, :] - right[None, :, :]
    return np.exp(-0.5 * np.sum(delta * delta, axis=-1) / length_scale**2)


def _predict(observed_x, observed_y, query_x):
    if not observed_x:
        return np.zeros(len(query_x)), np.ones(len(query_x))
    x = np.asarray(observed_x, dtype=float)
    y = np.asarray(observed_y, dtype=float)
    covariance = _kernel(x, x) + 1e-6 * np.eye(len(x))
    try:
        alpha = np.linalg.solve(covariance, y)
        cross = _kernel(np.asarray(query_x), x)
        mean = cross @ alpha
        variance = 1.0 - np.sum(cross * np.linalg.solve(covariance, cross.T).T, axis=1)
        return mean, np.sqrt(np.maximum(variance, 1e-9))
    except np.linalg.LinAlgError:
        return np.zeros(len(query_x)), np.ones(len(query_x))


def _next_point(observed_x, observed_y, lower, upper, fixed, rng):
    dimensions = len(lower)
    candidates = rng.uniform(lower, upper, size=(4096, dimensions))
    if fixed:
        for index, value in fixed.items():
            candidates[:, index] = value
    if observed_x:
        incumbent = np.asarray(observed_x[int(np.argmax(observed_y))])
        local = incumbent + rng.normal(0.0, 0.20, size=(512, dimensions)) * (upper - lower)
        candidates[: len(local)] = np.clip(local, lower, upper)
        if fixed:
            for index, value in fixed.items():
                candidates[: len(local), index] = value
    mean, std = _predict(observed_x, observed_y, (candidates - lower) / (upper - lower))
    acquisition = mean + 2.0 * std
    return candidates[int(np.argmax(acquisition))]


def _metrics_key(metrics, cfg):
    cot = metrics.get("mean_positive_mechanical_cot")
    cot_rank = -float(cot) if cot is not None else -float("inf")
    prefix = (float(metrics["completion_fraction"]), float(metrics["survival_fraction"]))
    if cfg.robot.model_backend == "draft":
        return prefix + (float(metrics["motor_feasible"]), float(metrics["speed_feasible"]),
                         cot_rank, -float(metrics["mean_cost_per_second"]))
    return prefix + (float(metrics["speed_feasible"]), cot_rank,
                     -float(metrics["mean_cost_per_second"]))


def _failed_key(cfg):
    return tuple([-1.0, -1.0] + ([-float("inf")] * (4 if cfg.robot.model_backend == "draft" else 3)))


def _evaluation_key(evaluation, cfg):
    if evaluation.get("error"):
        return _failed_key(cfg)
    return _metrics_key(evaluation["objective"], cfg)


def optimize_design_bayesian(cfg: Config, output, theta0=None, gait_inner_candidates=0,
                             gait_parameters=("duty", "cadence", "gait_start_delay"), *, resume=False):
    from dial_lerobot.gait_bayesian_optimize import optimize_gait_phase, with_gait_parameters

    cfg.validate()
    if gait_inner_candidates < 0:
        raise ValueError("gait_inner_candidates must be nonnegative")
    if gait_inner_candidates and cfg.robot.draft_auto_resize_motors:
        raise ValueError("Gait inner optimization requires draft_auto_resize_motors=false")
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    optimization_path = folder / "optimization.json"
    checkpoint_path = folder / "checkpoint.json"
    current_generation_path = folder / "current_generation.json"
    lower = np.asarray(cfg.design.lower_bounds, dtype=float)
    upper = np.asarray(cfg.design.upper_bounds, dtype=float)
    theta0 = _initial_theta(cfg, theta0)
    fixed = {}
    density = cfg.design.fixed_density_proxy
    if density is None and cfg.robot.preserve_link_masses:
        density = float(nominal_design(cfg.robot)[2])
    if density is not None:
        theta0 = theta0.copy()
        theta0[2] = density
        fixed[2] = density
    if resume:
        if not optimization_path.exists():
            raise ValueError(f"Cannot resume: no optimization.json in {folder}")
        previous_log = json.loads(optimization_path.read_text())
        if json.dumps(previous_log.get("config"), sort_keys=True) != json.dumps(cfg.to_dict(), sort_keys=True):
            raise ValueError("Cannot resume: requested config differs from the saved run config")
        if (previous_log.get("gait_inner_candidates", 0) != int(gait_inner_candidates)
                or previous_log.get("gait_inner_parameters", []) != list(gait_parameters)):
            raise ValueError("Cannot resume: gait inner-loop settings differ from the saved run")
        if previous_log.get("stop"):
            raise ValueError("Cannot resume an optimization already marked complete")
        log = previous_log
    else:
        if optimization_path.exists():
            raise ValueError(f"Results already exist in {folder}; use resume=True to continue")
        log = {"config": cfg.to_dict(),
               "runtime": runtime_versions(include_devices=not bool(gait_inner_candidates)),
               "controller": "DIAL",
               "optimizer": "bayesian_gp_ucb", "started_utc": datetime.now(timezone.utc).isoformat(),
               "gait_inner_candidates": int(gait_inner_candidates),
               "gait_inner_parameters": list(gait_parameters),
               "theta0": theta0.tolist(), "generations": [], "best_fitness": None,
               "best_theta": None, "best_objective": None,
               "selection_objective": ["completion_fraction", "survival_fraction"],
               "evaluation_seed": cfg.evaluation.seed, "fixed_design_parameters": {"density_proxy": density},
               "bayesian_scalarization": "lexicographic-compatible: completion, survival, motor feasibility, speed feasibility, lower CoT, lower running cost",
               "bayesian_hyperparameters": {"kernel_length_scale": 0.18, "ucb_exploration": 2.0,
                                             "global_candidates": 4096, "local_candidates": 512,
                                             "local_spread": 0.20}}
        if cfg.robot.model_backend == "draft":
            log["selection_objective"].append("motor_feasible")
        log["selection_objective"] += ["negative_mean_positive_mechanical_cot", "negative_mean_cost_per_second"]
        write_json(optimization_path, log)

    shared_kernel = None
    initial_count = max(8, 2 * len(lower))
    rng = np.random.default_rng(cfg.design.seed)
    accumulated_wall = float(log.get("wall_seconds", 0.0))
    generation = 0
    candidate_index = 0
    current_evaluations = []
    observations_x, observations_y = [], []
    best_key = (_metrics_key(log["best_objective"], cfg)
                if log.get("best_objective") is not None else None)

    def add_observation(evaluation):
        observations_x.append((np.asarray(evaluation["theta"], dtype=float) - lower) / (upper - lower))
        observations_y.append(float(evaluation.get("bayesian_utility", -1e6)))

    def replay_rng_for_candidate():
        if len(observations_x) < initial_count:
            rng.uniform(lower, upper)
        else:
            normalized_fixed = {
                index: (value - lower[index]) / (upper[index] - lower[index])
                for index, value in fixed.items()
            }
            _next_point(observations_x, observations_y, np.zeros(len(lower)),
                        np.ones(len(lower)), normalized_fixed, rng)

    if resume and checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text())
        log = checkpoint["log"]
        rng.bit_generator.state = checkpoint["rng_state"]
        observations_x = checkpoint["observations_x"]
        observations_y = checkpoint["observations_y"]
        generation = int(checkpoint["generation"])
        candidate_index = int(checkpoint["candidate"])
        current_evaluations = checkpoint["current_evaluations"]
        accumulated_wall = float(checkpoint.get("wall_seconds", accumulated_wall))
        best_key = (_metrics_key(log["best_objective"], cfg)
                    if log.get("best_objective") is not None else None)
    elif resume:
        for baseline_entry in (log.get("baseline", {}),):
            if baseline_entry.get("objective") is not None:
                baseline_eval = {"theta": theta0.tolist(),
                                 "bayesian_utility": _utility(baseline_entry["objective"], cfg)}
                add_observation(baseline_eval)
        for saved_generation in log.get("generations", []):
            for saved_evaluation in saved_generation.get("evaluations", []):
                replay_rng_for_candidate()
                add_observation(saved_evaluation)
        if current_generation_path.exists():
            partial = json.loads(current_generation_path.read_text())
            if int(partial.get("generation", -1)) == len(log.get("generations", [])):
                current_evaluations = partial.get("evaluations", [])
                for saved_evaluation in current_evaluations:
                    replay_rng_for_candidate()
                    add_observation(saved_evaluation)
                candidate_index = len(current_evaluations)
        generation = len(log.get("generations", []))

    start = time.perf_counter()

    def save_checkpoint():
        log["wall_seconds"] = accumulated_wall + time.perf_counter() - start
        write_json(checkpoint_path, {
            "log": log,
            "generation": generation,
            "candidate": candidate_index,
            "current_evaluations": current_evaluations,
            "observations_x": [np.asarray(point, dtype=float).tolist() for point in observations_x],
            "observations_y": observations_y,
            "rng_state": rng.bit_generator.state,
            "wall_seconds": log["wall_seconds"],
        })
        write_json(current_generation_path, {"generation": generation,
                                               "evaluations": current_evaluations})
        write_json(optimization_path, log)
    initial_count = max(8, 2 * len(lower))
    rng = np.random.default_rng(cfg.design.seed)

    def evaluate(theta, candidate_folder):
        nonlocal shared_kernel
        candidate_folder = Path(candidate_folder)
        cached_path = candidate_folder / "evaluation.json"
        if cached_path.exists():
            cached = json.loads(cached_path.read_text())
            if "episodes" in cached and "model" in cached:
                return cached, np.asarray(cached.get("model", {}).get("theta", theta), dtype=float), 0.0
        if candidate_folder.exists():
            attempt = 1
            while candidate_folder.with_name(f"{candidate_folder.name}_retry_{attempt:03d}").exists():
                attempt += 1
            candidate_folder = candidate_folder.with_name(f"{candidate_folder.name}_retry_{attempt:03d}")
        if gait_inner_candidates:
            gait_log = optimize_gait_phase(
                cfg, Path(candidate_folder) / "gait_inner", theta, gait_inner_candidates,
                parameters=gait_parameters, persistent_worker=True,
            )
            evaluation_cfg = with_gait_parameters(cfg, gait_log["best_gait"])
            selected_folder = Path(gait_log["best_trajectory_folder"])
            result = json.loads((selected_folder / "evaluation.json").read_text())
            result["gait_inner_loop"] = gait_log
            result["evaluation_config"] = evaluation_cfg.to_dict()
            result["trajectory_folder"] = str(selected_folder)
            write_json(Path(candidate_folder) / "evaluation.json", result)
            warmup = sum(entry.get("warmup_seconds", 0.0) for entry in gait_log["evaluations"])
            return result, np.asarray(theta, dtype=float), warmup
        evaluator = Evaluator(
            cfg, theta, shared_kernel=shared_kernel, reuse_compilation=True,
        )
        warmup = evaluator.warmup()
        shared_kernel = evaluator.shared_kernel
        result = evaluator.evaluate(seed=cfg.evaluation.seed, output=candidate_folder)
        evaluated = np.asarray(theta, dtype=float)
        if cfg.robot.model_backend == "draft" and len(theta) == 10 and cfg.robot.draft_auto_resize_motors:
            evaluated = motor_feedback_theta(theta, result)
            if not np.allclose(evaluated, theta):
                del evaluator
                evaluator = Evaluator(cfg, evaluated, shared_kernel=shared_kernel, reuse_compilation=True)
                warmup += evaluator.warmup()
                result = evaluator.evaluate(seed=cfg.evaluation.seed, output=candidate_folder)
            result["motor_feedback"] = {"proposed_theta": np.asarray(theta).tolist(),
                                         "evaluated_theta": np.asarray(evaluated).tolist(),
                                         "safety_margin": 1.20}
            write_json(Path(candidate_folder) / "evaluation.json", result)
        del evaluator
        return result, evaluated, warmup

    if not log.get("baseline"):
        baseline_folder = folder / "baseline"
        try:
            result, evaluated, warmup = evaluate(theta0, baseline_folder)
            key, metrics = design_objective(result, cfg)
            log["baseline"] = {"objective": metrics, "fitness": result["fitness"],
                                "warmup_seconds": warmup, "episodes": result["episodes"],
                                "best_gait": result.get("gait_inner_loop", {}).get("best_gait")}
            log["best_fitness"] = result["fitness"]
            log["best_theta"] = evaluated.tolist()
            log["best_objective"] = metrics
            best_key = key
            observations_x.append((theta0 - lower) / (upper - lower))
            observations_y.append(_utility(metrics, cfg))
            write_json(folder / "best_design.json", {"theta": evaluated.tolist(), "fitness": result["fitness"],
                         "config": result.get("evaluation_config", cfg.to_dict()), "model": result["model"], "generation": -1,
                         "seed": cfg.evaluation.seed, "objective": metrics,
                         "trajectory_folder": result.get("trajectory_folder", str(baseline_folder.resolve()))})
        except Exception as exc:
            log["baseline"] = {"error": str(exc)}
        save_checkpoint()

    while generation < cfg.design.maxiter:
        evaluations = current_evaluations
        keys = [_evaluation_key(evaluation, cfg) for evaluation in evaluations]
        if candidate_index >= cfg.design.popsize:
            ranks = minimization_ranks(keys)
            for evaluation, rank in zip(evaluations, ranks):
                evaluation["selection_rank"] = rank
            log["generations"].append({"generation": generation, "evaluations": evaluations,
                                       "best_so_far": log.get("best_fitness"),
                                       "best_theta_so_far": log.get("best_theta"),
                                       "best_objective_so_far": log.get("best_objective")})
            generation += 1
            candidate_index = 0
            current_evaluations = []
            save_checkpoint()
            continue
        candidate = candidate_index
        if len(observations_x) < initial_count:
            theta = rng.uniform(lower, upper)
            for index, value in fixed.items():
                theta[index] = value
        else:
            normalized_fixed = {
                index: (value - lower[index]) / (upper[index] - lower[index])
                for index, value in fixed.items()
            }
            theta = lower + (upper - lower) * _next_point(
                observations_x, observations_y, np.zeros(len(lower)),
                np.ones(len(lower)), normalized_fixed, rng)
            for index, value in fixed.items():
                theta[index] = value
        candidate_folder = folder / f"generation_{generation:03d}" / f"candidate_{candidate:03d}"
        print(f"Generation {generation + 1}/{cfg.design.maxiter}, candidate {candidate + 1}/{cfg.design.popsize}: {theta}", flush=True)
        try:
            result, evaluated, warmup = evaluate(theta, candidate_folder)
            key, metrics = design_objective(result, cfg)
            utility = _utility(metrics, cfg)
            observations_x.append((theta - lower) / (upper - lower))
            observations_y.append(utility)
            evaluation = {"theta": theta.tolist(), "evaluated_theta": evaluated.tolist(),
                          "seed": cfg.evaluation.seed, "fitness": result["fitness"],
                          "bayesian_utility": utility, "objective": metrics,
                          "warmup_seconds": warmup,
                          "best_gait": result.get("gait_inner_loop", {}).get("best_gait"),
                          "trajectory_folder": result.get("trajectory_folder", str(candidate_folder.resolve())),
                          "episodes": result["episodes"]}
            if best_key is None or key > best_key:
                best_key = key
                log["best_fitness"] = result["fitness"]
                log["best_theta"] = evaluated.tolist()
                log["best_objective"] = metrics
                write_json(folder / "best_design.json", {"theta": evaluated.tolist(),
                             "fitness": result["fitness"], "config": result.get("evaluation_config", cfg.to_dict()),
                             "model": result["model"], "generation": generation,
                             "seed": cfg.evaluation.seed, "objective": metrics,
                             "trajectory_folder": result.get("trajectory_folder", str(candidate_folder.resolve()))})
        except Exception as exc:
            evaluation = {"theta": theta.tolist(), "seed": cfg.evaluation.seed,
                          "fitness": -1e12, "error": str(exc), "bayesian_utility": -1e6}
            key = _failed_key(cfg)
            observations_x.append((theta - lower) / (upper - lower))
            observations_y.append(-1e6)
        current_evaluations.append(evaluation)
        candidate_index += 1
        save_checkpoint()

    log["wall_seconds"] = accumulated_wall + time.perf_counter() - start
    log["stop"] = {"maxiter": str(cfg.design.maxiter), "optimizer": "bayesian_gp_ucb"}
    save_checkpoint()
    return log
