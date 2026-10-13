"""CMA-ES outer design search around the SAME DIAL evaluator as the viewer."""

from datetime import datetime, timezone
from pathlib import Path
import time

import cma
import jax
import numpy as np

from dial_lerobot.assets.model_builder import nominal_design
from dial_lerobot.config import Config
from dial_lerobot.design_objective import design_objective, minimization_ranks
from dial_lerobot.evaluator import Evaluator, runtime_versions, write_json


def optimize_design(cfg: Config, output, theta0=None):
    cfg.validate()
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    d = cfg.design
    dimensions = len(d.lower_bounds)
    theta0 = nominal_design(cfg.robot, include_hip_width=dimensions == 4) if theta0 is None else np.asarray(theta0)
    if dimensions == 4 and theta0.shape == (3,):
        theta0 = np.append(theta0, nominal_design(cfg.robot, include_hip_width=True)[3])
    if theta0.shape != (dimensions,) or not np.isfinite(theta0).all():
        raise ValueError("Initial design must match the configured bounds and contain finite values")
    if np.any(theta0 < d.lower_bounds) or np.any(theta0 > d.upper_bounds):
        raise ValueError("Initial design is outside CMA bounds")
    fixed_variables = {}
    value = d.fixed_density_proxy
    if value is None and cfg.robot.preserve_link_masses:
        value = float(nominal_design(cfg.robot)[2])
    if value is not None:
        if not np.isfinite(value) or not d.lower_bounds[2] <= value <= d.upper_bounds[2]:
            raise ValueError("Fixed density proxy must be finite and inside its bounds")
        theta0 = np.asarray(theta0, dtype=float).copy()
        theta0[2] = value
        fixed_variables[2] = value
    es = cma.CMAEvolutionStrategy(theta0, d.sigma0, {
        "fixed_variables": fixed_variables,
        "bounds": [list(d.lower_bounds), list(d.upper_bounds)], "maxiter": d.maxiter,
        "popsize": d.popsize, "seed": d.seed, "verbose": -1, "verb_log": 0,
        # Ranks restart at zero each generation, so cross-generation scalar
        # fitness stagnation checks have no meaning. Keep geometric stops.
        "tolfun": 0, "tolfunhist": 0, "tolfunrel": 0, "tolstagnation": 0,
        # Lengths (metres) and the mass proxy have different units. Explore
        # the same fraction of each allowed range rather than equal numbers.
        "CMA_stds": ((np.asarray(d.upper_bounds) - np.asarray(d.lower_bounds))
                     / (d.upper_bounds[0] - d.lower_bounds[0])).tolist(),
    })
    log = {"config": cfg.to_dict(), "runtime": runtime_versions(), "controller": "DIAL",
           "started_utc": datetime.now(timezone.utc).isoformat(), "theta0": theta0.tolist(),
           "generations": [], "best_fitness": None, "best_theta": None}
    best_key = None
    log["selection_objective"] = ["completion_fraction", "survival_fraction", "negative_mean_cost_per_second"]
    log["evaluation_seed"] = cfg.evaluation.seed
    log["fixed_design_parameters"] = {"density_proxy": value} if fixed_variables else {}
    write_json(folder / "optimization.json", log)
    start = time.perf_counter()
    # Keep the starting morphology as an incumbent: a search must not replace
    # a good nominal controller with the least bad failed candidate.
    print("Evaluating starting morphology on the common episode seeds...", flush=True)
    baseline = None
    shared_kernel = None
    try:
        baseline = Evaluator(cfg, theta0, reuse_compilation=True)
        shared_kernel = baseline.shared_kernel
        warmup_seconds = baseline.warmup()
        print(f"  Baseline compilation/warmup: {warmup_seconds:.2f}s", flush=True)
        baseline_folder = folder / "baseline"
        result = baseline.evaluate(seed=cfg.evaluation.seed, output=baseline_folder)
        best_key, metrics = design_objective(result, cfg)
        log["baseline"] = {"objective": metrics, "fitness": result["fitness"],
                           "warmup_seconds": warmup_seconds, "episodes": result["episodes"]}
        log["best_fitness"] = result["fitness"]
        log["best_theta"] = theta0.tolist()
        log["best_objective"] = metrics
        write_json(folder / "best_design.json", {
            "theta": theta0.tolist(), "fitness": result["fitness"], "config": cfg.to_dict(),
            "model": result["model"], "generation": -1, "seed": cfg.evaluation.seed,
            "objective": metrics, "trajectory_folder": str(baseline_folder.resolve()),
        })
    except (ValueError, FloatingPointError) as exc:
        log["baseline"] = {"error": str(exc)}
        best_key = None
    finally:
        del baseline
        if shared_kernel is None:
            jax.clear_caches()
    write_json(folder / "optimization.json", log)
    while not es.stop():
        generation = len(log["generations"])
        candidates = es.ask()
        keys, evaluations = [], []
        for i, theta in enumerate(candidates):
            seed = cfg.evaluation.seed  # Common random numbers across every design.
            print(f"Generation {generation + 1}/{d.maxiter}, candidate {i + 1}/{d.popsize}: {theta}", flush=True)
            evaluator = None
            try:
                evaluator = Evaluator(cfg, theta, shared_kernel=shared_kernel, reuse_compilation=True)
                warmup_seconds = evaluator.warmup()
                print(f"  Compilation/warmup: {warmup_seconds:.2f}s", flush=True)
                candidate_folder = folder / f"generation_{generation:03d}" / f"candidate_{i:03d}"
                result = evaluator.evaluate(seed=seed, output=candidate_folder)
                score = result["fitness"]
                if not np.isfinite(score):
                    raise FloatingPointError("Non-finite candidate fitness")
                key, metrics = design_objective(result, cfg)
                evaluations.append({"theta": np.asarray(theta).tolist(), "seed": seed,
                                    "fitness": score, "objective": metrics,
                                    "warmup_seconds": warmup_seconds,
                                    "trajectory_folder": str(candidate_folder.resolve()),
                                    "episodes": result["episodes"]})
                if best_key is None or key > best_key:
                    best_key = key
                    log["best_fitness"] = score
                    log["best_theta"] = np.asarray(theta).tolist()
                    log["best_objective"] = metrics
                    write_json(folder / "best_design.json", {
                        "theta": log["best_theta"], "fitness": score, "config": cfg.to_dict(),
                        "model": result["model"], "generation": generation, "seed": seed,
                        "objective": metrics, "trajectory_folder": str(candidate_folder.resolve()),
                    })
            except (ValueError, FloatingPointError) as exc:
                # Invalid geometry or divergent dynamics lose; dependency and
                # programming errors intentionally propagate instead of faking scores.
                score = -1e12
                key = (-1.0, -1.0, -float("inf"))
                evaluations.append({"theta": np.asarray(theta).tolist(), "seed": seed,
                                    "fitness": score, "error": str(exc)})
            finally:
                del evaluator
                # Retain the shared executable across the entire design search.
                if shared_kernel is None:
                    jax.clear_caches()
            keys.append(key)
            print(f"  legacy fitness {score:.6f}; best objective {log.get('best_objective')}", flush=True)
            write_json(folder / "current_generation.json", {"generation": generation,
                                                           "evaluations": evaluations})
        ranks = minimization_ranks(keys)
        for evaluation, rank in zip(evaluations, ranks):
            evaluation["selection_rank"] = rank
        es.tell(candidates, ranks)
        log["generations"].append({"generation": generation, "evaluations": evaluations,
                                    "best_so_far": log["best_fitness"],
                                    "best_theta_so_far": log["best_theta"],
                                    "best_objective_so_far": log.get("best_objective")})
        log["wall_seconds"] = time.perf_counter() - start
        write_json(folder / "optimization.json", log)
    log["stop"] = {str(k): str(v) for k, v in es.stop().items()}
    write_json(folder / "optimization.json", log)
    if log["best_theta"] is None:
        raise RuntimeError(f"Every design failed; inspect {folder / 'optimization.json'}")
    return log


def design_heatmap(cfg: Config, output, points=5, density=2.0):
    """Optional thigh/shin sweep at fixed density, with the co-design fitness."""
    if points < 2:
        raise ValueError("points must be at least 2")
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    thighs = np.linspace(cfg.design.lower_bounds[0], cfg.design.upper_bounds[0], points)
    shins = np.linspace(cfg.design.lower_bounds[1], cfg.design.upper_bounds[1], points)
    rows = []
    shared_kernel = None
    for thigh in thighs:
        for shin in shins:
            theta = [float(thigh), float(shin), density]
            print(f"Heatmap {len(rows) + 1}/{points**2}: {theta}", flush=True)
            evaluator = Evaluator(cfg, theta, shared_kernel=shared_kernel, reuse_compilation=True)
            shared_kernel = evaluator.shared_kernel
            result = evaluator.evaluate(seed=cfg.evaluation.seed)
            rows.append({"theta": theta, "fitness": result["fitness"]})
            del evaluator
            write_json(folder / "heatmap.json", {"config": cfg.to_dict(), "rows": rows})
    return rows
