"""CMA-ES outer design search around the SAME DIAL evaluator as the viewer."""

from datetime import datetime, timezone
from pathlib import Path
import time

import cma
import jax
import numpy as np

from dial_lerobot.assets.model_builder import nominal_design
from dial_lerobot.config import Config
from dial_lerobot.evaluator import Evaluator, runtime_versions, write_json


def optimize_design(cfg: Config, output, theta0=None):
    cfg.validate()
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    theta0 = nominal_design(cfg.robot) if theta0 is None else np.asarray(theta0)
    d = cfg.design
    if np.any(theta0 < d.lower_bounds) or np.any(theta0 > d.upper_bounds):
        raise ValueError("Initial design is outside CMA bounds")
    es = cma.CMAEvolutionStrategy(theta0, d.sigma0, {
        "bounds": [list(d.lower_bounds), list(d.upper_bounds)], "maxiter": d.maxiter,
        "popsize": d.popsize, "seed": d.seed, "verbose": -1, "verb_log": 0,
    })
    log = {"config": cfg.to_dict(), "runtime": runtime_versions(), "controller": "DIAL",
           "started_utc": datetime.now(timezone.utc).isoformat(), "theta0": theta0.tolist(),
           "generations": [], "best_fitness": None, "best_theta": None}
    write_json(folder / "optimization.json", log)
    best = -np.inf
    start = time.perf_counter()
    while not es.stop():
        generation = len(log["generations"])
        candidates = es.ask()
        scores, evaluations = [], []
        for i, theta in enumerate(candidates):
            seed = generation * 100 + i  # Same candidate seed schedule as source.
            print(f"Generation {generation + 1}/{d.maxiter}, candidate {i + 1}/{d.popsize}: {theta}", flush=True)
            evaluator = None
            try:
                evaluator = Evaluator(cfg, theta)
                result = evaluator.evaluate(seed=seed)
                score = result["fitness"]
                if not np.isfinite(score):
                    raise FloatingPointError("Non-finite candidate fitness")
                evaluations.append({"theta": np.asarray(theta).tolist(), "seed": seed,
                                    "fitness": score, "episodes": result["episodes"]})
                if score > best:
                    best = score
                    log["best_fitness"] = score
                    log["best_theta"] = np.asarray(theta).tolist()
                    write_json(folder / "best_design.json", {
                        "theta": log["best_theta"], "fitness": score, "config": cfg.to_dict(),
                        "model": result["model"], "generation": generation, "seed": seed,
                    })
            except (ValueError, FloatingPointError) as exc:
                # Invalid geometry or divergent dynamics lose; dependency and
                # programming errors intentionally propagate instead of faking scores.
                score = -1e12
                evaluations.append({"theta": np.asarray(theta).tolist(), "seed": seed,
                                    "fitness": score, "error": str(exc)})
            finally:
                del evaluator
                # Geometry changes otherwise retain a large executable per design.
                jax.clear_caches()
            scores.append(score)
            print(f"  fitness {score:.6f}; best {best:.6f}", flush=True)
            write_json(folder / "current_generation.json", {"generation": generation,
                                                           "evaluations": evaluations})
        es.tell(candidates, [-score for score in scores])
        log["generations"].append({"generation": generation, "evaluations": evaluations,
                                    "best_so_far": log["best_fitness"]})
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
    for thigh in thighs:
        for shin in shins:
            theta = [float(thigh), float(shin), density]
            print(f"Heatmap {len(rows) + 1}/{points**2}: {theta}", flush=True)
            evaluator = Evaluator(cfg, theta)
            result = evaluator.evaluate(seed=cfg.evaluation.seed)
            rows.append({"theta": theta, "fitness": result["fitness"]})
            del evaluator
            jax.clear_caches()
            write_json(folder / "heatmap.json", {"config": cfg.to_dict(), "rows": rows})
    return rows
