"""Compare DRAFT motor torque/speed scales on one fixed nominal morphology."""

import argparse
from pathlib import Path

import numpy as np

from dial_lerobot.config import load_config
from dial_lerobot.design_objective import design_objective
from dial_lerobot.evaluator import Evaluator, write_json


CASES = {
    "torque_low": (0.75, 0.75, 0.75, 1.0, 1.0, 1.0),
    "nominal": (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    "torque_high": (1.25, 1.25, 1.25, 1.0, 1.0, 1.0),
    "speed_low": (1.0, 1.0, 1.0, 0.8, 0.8, 0.8),
    "speed_high": (1.0, 1.0, 1.0, 1.2, 1.2, 1.2),
}
NOMINAL_GEOMETRY = (0.34136558643337206, 0.2607679717488327, 2.0, 0.266319)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if cfg.robot.model_backend != "draft":
        raise ValueError("The motor sweep requires the DRAFT backend")
    if len(cfg.design.lower_bounds) != 10:
        raise ValueError("The motor sweep requires ten design variables")
    args.output.mkdir(parents=True, exist_ok=True)
    shared_kernel = None
    rows = []
    for name, motor_scales in CASES.items():
        theta = np.asarray(NOMINAL_GEOMETRY + motor_scales, dtype=float)
        folder = args.output / name
        evaluator = Evaluator(cfg, theta, shared_kernel=shared_kernel, reuse_compilation=True)
        warmup_seconds = evaluator.warmup()
        result = evaluator.evaluate(seed=cfg.evaluation.seed, output=folder)
        objective, metrics = design_objective(result, cfg)
        rank_key = [
            None if not np.isfinite(value) else float(value)
            for value in objective
        ]
        rows.append({
            "case": name,
            "theta": theta.tolist(),
            "warmup_seconds": warmup_seconds,
            "objective": metrics,
            "rank_key": rank_key,
            "model": result["model"],
            "episodes": result["episodes"],
            "trajectory_folder": str(folder.resolve()),
        })
        shared_kernel = evaluator.shared_kernel
        del evaluator
        write_json(args.output / "motor_sweep.json", {"config": cfg.to_dict(), "cases": rows})
    print(args.output / "motor_sweep.json")


if __name__ == "__main__":
    main()
