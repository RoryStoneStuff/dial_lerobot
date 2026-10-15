"""Run a resumable multi-morphology, multi-command DIAL benchmark."""

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path

import jax
import numpy as np

from dial_lerobot.assets.model_builder import nominal_design
from dial_lerobot.config import load_config
from dial_lerobot.design_objective import design_objective
from dial_lerobot.evaluator import Evaluator, runtime_versions, write_json


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--winner", type=Path)
    parser.add_argument("--morphology-plan", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--speeds", type=float, nargs="+", default=(0.5, 0.7, 0.85))
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=1045)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--solver-iterations", type=int)
    parser.add_argument("--ls-iterations", type=int)
    parser.add_argument("--expanded-morphologies", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_morphologies(cfg, winner_path, expanded=False):
    if winner_path is None:
        raise ValueError("--winner is required unless --morphology-plan is provided")
    nominal = nominal_design(cfg.robot, include_hip_width=True)
    winner = np.asarray(json.loads(winner_path.read_text())["theta"], dtype=float)
    if winner.shape != (4,):
        raise ValueError("Winner must contain thigh, shin, density, and hip-width values")
    density = cfg.design.fixed_density_proxy
    if density is None:
        density = float(nominal[2])
    lower = cfg.design.lower_bounds
    upper = cfg.design.upper_bounds
    morphologies = {
        "nominal": np.asarray(nominal, dtype=float),
        "winner": winner,
        "short_legs": np.asarray((lower[0], lower[1], density, nominal[3])),
        "long_legs": np.asarray((upper[0], upper[1], density, nominal[3])),
        "narrow_hips": np.asarray((nominal[0], nominal[1], density, lower[3])),
        "wide_hips": np.asarray((nominal[0], nominal[1], density, upper[3])),
    }
    if expanded:
        morphologies.update({
            "short_thigh": np.asarray((lower[0], nominal[1], density, nominal[3])),
            "long_thigh": np.asarray((upper[0], nominal[1], density, nominal[3])),
            "short_shin": np.asarray((nominal[0], lower[1], density, nominal[3])),
            "long_shin": np.asarray((nominal[0], upper[1], density, nominal[3])),
            "extra_wide_hips": np.asarray(
                (nominal[0], nominal[1], density, 1.3 * nominal[3])
            ),
        })
    return morphologies


def load_morphology_plan(path):
    plan = json.loads(path.read_text())
    entries = plan.get("morphologies")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Morphology plan requires a nonempty 'morphologies' list")
    morphologies = {}
    for entry in entries:
        name = entry.get("name")
        theta = np.asarray(entry.get("theta"), dtype=float)
        if not name or name in morphologies:
            raise ValueError(f"Morphology plan name is missing or duplicated: {name!r}")
        if theta.shape != (4,) or not np.isfinite(theta).all() or np.any(theta <= 0):
            raise ValueError(f"Morphology {name!r} requires four finite positive values")
        morphologies[name] = theta
    return morphologies


def condition_name(speed, morphology):
    speed_label = f"{speed:.2f}".replace(".", "p")
    return f"vx_{speed_label}/{morphology}"


def main():
    args = parse_args()
    if args.episodes < 1 or args.seconds <= 0 or not args.speeds:
        raise ValueError("Episodes, duration, and speed list must be positive")
    base_cfg = load_config(args.config)
    base_cfg = replace(
        base_cfg,
        robot=replace(
            base_cfg.robot,
            solver_iterations=(args.solver_iterations
                               if args.solver_iterations is not None
                               else base_cfg.robot.solver_iterations),
            ls_iterations=(args.ls_iterations
                           if args.ls_iterations is not None
                           else base_cfg.robot.ls_iterations),
        ),
        mpc=replace(
            base_cfg.mpc,
            num_samples=args.samples if args.samples is not None else base_cfg.mpc.num_samples,
        ),
    )
    update_seconds = base_cfg.robot.timestep * base_cfg.mpc.control_decimation
    episode_steps = round(args.seconds / update_seconds)
    if not np.isclose(episode_steps * update_seconds, args.seconds):
        raise ValueError("Duration must be an integer number of MPC updates")
    base_cfg = replace(
        base_cfg,
        evaluation=replace(
            base_cfg.evaluation,
            episode_steps=episode_steps,
            num_episodes=args.episodes,
            seed=args.seed,
        ),
    ).validate()
    if args.morphology_plan is not None:
        if args.expanded_morphologies:
            raise ValueError("--expanded-morphologies cannot be combined with --morphology-plan")
        morphologies = load_morphology_plan(args.morphology_plan)
    else:
        morphologies = load_morphologies(
            base_cfg, args.winner, expanded=args.expanded_morphologies
        )
    matrix = [
        {
            "name": condition_name(speed, name),
            "speed_m_s": speed,
            "morphology": name,
            "theta": theta.tolist(),
        }
        for speed in args.speeds
        for name, theta in morphologies.items()
    ]
    print(json.dumps({"conditions": len(matrix), "episodes": len(matrix) * args.episodes,
                      "samples": base_cfg.mpc.num_samples,
                      "solver_iterations": base_cfg.robot.solver_iterations,
                      "ls_iterations": base_cfg.robot.ls_iterations,
                      "matrix": matrix}, indent=2), flush=True)
    if args.dry_run:
        return

    devices = jax.devices()
    if not devices or not all(device.platform == "gpu" for device in devices):
        raise RuntimeError(f"GPU execution is required; JAX devices are {devices}")
    args.output.mkdir(parents=True, exist_ok=True)
    progress_path = args.output / "benchmark.json"
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
    else:
        progress = {
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "config_source": str(args.config.resolve()),
            "winner_source": str(args.winner.resolve()) if args.winner else None,
            "morphology_plan_source": (str(args.morphology_plan.resolve())
                                       if args.morphology_plan else None),
            "runtime": runtime_versions(),
            "config": base_cfg.to_dict(),
            "episode_seconds": args.seconds,
            "episodes_per_condition": args.episodes,
            "evaluation_seed": args.seed,
            "conditions": {},
        }
    progress["matrix"] = matrix
    write_json(progress_path, progress)

    for speed in args.speeds:
        cfg = replace(base_cfg, cost=replace(base_cfg.cost, target_vx=speed)).validate()
        pending = [
            (name, theta) for name, theta in morphologies.items()
            if not (args.output / condition_name(speed, name) / "evaluation.json").exists()
        ]
        if not pending:
            print(f"Speed {speed:.2f} m/s already complete", flush=True)
            continue
        shared_kernel = None
        for index, (name, theta) in enumerate(pending):
            key = condition_name(speed, name)
            folder = args.output / key
            print(f"Running {key}: theta={theta.tolist()}", flush=True)
            evaluator = Evaluator(
                cfg,
                theta,
                shared_kernel=shared_kernel,
                reuse_compilation=True,
            )
            if shared_kernel is None:
                shared_kernel = evaluator.shared_kernel
            warmup_seconds = evaluator.warmup()
            print(f"  warmup {warmup_seconds:.2f}s", flush=True)
            progress["conditions"][key] = {
                "status": "running",
                "theta": theta.tolist(),
                "speed_m_s": speed,
                "warmup_seconds": warmup_seconds,
                "started_utc": datetime.now(timezone.utc).isoformat(),
            }
            write_json(progress_path, progress)
            result = evaluator.evaluate(seed=args.seed, output=folder, progress=True)
            _, metrics = design_objective(result, cfg)
            progress["conditions"][key] |= {
                "status": "complete",
                "objective": metrics,
                "fitness": result["fitness"],
                "episodes": result["episodes"],
                "output": str(folder.resolve()),
                "completed_utc": datetime.now(timezone.utc).isoformat(),
            }
            write_json(progress_path, progress)
            print(f"Completed {key}: {metrics}", flush=True)
            del evaluator

    progress["completed_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(progress_path, progress)
    print(f"Benchmark complete: {progress_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
