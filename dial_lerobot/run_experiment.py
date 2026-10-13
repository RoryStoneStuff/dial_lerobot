"""Command line counterpart of co_design's run_experiment for the zoo lower body."""

import argparse
from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path

from dial_lerobot.config import from_dict, load_config


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", type=Path, help="TOML settings; omitted values use baseline defaults")
    common.add_argument("--zoo-root", help="Path to existing lerobot-legged-zoo")
    common.add_argument("--timestep", type=float, help="Physics timestep in seconds")
    source = common.add_mutually_exclusive_group()
    source.add_argument("--theta", nargs="+", type=float, metavar="VALUE",
                        help="THIGH_M SHIN_M DENSITY_PROXY [HIP_WIDTH_M]; three-value designs retain nominal hip spacing")
    source.add_argument("--design", type=Path, help="best_design.json; loads saved config unless --config is given")
    for name, help_text in (("vx", "Forward target in m/s"), ("vy", "Lateral target in m/s"), ("wz", "Yaw-rate target in rad/s"),
                            ("horizon", "Planning horizon in seconds"), ("noise", "DIAL sampling noise"),
                            ("temperature", "DIAL weighting temperature")):
        common.add_argument(f"--{name}", type=float, help=help_text)
    for name, help_text in (("samples", "DIAL sample count"), ("knots", "Control spline knots"),
                            ("iterations", "DIAL refinement iterations per MPC update"),
                            ("decimation", "Physics steps per MPC update: baseline 1, PPO cadence 4"),
                            ("steps", "MPC updates per episode"), ("episodes", "Evaluation episodes"),
                            ("seed", "Evaluation seed")):
        common.add_argument(f"--{name}", type=int, help=help_text)
    cmd = sub.add_parser("inspect", parents=[common], help="Describe model without compiling DIAL")
    cmd.add_argument("--export-xml", type=Path)
    for name, help_text in (("test", "Run headless DIAL; save trajectories and costs"),
                            ("optimize", "Run CMA-ES morphology search"),
                            ("heatmap", "Sweep thigh/shin lengths using DIAL")):
        cmd = sub.add_parser(name, parents=[common], help=help_text)
        cmd.add_argument("--output", type=Path, help="New results folder; default is timestamped")
        if name == "optimize":
            cmd.add_argument("--generations", type=int)
            cmd.add_argument("--popsize", type=int)
        if name == "heatmap":
            cmd.add_argument("--points", type=int, default=5)
            cmd.add_argument("--density", type=float, default=2.0)
    cmd = sub.add_parser("render", parents=[common], help="Live MuJoCo viewer using DIAL")
    cmd.add_argument("--seconds", type=float, default=10, help="Pause at this simulation duration; 0 runs until closed")
    cmd.add_argument("--rate", type=float, default=1.0, help="Maximum playback speed")
    cmd = sub.add_parser("snapshot", parents=[common], help="Render model to PNG")
    cmd.add_argument("--output", type=Path, default=Path("results/nominal_model.png"))
    sub.add_parser("contacts", parents=[common],
                   help="Open a MuJoCo-only viewer for generated foot contact geometry")
    cmd = sub.add_parser("replay", help="MuJoCo playback of saved .npz, without computing MPC")
    cmd.add_argument("trajectory", type=Path)
    cmd.add_argument("--rate", type=float, default=1.0)
    cmd.add_argument("--loop", action="store_true")
    cmd = sub.add_parser("visualize", help="Plot .npz, optimization.json or heatmap.json")
    cmd.add_argument("input", type=Path)
    cmd.add_argument("--output", type=Path)
    return p


def resolve(args):
    design = json.loads(args.design.read_text()) if args.design else None
    cfg = from_dict(design["config"]) if design and not args.config else load_config(args.config)
    theta = design["theta"] if design else args.theta
    if theta is not None and len(theta) not in (3, 4):
        raise ValueError("--theta requires three or four values")
    mappings = {
        "robot": {"zoo_root": "zoo_root", "timestep": "timestep"},
        "cost": {"vx": "target_vx", "vy": "target_vy", "wz": "target_wz"},
        "mpc": {"samples": "num_samples", "horizon": "plan_horizon", "knots": "num_knots",
                "iterations": "iterations", "noise": "noise_level", "temperature": "temperature",
                "decimation": "control_decimation"},
        "evaluation": {"steps": "episode_steps", "episodes": "num_episodes", "seed": "seed"},
        "design": {"generations": "maxiter", "popsize": "popsize"},
    }
    for section, mapping in mappings.items():
        updates = {dest: getattr(args, arg) for arg, dest in mapping.items() if getattr(args, arg, None) is not None}
        cfg = replace(cfg, **{section: replace(getattr(cfg, section), **updates)})
    return cfg.validate(), theta


def new_output(args):
    output = args.output or Path("results") / f"{args.command}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Results folder is not empty: {output}. Choose a new --output folder.")
    output.mkdir(parents=True, exist_ok=True)
    print(f"Results: {output.resolve()}", flush=True)
    return output


def main():
    p = parser()
    args = p.parse_args()
    if getattr(args, "rate", 1) <= 0:
        p.error("--rate must be positive")
    if getattr(args, "seconds", 0) < 0:
        p.error("--seconds must be nonnegative")
    if args.command == "replay":
        from dial_lerobot.visualize_lerobot import replay
        replay(args.trajectory, args.rate, args.loop)
        return
    if args.command == "visualize":
        from dial_lerobot.visualize_lerobot import plot_results
        print(plot_results(args.input, args.output))
        return
    cfg, theta = resolve(args)
    if args.command in ("inspect", "snapshot", "contacts"):
        from dial_lerobot.assets.model_builder import build_model, describe_model, model_xml, nominal_design
        theta = nominal_design(cfg.robot) if theta is None else theta
        model = build_model(theta, cfg.robot)
        if args.command == "inspect":
            print(json.dumps(describe_model(model, theta, cfg.robot), indent=2))
            if args.export_xml:
                args.export_xml.parent.mkdir(parents=True, exist_ok=True)
                args.export_xml.write_text(model_xml(theta, cfg.robot))
        elif args.command == "snapshot":
            from dial_lerobot.visualize_lerobot import snapshot
            snapshot(model, args.output)
            print(args.output.resolve())
        else:
            from dial_lerobot.visualize_lerobot import contact_view
            contact_view(model)
        return
    if args.command in ("test", "render"):
        from dial_lerobot.evaluator import Evaluator
        evaluator = Evaluator(cfg, theta)
        if args.command == "render":
            from dial_lerobot.visualize_lerobot import live_view
            live_view(evaluator, args.seconds, args.rate)
        else:
            output = new_output(args)
            print("Compiling and warming up DIAL...", flush=True)
            print(f"Warmup: {evaluator.warmup():.2f}s", flush=True)
            result = evaluator.evaluate(output=output, progress=True)
            print(json.dumps({"fitness": result["fitness"], "episodes": result["episodes"]}, indent=2))
    elif args.command == "optimize":
        from dial_lerobot.optimize import optimize_design
        result = optimize_design(cfg, new_output(args), theta)
        print(f"Best theta: {result['best_theta']}; objective: {result['best_objective']}")
    elif args.command == "heatmap":
        from dial_lerobot.optimize import design_heatmap
        if theta is not None:
            p.error("heatmap uses design bounds and --density, not --theta/--design")
        design_heatmap(cfg, new_output(args), args.points, args.density)


if __name__ == "__main__":
    main()
