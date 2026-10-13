"""Matched, fall-terminated screening of current and source H1 reward weights."""
import argparse
import hashlib
from dataclasses import replace
from pathlib import Path
import subprocess

import jax
import numpy as np
from dial_lerobot.config import load_config
from dial_lerobot.evaluator import Evaluator, write_json
from dial_lerobot.assets.model_builder import model_xml
from scripts.analyze_stance import analyze


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seconds", type=float, default=3.0)
    p.add_argument("--samples", type=int, default=1024)
    p.add_argument("--case", choices=("all", "paper"), default="all")
    p.add_argument("--paper-temperature", type=float)
    p.add_argument("--paper-config", default="configs/paper_weights.toml")
    p.add_argument("--seeds", type=int, nargs="+", default=[45, 46],
                   help="Direct episode seeds for the paper preset")
    a = p.parse_args()
    if a.seconds <= 0 or (a.output.exists() and any(a.output.iterdir())):
        p.error("Use positive seconds and a new output directory")
    a.output.mkdir(parents=True, exist_ok=True)
    git = ["git", "-c", "safe.directory=" + str(Path.cwd())]
    source = Path("../dial-mpc")
    write_json(a.output / "provenance.json", {
        "git_head": subprocess.check_output(git + ["rev-parse", "HEAD"], text=True).strip(),
        "git_diff": subprocess.check_output(git + ["diff"], text=True),
        "source_h1_sha256": hashlib.sha256((source / "dial_mpc/envs/unitree_h1_env.py").read_bytes()).hexdigest(),
        "runner": Path(__file__).read_text(),
        "devices": [str(d) for d in jax.devices()],
    })
    for case, config, seeds in [
        ("current", "configs/stance_contact.toml", [45]),
        ("paper", a.paper_config, a.seeds),
    ]:
        if a.case == "paper" and case != "paper":
            continue
        cfg = load_config(config)
        if case == "paper" and a.paper_temperature is not None:
            cfg = replace(cfg, mpc=replace(cfg.mpc, temperature=a.paper_temperature))
        dt = cfg.robot.timestep * cfg.mpc.control_decimation
        cfg = replace(cfg, mpc=replace(cfg.mpc, num_samples=a.samples),
                      evaluation=replace(cfg.evaluation, episode_steps=round(a.seconds / dt),
                                         height_jitter=0.0, terminate_on_fall=True))
        folder = a.output / case
        folder.mkdir()
        (folder / "preset.toml").write_text(Path(config).read_text())
        e = Evaluator(cfg)
        (folder / "model.xml").write_text(model_xml(e.theta, cfg.robot))
        write_json(folder / "metadata.json", e.metadata())
        print(f"Case {case}, direct seeds {seeds}, devices {jax.devices()}; compiling", flush=True)
        warmup = e.warmup()
        print(f"Warmup {warmup:.2f}s", flush=True)
        episodes = []
        for seed in seeds:
            summary, arrays = e.episode(seed, progress=True)
            np.savez_compressed(folder / f"seed_{seed}.npz", **arrays)
            episodes.append(summary)
            write_json(folder / "results.json", {"warmup_seconds": warmup, "episodes": episodes})
            print(summary, flush=True)
            analyze(folder)


if __name__ == "__main__":
    main()
