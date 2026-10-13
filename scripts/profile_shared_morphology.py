"""Validate shared compilation and profile a single, warmed 256-sample episode."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import numpy as np

from dial_lerobot.config import load_config
from dial_lerobot.evaluator import Evaluator, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--design", type=Path, help="Optional saved best_design.json; otherwise profile nominal morphology")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    cfg = load_config("configs/codesign_extreme_256.toml")
    cfg = replace(cfg, evaluation=replace(cfg.evaluation, num_episodes=1))
    theta = json.loads(args.design.read_text())["theta"] if args.design else None
    evaluator = Evaluator(cfg, theta, reuse_compilation=True)
    print("Compiling shared 256-sample evaluator...", flush=True)
    timings = {"first_warmup_seconds": evaluator.warmup()}
    kernel = evaluator.shared_kernel
    variants = kernel.step._cache_size()
    print(timings, "cache entries:", variants, flush=True)
    # Exercise genuinely different lengths, masses and hip width using the
    # SAME function, including bounds of the wider morphology search.
    for label, design in (("nominal", None), ("lower_bounds", cfg.design.lower_bounds),
                          ("upper_bounds", cfg.design.upper_bounds)):
        other = Evaluator(cfg, design, shared_kernel=kernel)
        t = time.perf_counter()
        state, params = other.reset(height_offset=0)
        state, params, row = other.step(state, params)
        jax.block_until_ready(state.qpos)
        timings[label + "_first_step_seconds"] = time.perf_counter() - t
        print("Cache size after switch:", kernel.step._cache_size(), flush=True)
        assert kernel.step._cache_size() == variants, "Morphology triggered recompilation"
        assert np.isfinite(np.asarray(row["plan_cost_min"]))
        print(label, timings[label + "_first_step_seconds"], "cache entries:", kernel.step._cache_size(), flush=True)
    write_json(output / "compilation.json", timings)
    print("Profiling one episode; compilation is excluded...", flush=True)
    gpu_path = output / "gpu.csv"
    with gpu_path.open("w") as gpu:
        monitor = subprocess.Popen([
            "nvidia-smi", "--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw",
            "--format=csv,noheader,nounits", "-lms", "200"], stdout=gpu)
        try:
            result = evaluator.evaluate(seed=cfg.evaluation.seed, output=output, progress=True)
        finally:
            monitor.terminate()
            monitor.wait(timeout=10)
    samples = [line.split(",") for line in gpu_path.read_text().splitlines() if line.strip()]
    values = np.asarray([[float(x) for x in line[1:]] for line in samples])
    report = {"samples": cfg.mpc.num_samples, "compilation": timings,
              "compiled_variants": kernel.step._cache_size(), "gpu_observations": len(samples),
              "gpu_utilization_mean_percent": float(values[:, 0].mean()),
              "gpu_utilization_p95_percent": float(np.percentile(values[:, 0], 95)),
              "gpu_memory_peak_MiB": float(values[:, 2].max()),
              "gpu_memory_total_MiB": float(values[:, 3].max()),
              "gpu_power_mean_watts": float(values[:, 4].mean()),
              "episode": result["episodes"][0]}
    write_json(output / "profile.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
