"""Small Bayesian inner loop for gait timing around a fixed morphology."""

from datetime import datetime, timezone
import atexit
from dataclasses import replace
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time

import numpy as np

from dial_lerobot.assets.model_builder import nominal_design
from dial_lerobot.bayesian_optimize import _next_point, _utility
from dial_lerobot.config import Config
from dial_lerobot.design_objective import design_objective
from dial_lerobot.evaluator import Evaluator, runtime_versions, write_json


GAIT_NAMES = ("duty", "cadence", "gait_start_delay")
LOWER = np.asarray((0.40, 0.70, 0.00), dtype=float)
UPPER = np.asarray((0.70, 1.40, 0.50), dtype=float)
RANDOM_INITIAL_CANDIDATES = 2
WORKER_TIMEOUT_SECONDS = 900.0
_PERSISTENT_WORKER = None
_PERSISTENT_WORKER_LOG = None
_WORKER_SHARED_KERNEL = None


class GaitWorkerError(RuntimeError):
    """The isolated gait search failed; the outer search may reject this design."""


def with_gait_parameters(cfg, gait):
    gait = np.asarray(gait, dtype=float)
    if gait.shape != (3,) or not np.isfinite(gait).all():
        raise ValueError("Gait parameters must contain duty, cadence, and gait start delay")
    return replace(
        cfg,
        cost=replace(cfg.cost, duty=float(gait[0]), cadence=float(gait[1]),
                     gait_start_delay=float(gait[2])),
    )


def _default_theta(cfg):
    return nominal_design(cfg.robot, include_hip_width=len(cfg.design.lower_bounds) >= 4,
                           include_motor_design=len(cfg.design.lower_bounds) == 10)


def _optimize_gait_phase_impl(cfg: Config, output, theta, candidates, parameters=GAIT_NAMES):
    global _WORKER_SHARED_KERNEL
    cfg.validate()
    if candidates < 1:
        raise ValueError("gait candidates must be positive")
    parameters = tuple(parameters)
    if not parameters or len(set(parameters)) != len(parameters) or any(name not in GAIT_NAMES for name in parameters):
        raise ValueError(f"Gait parameters must be unique names from {GAIT_NAMES}")
    parameter_indices = np.asarray([GAIT_NAMES.index(name) for name in parameters], dtype=int)
    lower = LOWER[parameter_indices]
    upper = UPPER[parameter_indices]
    theta = _default_theta(cfg) if theta is None else np.asarray(theta, dtype=float)
    if theta.shape != (len(cfg.design.lower_bounds),):
        raise ValueError("Fixed morphology must match the configured design dimensions")
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    log = {"optimizer": "gait_bayesian_gp_ucb", "started_utc": datetime.now(timezone.utc).isoformat(),
           "config": cfg.to_dict(), "runtime": runtime_versions(), "theta": theta.tolist(),
           "parameters": GAIT_NAMES, "optimized_parameters": parameters,
           "bounds": {"lower": lower.tolist(), "upper": upper.tolist()},
           "candidates": candidates, "evaluations": [], "best_objective": None,
           "best_gait": None}
    write_json(folder / "optimization.json", log)
    rng = np.random.default_rng(cfg.design.seed)
    observations_x, observations_y = [], []
    best_key = None
    shared_kernel = _WORKER_SHARED_KERNEL
    start = time.perf_counter()

    random_initial = min(RANDOM_INITIAL_CANDIDATES, candidates)
    for index in range(candidates + 1):
        if index == 0:
            gait = np.asarray((cfg.cost.duty, cfg.cost.cadence, cfg.cost.gait_start_delay), dtype=float)
            name = "baseline"
        elif len(observations_x) < random_initial:
            proposal = rng.uniform(lower, upper)
            name = f"candidate_{index - 1:03d}"
        else:
            normalized_x = [(point - lower) / (upper - lower) for point in observations_x]
            normalized_fixed = {}
            proposal = _next_point(normalized_x, observations_y, np.zeros(len(parameters)),
                                   np.ones(len(parameters)), normalized_fixed, rng)
            proposal = lower + (upper - lower) * proposal
            name = f"candidate_{index - 1:03d}"
        if index > 0:
            gait = np.asarray((cfg.cost.duty, cfg.cost.cadence, cfg.cost.gait_start_delay), dtype=float)
            gait[parameter_indices] = proposal
        gait_cfg = with_gait_parameters(cfg, gait)
        candidate_folder = folder / name
        log["current_candidate"] = name
        write_json(folder / "optimization.json", log)
        print(f"Gait {index + 1}/{candidates + 1}: {name} {gait.tolist()}", flush=True)
        try:
            evaluator = Evaluator(gait_cfg, theta, shared_kernel=shared_kernel, reuse_compilation=True)
            warmup = evaluator.warmup()
            shared_kernel = evaluator.shared_kernel
            _WORKER_SHARED_KERNEL = shared_kernel
            result = evaluator.evaluate(seed=gait_cfg.evaluation.seed, output=candidate_folder)
            key, metrics = design_objective(result, gait_cfg)
            utility = _utility(metrics, gait_cfg)
        except Exception as exc:
            evaluation = {"gait": gait.tolist(), "error": str(exc),
                          "fitness": -1e12, "utility": -1e6}
            log["evaluations"].append(evaluation)
            log["wall_seconds"] = time.perf_counter() - start
            write_json(folder / "optimization.json", log)
            continue
        finally:
            if "evaluator" in locals():
                del evaluator
        observations_x.append(gait[parameter_indices].copy())
        observations_y.append(utility)
        evaluation = {"gait": gait.tolist(), "objective": metrics, "fitness": result["fitness"],
                      "proposal": "baseline" if index == 0 else
                      ("random" if index <= random_initial else "gp_ucb"),
                      "utility": utility, "warmup_seconds": warmup,
                      "trajectory_folder": str(candidate_folder.resolve())}
        log["evaluations"].append(evaluation)
        if best_key is None or key > best_key:
            best_key = key
            log["best_objective"] = metrics
            log["best_gait"] = gait.tolist()
            log["best_trajectory_folder"] = str(candidate_folder.resolve())
        log["wall_seconds"] = time.perf_counter() - start
        write_json(folder / "optimization.json", log)
    if log["best_gait"] is None:
        raise ValueError("All gait-phase evaluations failed")
    log["stop"] = {"candidates": candidates}
    write_json(folder / "optimization.json", log)
    return log


def optimize_gait_phase(cfg: Config, output, theta=None, candidates=12, *,
                        parameters=GAIT_NAMES, timeout_seconds=None,
                        persistent_worker=False):
    """Run gait optimization in a standalone Python process."""
    cfg.validate()
    if candidates < 1:
        raise ValueError("gait candidates must be positive")
    theta = _default_theta(cfg) if theta is None else np.asarray(theta, dtype=float)
    if theta.shape != (len(cfg.design.lower_bounds),):
        raise ValueError("Fixed morphology must match the configured design dimensions")
    folder = Path(output).expanduser().resolve()
    folder.mkdir(parents=True, exist_ok=True)
    request_path = folder / "worker_request.json"
    result_path = folder / "worker_result.json"
    if request_path.exists() or result_path.exists():
        raise ValueError(f"Gait worker output already exists: {folder}")
    write_json(request_path, {"config": cfg.to_dict(), "theta": theta.tolist(),
                              "output": str(folder), "candidates": int(candidates),
                              "parameters": list(parameters)})
    environment = os.environ.copy()
    source_root = str(Path(__file__).resolve().parents[1])
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, (source_root, environment.get("PYTHONPATH"))))
    environment["PYTHONFAULTHANDLER"] = "1"
    command = [sys.executable, "-u", "-X", "faulthandler", "-m", "dial_lerobot.gait_bayesian_worker",
               "--request", str(request_path), "--result", str(result_path)]
    log_path = folder / "worker.log"
    status_path = folder / "worker_status.json"
    status = {"state": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
              "log": str(log_path), "command": command}
    write_json(status_path, status)
    timeout = timeout_seconds if timeout_seconds is not None else WORKER_TIMEOUT_SECONDS * (candidates + 1)
    if persistent_worker and timeout_seconds is None:
        return _optimize_gait_phase_persistent(
            request_path, result_path, log_path, status_path, command, timeout,
        )
    try:
        with log_path.open("w") as worker_log:
            completed = subprocess.run(
                command, cwd=source_root, env=environment, stdout=worker_log,
                stderr=subprocess.STDOUT, timeout=timeout, check=False,
            )
        status["returncode"] = completed.returncode
        if completed.returncode < 0:
            status["signal"] = signal.Signals(-completed.returncode).name
        if completed.returncode != 0:
            raise GaitWorkerError(f"Gait worker exited with status {completed.returncode}; see {log_path}")
        if not result_path.exists():
            raise GaitWorkerError(f"Gait worker returned no result; see {log_path}")
        worker_result = json.loads(result_path.read_text())
        if not worker_result.get("ok"):
            raise GaitWorkerError(worker_result.get("error", "Unknown gait worker failure"))
    except (subprocess.TimeoutExpired, OSError, ValueError, GaitWorkerError) as exc:
        status.update(state="failed", error=str(exc), finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(status_path, status)
        raise GaitWorkerError(f"Gait worker failed: {exc}; see {log_path}") from exc
    status.update(state="completed", finished_utc=datetime.now(timezone.utc).isoformat())
    write_json(status_path, status)
    return worker_result["result"]


def _stop_persistent_worker():
    global _PERSISTENT_WORKER, _PERSISTENT_WORKER_LOG
    worker = _PERSISTENT_WORKER
    _PERSISTENT_WORKER = None
    if worker is not None:
        if worker.stdin is not None:
            try:
                worker.stdin.close()
            except OSError:
                pass
        try:
            worker.wait(timeout=10)
        except subprocess.TimeoutExpired:
            worker.terminate()
            try:
                worker.wait(timeout=5)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait()
    if _PERSISTENT_WORKER_LOG is not None:
        _PERSISTENT_WORKER_LOG.close()
        _PERSISTENT_WORKER_LOG = None


def _start_persistent_worker(command, log_path):
    global _PERSISTENT_WORKER, _PERSISTENT_WORKER_LOG
    _stop_persistent_worker()
    _PERSISTENT_WORKER_LOG = log_path.open("a")
    try:
        _PERSISTENT_WORKER = subprocess.Popen(
            [*command, "--server"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=_PERSISTENT_WORKER_LOG, text=True, bufsize=1,
            cwd=Path(__file__).resolve().parents[1], env=os.environ.copy(),
        )
    except OSError:
        _PERSISTENT_WORKER_LOG.close()
        _PERSISTENT_WORKER_LOG = None
        raise
    return _PERSISTENT_WORKER


def _optimize_gait_phase_persistent(request_path, result_path, log_path, status_path,
                                    command, timeout):
    global _PERSISTENT_WORKER
    if _PERSISTENT_WORKER is None or _PERSISTENT_WORKER.poll() is not None:
        worker = _start_persistent_worker(command, log_path)
    else:
        worker = _PERSISTENT_WORKER
    status = json.loads(status_path.read_text())
    status["worker_pid"] = worker.pid
    try:
        worker.stdin.write(json.dumps({
            "request": str(request_path), "result": str(result_path),
            "log": str(log_path),
        }) + "\n")
        worker.stdin.flush()
        ready, _, _ = select.select([worker.stdout], [], [], timeout)
        if not ready:
            raise subprocess.TimeoutExpired(command, timeout)
        response_line = worker.stdout.readline()
        if not response_line:
            return_code = worker.poll()
            raise GaitWorkerError(
                f"Persistent gait worker exited with status {return_code}; see {log_path}"
            )
        response = json.loads(response_line)
        if not response.get("ok"):
            raise GaitWorkerError(response.get("error", "Persistent gait worker failed"))
        worker_result = json.loads(result_path.read_text())
        if not worker_result.get("ok"):
            raise GaitWorkerError(worker_result.get("error", "Gait optimization failed"))
    except (OSError, ValueError, GaitWorkerError, subprocess.TimeoutExpired) as exc:
        if _PERSISTENT_WORKER is not None:
            _stop_persistent_worker()
        status.update(state="failed", error=str(exc),
                      returncode=worker.poll() if worker.poll() is not None else None,
                      finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(status_path, status)
        raise GaitWorkerError(f"Gait worker failed: {exc}; see {log_path}") from exc
    status.update(state="completed", finished_utc=datetime.now(timezone.utc).isoformat())
    write_json(status_path, status)
    return worker_result["result"]


atexit.register(_stop_persistent_worker)
