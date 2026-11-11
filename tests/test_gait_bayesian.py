from dataclasses import replace
import json
from pathlib import Path
import signal
import subprocess
import sys

import pytest

from dial_lerobot.config import Config
from dial_lerobot.evaluator import runtime_versions, write_json
from dial_lerobot import gait_bayesian_optimize as gait_module


def test_worker_native_failure_is_recorded(monkeypatch, tmp_path):
    run = subprocess.run

    def crash(command, **kwargs):
        return run([sys.executable, "-c",
                    "import os, signal, resource; resource.setrlimit(resource.RLIMIT_CORE, (0, 0)); os.kill(os.getpid(), signal.SIGSEGV)"],
                   **kwargs)

    monkeypatch.setattr(gait_module.subprocess, "run", crash)
    with pytest.raises(gait_module.GaitWorkerError, match="status -11"):
        gait_module.optimize_gait_phase(Config(), tmp_path, [0.3, 0.3, 2.0], 1)
    status = json.loads((tmp_path / "worker_status.json").read_text())
    assert status["state"] == "failed"
    assert status["returncode"] == -signal.SIGSEGV
    assert status["signal"] == "SIGSEGV"


def test_worker_timeout_is_recorded(monkeypatch, tmp_path):
    run = subprocess.run

    def hang(command, **kwargs):
        return run([sys.executable, "-c", "import time; time.sleep(30)"], **kwargs)

    monkeypatch.setattr(gait_module.subprocess, "run", hang)
    with pytest.raises(gait_module.GaitWorkerError, match="timed out"):
        gait_module.optimize_gait_phase(Config(), tmp_path, [0.3, 0.3, 2.0], 1,
                                        timeout_seconds=0.1)
    assert json.loads((tmp_path / "worker_status.json").read_text())["state"] == "failed"


def test_runtime_metadata_can_avoid_initializing_parent_jax_device(monkeypatch):
    import dial_lerobot.evaluator as evaluator_module

    def unexpected_device_query():
        pytest.fail("Nested outer metadata must not initialize a parent GPU context")

    monkeypatch.setattr(evaluator_module.jax, "devices", unexpected_device_query)
    runtime = runtime_versions(include_devices=False)
    assert runtime["devices"] is None
    assert runtime["device_query"] == "deferred_to_isolated_worker"


def test_outer_continues_after_failed_worker_and_saves_tuned_gait(monkeypatch, tmp_path):
    from dial_lerobot import bayesian_optimize as outer

    cfg = Config()
    cfg = replace(cfg, design=replace(cfg.design, maxiter=1, popsize=2),
                  evaluation=replace(cfg.evaluation, num_episodes=1, episode_steps=10))
    monkeypatch.setattr(outer, "runtime_versions", lambda **kwargs: {})
    calls = []
    selected_gait = [0.6, 1.2, 0.1]

    def inner(cfg, output, theta, candidates, **kwargs):
        calls.append(output)
        assert kwargs.get("persistent_worker") is True
        assert kwargs.get("parameters", ("duty", "cadence", "gait_start_delay")) == (
            "duty", "cadence", "gait_start_delay")
        if len(calls) < 3:
            raise gait_module.GaitWorkerError("injected worker crash")
        selected = Path(output) / "candidate_000"
        write_json(selected / "evaluation.json", {
            "fitness": -1.0, "model": {}, "episodes": [{
                "sim_seconds": 0.1, "integrated_running_cost": 1.0,
                "positive_mechanical_cot": 0.5, "fell": False,
                "updates": 10, "mean_vx_m_s": 0.85,
            }],
        })
        return {"best_gait": selected_gait, "best_trajectory_folder": str(selected),
                "evaluations": [{"warmup_seconds": 2.0}]}

    def no_parent_evaluator(*args, **kwargs):
        pytest.fail("Nested mode must use the saved worker evaluation")

    monkeypatch.setattr(gait_module, "optimize_gait_phase", inner)
    monkeypatch.setattr(outer, "Evaluator", no_parent_evaluator)
    result = outer.optimize_design_bayesian(cfg, tmp_path, [0.3, 0.3, 2.0], 1)
    assert len(calls) == 3
    assert "error" in result["baseline"]
    assert "error" in result["generations"][0]["evaluations"][0]
    assert result["stop"]["maxiter"] == "1"
    saved = json.loads((tmp_path / "best_design.json").read_text())
    assert [saved["config"]["cost"][name] for name in gait_module.GAIT_NAMES] == selected_gait
    assert Path(saved["trajectory_folder"], "evaluation.json").exists()


def test_worker_server_handles_multiple_requests(monkeypatch, tmp_path):
    from dial_lerobot import gait_bayesian_worker as worker_module

    request_path = tmp_path / "request.json"
    result_path = tmp_path / "result.json"
    log_path = tmp_path / "worker.log"
    request_path.write_text(json.dumps({"value": 1}))
    messages = [{"request": str(request_path), "result": str(result_path),
                 "log": str(log_path)}] * 2
    monkeypatch.setattr(worker_module.sys, "stdin", iter(json.dumps(message) + "\n"
                                                          for message in messages))

    class ResponseOutput:
        def __init__(self):
            self.lines = []

        def write(self, value):
            self.lines.append(value)

        def flush(self):
            pass

    response_output = ResponseOutput()
    monkeypatch.setattr(worker_module.sys, "stdout", response_output)
    values = []

    def run_request(request):
        values.append(request["value"])
        return {"count": len(values)}

    monkeypatch.setattr(worker_module, "_run_request", run_request)
    worker_module._serve_requests()

    assert values == [1, 1]
    assert len(response_output.lines) == 2
    assert json.loads(result_path.read_text())["result"] == {"count": 2}


def test_persistent_worker_reuses_child_process(monkeypatch, tmp_path):
    script = (
        "import json, os, sys\n"
        "for line in sys.stdin:\n"
        " message = json.loads(line)\n"
        " with open(message['result'], 'w') as result_file:\n"
        "  json.dump({'ok': True, 'result': {'pid': os.getpid()}}, result_file)\n"
        " print(json.dumps({'ok': True}), flush=True)\n"
    )
    monkeypatch.setattr(gait_module, "_PERSISTENT_WORKER", None)
    monkeypatch.setattr(gait_module, "_PERSISTENT_WORKER_LOG", None)
    worker_command = [sys.executable, "-u", "-c", script]
    worker_pids = []
    try:
        for candidate_index in range(2):
            candidate_folder = tmp_path / f"candidate_{candidate_index}"
            candidate_folder.mkdir()
            request_path = candidate_folder / "request.json"
            request_path.write_text("{}")
            result_path = candidate_folder / "result.json"
            log_path = candidate_folder / "worker.log"
            status_path = candidate_folder / "worker_status.json"
            write_json(status_path, {"state": "running"})
            result = gait_module._optimize_gait_phase_persistent(
                request_path, result_path, log_path, status_path, worker_command, 5,
            )
            worker_pids.append(result["pid"])
    finally:
        gait_module._stop_persistent_worker()

    assert worker_pids[0] == worker_pids[1]


def test_all_failed_workers_finalize_without_best_design(monkeypatch, tmp_path):
    from dial_lerobot import bayesian_optimize as outer

    cfg = Config()
    cfg = replace(cfg, design=replace(cfg.design, maxiter=1, popsize=2))
    monkeypatch.setattr(outer, "runtime_versions", lambda **kwargs: {})

    def fail(*args, **kwargs):
        raise gait_module.GaitWorkerError("injected failure")

    monkeypatch.setattr(gait_module, "optimize_gait_phase", fail)
    result = outer.optimize_design_bayesian(cfg, tmp_path, [0.3, 0.3, 2.0], 1)
    assert result["best_theta"] is None
    assert result["best_objective"] is None
    assert result["stop"]
    assert not (tmp_path / "best_design.json").exists()


@pytest.mark.parametrize("drop_checkpoint", [False, True])
def test_bayesian_checkpoint_resumes_after_interrupted_candidate(monkeypatch, tmp_path, drop_checkpoint):
    from dial_lerobot import bayesian_optimize as outer

    cfg = Config()
    cfg = replace(cfg, design=replace(cfg.design, maxiter=1, popsize=2),
                  evaluation=replace(cfg.evaluation, num_episodes=1, episode_steps=10))
    monkeypatch.setattr(outer, "runtime_versions", lambda **kwargs: {})
    calls = []

    def inner(cfg, output, theta, candidates, **kwargs):
        calls.append(output)
        if len(calls) == 3:
            raise SystemExit("simulated abrupt parent termination")
        selected = Path(output) / "candidate_000"
        write_json(selected / "evaluation.json", {
            "fitness": -1.0, "model": {"theta": theta}, "episodes": [{
                "sim_seconds": 0.1, "integrated_running_cost": 1.0,
                "positive_mechanical_cot": 0.5, "fell": False,
                "updates": 10, "mean_vx_m_s": 0.85,
            }],
        })
        return {"best_gait": [cfg.cost.duty, cfg.cost.cadence, cfg.cost.gait_start_delay],
                "best_trajectory_folder": str(selected),
                "evaluations": [{"warmup_seconds": 0.0}]}

    monkeypatch.setattr(gait_module, "optimize_gait_phase", inner)
    with pytest.raises(SystemExit, match="simulated abrupt"):
        outer.optimize_design_bayesian(cfg, tmp_path, [0.3, 0.3, 2.0], 1)

    checkpoint = json.loads((tmp_path / "checkpoint.json").read_text())
    assert checkpoint["generation"] == 0
    assert checkpoint["candidate"] == 1
    if drop_checkpoint:
        (tmp_path / "checkpoint.json").unlink()

    result = outer.optimize_design_bayesian(cfg, tmp_path, [0.3, 0.3, 2.0], 1, resume=True)
    assert result["stop"]["maxiter"] == "1"
    assert len(result["generations"]) == 1
    assert len(calls) == 4
