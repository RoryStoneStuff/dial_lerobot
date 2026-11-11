"""Standalone worker for the Bayesian gait inner loop."""

import argparse
from contextlib import redirect_stderr, redirect_stdout
import json
from pathlib import Path
import sys

from dial_lerobot.config import from_dict
from dial_lerobot.evaluator import write_json
from dial_lerobot.gait_bayesian_optimize import _optimize_gait_phase_impl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--server", action="store_true")
    args = parser.parse_args()
    if args.server:
        _serve_requests()
        return
    if args.request is None or args.result is None:
        parser.error("--request and --result are required unless --server is used")
    request = json.loads(args.request.read_text())
    try:
        result = _run_request(request)
        write_json(args.result, {"ok": True, "result": result})
    except BaseException as exc:
        write_json(args.result, {"ok": False,
                                 "error": f"{type(exc).__name__}: {exc}"})
        raise


def _run_request(request):
    return _optimize_gait_phase_impl(
        from_dict(request["config"]), request["output"], request["theta"],
        request["candidates"], request.get("parameters"),
    )


def _serve_requests():
    for line in sys.stdin:
        message = json.loads(line)
        request = json.loads(Path(message["request"]).read_text())
        result_path = Path(message["result"])
        log_path = Path(message["log"])
        try:
            with log_path.open("a") as worker_log:
                with redirect_stdout(worker_log), redirect_stderr(worker_log):
                    result = _run_request(request)
            write_json(result_path, {"ok": True, "result": result})
            response = {"ok": True}
        except BaseException as exc:
            write_json(result_path, {"ok": False,
                                     "error": f"{type(exc).__name__}: {exc}"})
            response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
