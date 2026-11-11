#!/usr/bin/env python3
"""Generate and inspect the nominal DRAFT-backed LeRobot lower body."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dial_lerobot.assets.draft_adapter import generate_draft_artifact, load_draft_model
from dial_lerobot.config import RobotConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--thigh", type=float, default=0.34136558643337206)
    parser.add_argument("--shin", type=float, default=0.2607679717488327)
    parser.add_argument("--hip-width", type=float, default=0.266319)
    parser.add_argument("--allow-infeasible", action="store_true")
    args = parser.parse_args()
    artifact = generate_draft_artifact(
        args.output,
        (args.thigh, args.shin, 2.0, args.hip_width),
        RobotConfig(),
        strict=not args.allow_infeasible,
    )
    model = load_draft_model(artifact)
    summary = {
        "xml": str(artifact.xml_path.resolve()),
        "nq": model.nq,
        "nv": model.nv,
        "nu": model.nu,
        "nbody": model.nbody,
        "ngeom": model.ngeom,
        "total_mass_kg": float(model.body_mass.sum()),
        "warnings": artifact.feasibility_report.get("warnings", []),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
