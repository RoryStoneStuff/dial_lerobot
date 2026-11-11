#!/usr/bin/env python3
"""Compare the current LeRobot model with the nominal DRAFT parity model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dial_lerobot.assets.draft_adapter import generate_draft_artifact, load_draft_model
from dial_lerobot.assets.foot_geometry import capsule_bottoms, foot_capsules
from dial_lerobot.assets.model_builder import JOINT_NAMES, build_model, initial_data


def summarize(model):
    data = initial_data(model)
    actuator_dofs = model.jnt_dofadr[model.actuator_trnid[:, 0]]
    return {
        "nq": model.nq,
        "nv": model.nv,
        "nu": model.nu,
        "nbody": model.nbody,
        "ngeom": model.ngeom,
        "total_mass_kg": float(model.body_mass.sum()),
        "neutral_root_height_m": float(data.qpos[2]),
        "minimum_foot_clearance_m": float(
            capsule_bottoms(model, data, foot_capsules(model)).min()
        ),
        "torque_limits_Nm": model.actuator_forcerange[:, 1].tolist(),
        "joint_armature_kgm2": model.dof_armature[actuator_dofs].tolist(),
    }


def kinematic_parity(legacy, draft):
    legacy_data = initial_data(legacy)
    draft_data = initial_data(draft)
    anchor_errors = []
    axis_errors = []
    joints = {}
    for name in JOINT_NAMES:
        legacy_id = legacy.joint(name).id
        draft_id = draft.joint(name).id
        anchor_error = float(np.linalg.norm(
            draft_data.xanchor[draft_id] - legacy_data.xanchor[legacy_id]
        ))
        cosine = float(np.clip(
            np.dot(draft_data.xaxis[draft_id], legacy_data.xaxis[legacy_id]), -1.0, 1.0
        ))
        axis_error = float(np.degrees(np.arccos(cosine)))
        anchor_errors.append(anchor_error)
        axis_errors.append(axis_error)
        joints[name] = {
            "anchor_error_mm": 1000.0 * anchor_error,
            "axis_error_degrees": axis_error,
        }
    return {
        "anchor_rms_mm": 1000.0 * float(np.sqrt(np.mean(np.square(anchor_errors)))),
        "anchor_max_mm": 1000.0 * max(anchor_errors),
        "axis_rms_degrees": float(np.sqrt(np.mean(np.square(axis_errors)))),
        "axis_max_degrees": max(axis_errors),
        "joints": joints,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    artifact = generate_draft_artifact(args.output)
    legacy_model = build_model()
    draft_model = load_draft_model(artifact)
    legacy = summarize(legacy_model)
    draft = summarize(draft_model)
    comparison = {
        "legacy": legacy,
        "draft": draft,
        "difference": {
            "total_mass_kg": draft["total_mass_kg"] - legacy["total_mass_kg"],
            "total_mass_percent": 100.0 * (
                draft["total_mass_kg"] / legacy["total_mass_kg"] - 1.0
            ),
            "neutral_root_height_m": (
                draft["neutral_root_height_m"] - legacy["neutral_root_height_m"]
            ),
        },
        "kinematic_parity": kinematic_parity(legacy_model, draft_model),
        "draft_warning_count": len(artifact.feasibility_report.get("warnings", [])),
    }
    encoded = json.dumps(comparison, indent=2) + "\n"
    if args.json:
        args.json.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
