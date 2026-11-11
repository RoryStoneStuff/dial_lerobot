"""Audit DRAFT mass composition against the legacy LeRobot nominal model."""

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco


def _mass_breakdown(xml_path: Path) -> dict[str, float]:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    motor_mass = 0.0
    for geom in ET.parse(xml_path).getroot().findall(".//geom"):
        if not geom.get("name", "").endswith("_motor"):
            continue
        if geom.get("type") != "cylinder":
            raise ValueError(f"Expected cylindrical DRAFT motor geom: {geom.get('name')}")
        radius, half_length = (float(value) for value in geom.get("size").split()[:2])
        density = float(geom.get("density", "0"))
        motor_mass += 3.141592653589793 * radius * radius * (2.0 * half_length) * density
    total_mass = float(model.body_mass.sum())
    breakdown = {"motor": motor_mass, "structural": total_mass - motor_mass}
    breakdown["total"] = breakdown["motor"] + breakdown["structural"]
    return breakdown


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft-xml", type=Path, required=True)
    parser.add_argument("--legacy-metadata", type=Path, required=True)
    args = parser.parse_args()

    legacy = json.loads(args.legacy_metadata.read_text())
    target_mass = float(legacy["model"]["total_mass_kg"])
    draft = _mass_breakdown(args.draft_xml)
    structural_scale = (target_mass - draft["motor"]) / draft["structural"]
    motor_scale_if_structure_fixed = (target_mass - draft["structural"]) / draft["motor"]
    report = {
        "legacy_target_mass_kg": target_mass,
        "draft_mass_kg": draft,
        "required_structural_scale_at_current_motor_mass": structural_scale,
        "required_motor_scale_at_current_structural_mass": motor_scale_if_structure_fixed,
        "feasible_with_current_components": bool(
            draft["motor"] <= target_mass and draft["structural"] <= target_mass
        ),
        "interpretation": (
            "Current DRAFT motor requirements cannot fit inside the legacy total mass; "
            "a motor catalog or calibrated motor-mass model is required before setting link density."
        ),
    }
    print(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
