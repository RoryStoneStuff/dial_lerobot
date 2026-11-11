"""Opt-in DRAFT generation and adaptation for the 12-DOF locomotion task."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from dial_lerobot.assets.foot_geometry import capsule_bottoms, foot_capsules
from dial_lerobot.config import RobotConfig


DRAFT_JOINT_NAMES = (
    "hipz_right", "hipz_left", "hipx_right", "hipx_left",
    "hipy_right", "hipy_left", "knee_right", "knee_left",
    "ankley_right", "anklex_right", "ankley_left", "anklex_left",
)
SOURCE_JOINT_NAMES = {
    "hipz_right": "right_hipz",
    "hipz_left": "left_hipz",
    "hipx_right": "right_hipx",
    "hipx_left": "left_hipx",
    "hipy_right": "right_hipy",
    "hipy_left": "left_hipy",
    "knee_right": "right_knee",
    "knee_left": "left_knee",
    "ankley_right": "right_ankley",
    "anklex_right": "right_anklex",
    "ankley_left": "left_ankley",
    "anklex_left": "left_anklex",
}
DRAFT_NOMINAL_DESIGN = np.asarray(
    (0.34136558643337206, 0.2607679717488327, 2.0, 0.266319), dtype=float
)
DRAFT_MOTOR_CLASSES = ("HY", "KN", "AN")
DRAFT_NOMINAL_MOTOR_SCALES = np.ones(6, dtype=float)
DRAFT_BASE_MOTOR_REQUIREMENTS = {
    "HY": (200.0, 18.0),
    "KN": (300.0, 18.0),
    "AN": (40.0, 25.0),
}
DRAFT_JOINT_MOTOR_CLASS = {
    **{f"hip{axis}_{side}": "HY" for axis in ("z", "x", "y") for side in ("left", "right")},
    "knee_left": "KN", "knee_right": "KN",
    **{f"ankle{axis}_{side}": "AN" for axis in ("y", "x") for side in ("left", "right")},
}
CACHE_FORMAT_VERSION = 7
JOINT_MARKER_RADIUS = 0.025

# Source LeRobot motor centers, expressed in the source thigh/shin body frames
# at nominal scale. The knee and ankle motors are remote-mounted; they are not
# located at the corresponding downstream joint centers.
REMOTE_MOTOR_MOUNTS = {
    "left_thigh_link_motor": ("left_thigh_link", np.array((-0.058, -0.100459, -0.0575)), "thigh"),
    "right_thigh_link_motor": ("right_thigh_link", np.array((0.058, 0.100459, -0.0575)), "thigh"),
    "left_shin_link_motor": ("left_shin_link", np.array((0.0, -0.106, 0.0065)), "shin"),
    "right_shin_link_motor": ("right_shin_link", np.array((-0.045215, -0.027654, 0.0065)), "shin"),
    "left_ankle_pitch_link_motor": ("left_shin_link", np.array((0.0, -0.053, -0.0065)), "shin"),
    "right_ankle_pitch_link_motor": ("right_shin_link", np.array((-0.090428, -0.055307, -0.0065)), "shin"),
}


@dataclass(frozen=True)
class DraftArtifact:
    output_dir: Path
    xml_path: Path
    feasibility_report: dict
    resolved_parameters: dict


def draft_spec_dir() -> Path:
    return Path(__file__).with_name("draft_lower_body")


def draft_nominal_design(include_hip_width=False, include_motor_design=False) -> np.ndarray:
    count = 4 if include_hip_width else 3
    result = DRAFT_NOMINAL_DESIGN[:count].copy()
    if include_motor_design:
        if not include_hip_width:
            raise ValueError("DRAFT motor design requires hip width in theta")
        result = np.concatenate((result, DRAFT_NOMINAL_MOTOR_SCALES))
    return result


def _motor_requirements(theta: np.ndarray) -> dict:
    if theta.shape == (4,):
        scales = DRAFT_NOMINAL_MOTOR_SCALES
    elif theta.shape == (10,):
        scales = theta[4:]
        if not np.isfinite(scales).all() or np.any(scales <= 0):
            raise ValueError("DRAFT motor design scales must be finite and positive")
    else:
        raise ValueError("DRAFT theta requires 4 geometry values or 10 geometry/motor values")
    requirements = {}
    for index, motor_class in enumerate(DRAFT_MOTOR_CLASSES):
        torque_scale = float(scales[index])
        speed_scale = float(scales[index + 3])
        torque, speed = DRAFT_BASE_MOTOR_REQUIREMENTS[motor_class]
        requirements[motor_class] = {
            "effort": torque * torque_scale,
            "velocity": speed * speed_scale,
            "torque_scale": torque_scale,
            "speed_scale": speed_scale,
        }
    return requirements


def motor_feedback_theta(theta, result, safety_margin=1.20) -> np.ndarray:
    """Upsize DRAFT motors from observed MPC demand, preserving CMA's floor."""
    theta = np.asarray(theta, dtype=float).copy()
    if theta.shape != (10,):
        return theta
    peak_torque = {name: 0.0 for name in DRAFT_MOTOR_CLASSES}
    peak_speed = {name: 0.0 for name in DRAFT_MOTOR_CLASSES}
    for episode in result.get("episodes", []):
        for joint, value in episode.get("max_torque_by_joint", {}).items():
            motor_class = DRAFT_JOINT_MOTOR_CLASS.get(joint)
            if motor_class is not None:
                peak_torque[motor_class] = max(peak_torque[motor_class], float(value))
        for joint, value in episode.get("max_speed_by_joint", {}).items():
            motor_class = DRAFT_JOINT_MOTOR_CLASS.get(joint)
            if motor_class is not None:
                peak_speed[motor_class] = max(peak_speed[motor_class], float(value))
    for index, motor_class in enumerate(DRAFT_MOTOR_CLASSES):
        base_torque, base_speed = DRAFT_BASE_MOTOR_REQUIREMENTS[motor_class]
        theta[4 + index] = max(theta[4 + index], safety_margin * peak_torque[motor_class] / base_torque)
        theta[7 + index] = max(theta[7 + index], safety_margin * peak_speed[motor_class] / base_speed)
    return theta


def _robot_generator():
    try:
        from draft.generation.generator import RobotGenerator
    except ImportError as exc:
        raise RuntimeError(
            "DRAFT support is optional; install with `pip install -e '.[draft]'`"
        ) from exc
    return RobotGenerator


def _fmt(values) -> str:
    return " ".join(f"{float(value):.17g}" for value in np.atleast_1d(values))


def _find_joint(root: ET.Element, name: str) -> ET.Element:
    joint = root.find(f".//joint[@name='{name}']")
    if joint is None:
        raise ValueError(f"DRAFT model is missing required joint {name!r}")
    return joint


def _add_foot_contacts(root: ET.Element, parameters: dict) -> None:
    contact = {
        "type": "capsule",
        "size": "0.015 0.025",
        "group": "0",
        "rgba": "1 0.08 0.03 1",
        "contype": "1",
        "conaffinity": "0",
        "density": "0",
        "condim": "3",
        "priority": "0",
        "friction": "1 0.005 0.0001",
        "solref": "0.02 1",
        "solimp": "0.9 0.95 0.001 0.5 2",
    }
    for side in ("left", "right"):
        body = root.find(f".//body[@name='{side}_foot']")
        if body is None:
            raise ValueError(f"DRAFT model is missing {side}_foot")
        for geom in body.findall("geom"):
            geom.set("contype", "0")
            geom.set("conaffinity", "0")
        for label, local_z in (("front", 0.0), ("rear", -0.08)):
            ET.SubElement(
                body,
                "geom",
                name=f"{side}_foot_{label}_collision",
                fromto=_fmt([-0.009, -0.02, local_z, 0.009, -0.02, local_z]),
                **contact,
            )
        ET.SubElement(
            body, "site", name=f"foot_{side}",
            pos=_fmt([-6.35351e-08, -0.035, -0.0480606]),
        )


def _quat_from_z(vector: np.ndarray) -> np.ndarray:
    """Return a MuJoCo quaternion rotating the local +Z axis to ``vector``."""
    target = np.asarray(vector, dtype=float)
    target /= np.linalg.norm(target)
    source = np.array((0.0, 0.0, 1.0))
    dot = float(np.dot(source, target))
    if dot < -1.0 + 1e-10:
        return np.array((0.0, 1.0, 0.0, 0.0))
    cross = np.cross(source, target)
    quat = np.array((1.0 + dot, cross[0], cross[1], cross[2]))
    return quat / np.linalg.norm(quat)


def _quat_from_matrix(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        quat = np.array((
            0.25 * scale,
            (matrix[2, 1] - matrix[1, 2]) / scale,
            (matrix[0, 2] - matrix[2, 0]) / scale,
            (matrix[1, 0] - matrix[0, 1]) / scale,
        ))
    else:
        diagonal = np.diag(matrix)
        index = int(np.argmax(diagonal))
        next_index = (index + 1) % 3
        last_index = (index + 2) % 3
        scale = 2.0 * np.sqrt(
            max(1.0 + matrix[index, index] - diagonal[next_index] - diagonal[last_index], 0.0)
        )
        quat = np.zeros(4)
        quat[index + 1] = 0.25 * scale
        quat[0] = (matrix[last_index, next_index] - matrix[next_index, last_index]) / scale
        quat[next_index + 1] = (matrix[next_index, index] + matrix[index, next_index]) / scale
        quat[last_index + 1] = (matrix[last_index, index] + matrix[index, last_index]) / scale
    return quat / np.linalg.norm(quat)


def _orient_foot_mass_boxes(xml_path: Path) -> int:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    desired_world = np.asarray(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
    tree = ET.parse(xml_path)
    root = tree.getroot()
    changed = 0
    for side in ("left", "right"):
        geom = root.find(f".//geom[@name='{side}_foot_box']")
        if geom is None:
            continue
        body_id = model.body(f"{side}_foot").id
        body_world = data.xmat[body_id].reshape(3, 3)
        local_rotation = body_world.T @ desired_world
        geom.set("quat", _fmt(_quat_from_matrix(local_rotation)))
        geom.attrib.pop("euler", None)
        changed += 1
    tree.write(xml_path, encoding="unicode")
    return changed


def _repair_generated_geometry(xml_path: Path) -> dict:
    """Align DRAFT's synthetic links and motors with their child joints.

    DRAFT's tree stores the correct body transforms, but its cardinal ``link_axis``
    approximation can draw a link away from the next joint when the source robot
    uses an oblique CAD frame.  The generated lower body has one child per link,
    so the child body's local position is the authoritative joint-to-joint vector.
    Motor positions are deliberately left untouched: LeRobot uses remote linkages,
    so they require a separate source-model placement map rather than joint-centre
    placement.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    repaired = []
    for body in root.findall(".//body"):
        children = body.findall("body")
        if len(children) != 1:
            continue
        child_pos = np.fromstring(children[0].get("pos", "0 0 0"), sep=" ")
        length = float(np.linalg.norm(child_pos))
        if child_pos.shape != (3,) or length <= 1e-9:
            continue
        direction = child_pos / length
        quat = _quat_from_z(direction)
        body_name = body.get("name", "")
        link = body.find(f"geom[@name='{body_name}_link']")
        if link is not None and link.get("type") == "cylinder":
            half_length = float(np.fromstring(link.get("size", "0 0"), sep=" ")[1])
            link.set("pos", _fmt(direction * half_length))
            link.set("quat", _fmt(quat))
            link.attrib.pop("euler", None)
            repaired.append(link.get("name"))
    tree.write(xml_path, encoding="unicode")
    return {"repaired_geoms": repaired, "count": len(repaired)}


def _apply_remote_motor_mounts(xml_path: Path, theta: np.ndarray) -> dict:
    """Scale known LeRobot remote motor mounts with their supporting link."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    thigh_scale = float(theta[0] / DRAFT_NOMINAL_DESIGN[0])
    shin_scale = float(theta[1] / DRAFT_NOMINAL_DESIGN[1])
    applied = []
    for geom_name, (support_body, nominal_pos, scale_name) in REMOTE_MOTOR_MOUNTS.items():
        geom = root.find(f".//geom[@name='{geom_name}']")
        if geom is None:
            continue
        body_id = model.body(support_body).id
        scale = thigh_scale if scale_name == "thigh" else shin_scale
        world_pos = data.xpos[body_id] + data.xmat[body_id].reshape(3, 3) @ (nominal_pos * scale)
        host_id = int(model.geom_bodyid[model.geom(geom_name).id])
        host_local = data.xmat[host_id].reshape(3, 3).T @ (world_pos - data.xpos[host_id])
        geom.set("pos", _fmt(host_local))
        detail_name = f"{geom_name.removesuffix('_motor')}_detail"
        detail = root.find(f".//geom[@name='{detail_name}']")
        if detail is not None:
            detail.set("pos", _fmt(host_local))
        applied.append({"geom": geom_name, "support_body": support_body,
                        "scale": scale_name, "factor": scale})
    tree.write(xml_path, encoding="unicode")
    return {"mounts": applied, "count": len(applied)}


def _add_joint_markers(root: ET.Element) -> int:
    parents = {child: parent for parent in root.iter() for child in parent}
    added = 0
    for joint_name in DRAFT_JOINT_NAMES:
        joint = root.find(f".//joint[@name='{joint_name}']")
        if joint is None or joint not in parents:
            continue
        body = parents[joint]
        ET.SubElement(
            body,
            "geom",
            name=f"{joint_name}_marker",
            type="sphere",
            pos=joint.get("pos", "0 0 0"),
            size=_fmt(JOINT_MARKER_RADIUS),
            density="0",
            contype="0",
            conaffinity="0",
            rgba="0.95 0.75 0.15 1",
        )
        added += 1
    return added


def _remove_decorative_motor_details(root: ET.Element) -> int:
    removed = 0
    for body in root.findall(".//body"):
        for geom in list(body):
            if geom.tag == "geom" and geom.get("name", "").endswith("_detail"):
                body.remove(geom)
                removed += 1
    return removed


def _hide_oversized_foot_visuals(root: ET.Element) -> int:
    hidden = 0
    for body in root.findall(".//body"):
        for geom in body:
            if geom.tag == "geom" and geom.get("name", "").endswith("_foot_box"):
                geom.set("rgba", "0 0 0 0")
                geom.set("contype", "0")
                geom.set("conaffinity", "0")
                hidden += 1
    return hidden


def _mass_calibration(xml_path: Path, cfg: RobotConfig) -> dict | None:
    target = cfg.draft_target_mass_kg
    if target is None:
        return None
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for geom in root.findall(".//geom"):
        density = geom.get("density")
        if density is None:
            continue
        factor = cfg.draft_motor_mass_scale if geom.get("name", "").endswith("_motor") else 1.0
        geom.set("density", _fmt(float(density) * factor))
    tree.write(xml_path, encoding="unicode")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    total_mass = float(model.body_mass.sum())
    motor_mass = 0.0
    for geom in root.findall(".//geom"):
        if not geom.get("name", "").endswith("_motor"):
            continue
        if geom.get("type") != "cylinder":
            raise ValueError(f"Expected cylindrical DRAFT motor geom: {geom.get('name')}")
        radius, half_length = (float(value) for value in geom.get("size").split()[:2])
        motor_mass += np.pi * radius * radius * (2.0 * half_length) * float(geom.get("density"))
    structural_mass = total_mass - motor_mass
    if target <= motor_mass or structural_mass <= 0:
        raise ValueError(
            f"DRAFT mass target {target:.3f} kg is infeasible: "
            f"motor mass {motor_mass:.3f} kg, structural mass {structural_mass:.3f} kg"
        )
    structural_geoms = [
        geom for geom in root.findall(".//geom")
        if geom.get("density") is not None and not geom.get("name", "").endswith("_motor")
    ]
    base_densities = [float(geom.get("density")) for geom in structural_geoms]
    for geom in structural_geoms:
        geom.set("density", "0")
    tree.write(xml_path, encoding="unicode")
    fixed_mass = float(mujoco.MjModel.from_xml_path(str(xml_path)).body_mass.sum())
    for geom, density in zip(structural_geoms, base_densities):
        geom.set("density", _fmt(density))
    tree.write(xml_path, encoding="unicode")
    unscaled_mass = float(mujoco.MjModel.from_xml_path(str(xml_path)).body_mass.sum())
    scalable_structural_mass = unscaled_mass - fixed_mass
    if target <= fixed_mass or scalable_structural_mass <= 0:
        raise ValueError(
            f"DRAFT mass target {target:.3f} kg is infeasible: "
            f"fixed mass {fixed_mass:.3f} kg, scalable structural mass {scalable_structural_mass:.3f} kg"
        )
    structural_scale = (target - fixed_mass) / scalable_structural_mass
    for geom, density in zip(structural_geoms, base_densities):
        geom.set("density", _fmt(density * structural_scale))
    tree.write(xml_path, encoding="unicode")
    calibrated_model = mujoco.MjModel.from_xml_path(str(xml_path))
    return {
        "target_mass_kg": float(target),
        "pre_calibration_mass_kg": total_mass,
        "motor_mass_kg": motor_mass,
        "structural_mass_before_scale_kg": structural_mass,
        "fixed_mass_kg": fixed_mass,
        "scalable_structural_mass_kg": scalable_structural_mass,
        "motor_mass_scale": float(cfg.draft_motor_mass_scale),
        "structural_mass_scale": float(structural_scale),
        "post_calibration_mass_kg": float(calibrated_model.body_mass.sum()),
    }


def _adapt_xml(xml_path: Path, parameters: dict, cfg: RobotConfig) -> None:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    root.set("model", "draft_lerobot_lower_body")
    _remove_decorative_motor_details(root)
    _hide_oversized_foot_visuals(root)
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.attrib.update(
        timestep=_fmt(cfg.timestep), integrator="Euler", solver="Newton",
        iterations=str(cfg.solver_iterations), ls_iterations=str(cfg.ls_iterations),
        gravity="0 0 -9.81",
    )
    for geom in root.findall(".//worldbody//geom"):
        geom.set("conaffinity", "0")
        if geom.get("mesh") is not None or geom.get("type") == "mesh":
            geom.set("contype", "0")
    for target, source in SOURCE_JOINT_NAMES.items():
        joint = _find_joint(root, source)
        joint.set("name", target)
        joint.set("damping", _fmt(cfg.joint_damping))
    _add_joint_markers(root)
    actuator = root.find("actuator")
    if actuator is not None:
        root.remove(actuator)
    actuator = ET.SubElement(root, "actuator")
    for name in DRAFT_JOINT_NAMES:
        joint = _find_joint(root, name)
        force_range = joint.get("actuatorfrcrange")
        if not force_range:
            raise ValueError(f"DRAFT joint {name!r} has no actuator force range")
        ET.SubElement(
            actuator, "motor", name=name, joint=name, ctrllimited="true",
            ctrlrange=force_range, forcelimited="true", forcerange=force_range,
        )
    _add_foot_contacts(root, parameters)
    world = root.find("worldbody")
    ET.SubElement(
        world, "geom", name="ground", type="plane", size="0 0 0.1",
        contype="1", conaffinity="1", condim="3", priority="0",
        friction="1 0.005 0.0001", solref="0.02 1",
        solimp="0.9 0.95 0.001 0.5 2", rgba="0.25 0.3 0.35 1",
    )
    ET.SubElement(world, "light", name="mpc_light", pos="0 -3 4", dir="0 0 -1")
    tree.write(xml_path, encoding="unicode")

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    clearance = float(capsule_bottoms(model, data, foot_capsules(model)).min())
    root_body = root.find(".//body[@name='torso_subassembly']")
    position = np.fromstring(root_body.get("pos", "0 0 0"), sep=" ")
    position[2] += cfg.initial_clearance - clearance
    root_body.set("pos", _fmt(position))
    tree.write(xml_path, encoding="unicode")
    _orient_foot_mass_boxes(xml_path)


def generate_draft_artifact(
    output_dir: str | Path,
    theta=None,
    cfg: RobotConfig = RobotConfig(),
    *,
    strict: bool = True,
) -> DraftArtifact:
    import yaml

    theta = DRAFT_NOMINAL_DESIGN if theta is None else np.asarray(theta, dtype=float)
    if theta.shape not in ((4,), (10,)) or not np.isfinite(theta).all() or np.any(theta <= 0):
        raise ValueError("DRAFT theta requires positive geometry and motor design values")
    motor_requirements = _motor_requirements(theta)
    output_dir = Path(output_dir)
    overrides = {
        "upper_leg_link_length": float(theta[0]),
        "lower_leg_link_length": float(theta[1]),
        "hip_width": float(theta[3]),
    }
    for motor_class, requirement in motor_requirements.items():
        overrides[f"{motor_class}_motor_effort"] = requirement["effort"]
        overrides[f"{motor_class}_motor_velocity"] = requirement["velocity"]
    generator = _robot_generator()(draft_spec_dir())
    generation_log = io.StringIO()
    with redirect_stdout(generation_log), redirect_stderr(generation_log):
        generator.generate(output_dir, params_override=overrides, strict=strict)
    (output_dir / "generation.log").write_text(generation_log.getvalue())
    report_path = output_dir / "feasibility_report.yaml"
    parameters_path = output_dir / "parameters_resolved.yaml"
    report = yaml.safe_load(report_path.read_text())
    parameters = yaml.safe_load(parameters_path.read_text())
    xml_path = output_dir / "draft_lower_body.xml"
    geometry_repair = _repair_generated_geometry(xml_path)
    remote_mounts = _apply_remote_motor_mounts(xml_path, theta)
    _adapt_xml(xml_path, parameters, cfg)
    mass_calibration = _mass_calibration(xml_path, cfg)
    provenance = {
        "theta": theta.tolist(),
        "overrides": overrides,
        "specification": str(draft_spec_dir()),
        "strict": strict,
        "motor_requirements": motor_requirements,
        "geometry_repair": geometry_repair,
        "remote_motor_mounts": remote_mounts,
        "mass_calibration": mass_calibration,
    }
    (output_dir / "dial_adapter.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return DraftArtifact(output_dir, xml_path, report, parameters)


def _cache_root(cfg: RobotConfig) -> Path:
    if cfg.draft_cache_dir:
        return Path(cfg.draft_cache_dir).expanduser().resolve()
    account_home = Path.home()
    cache_base = Path(os.environ.get("XDG_CACHE_HOME", account_home / ".cache"))
    return (cache_base / f"dial_lerobot-{os.geteuid()}" / "draft").resolve()


def _cache_key(theta, cfg: RobotConfig) -> str:
    specification_hash = hashlib.sha256()
    for path in sorted(path for path in draft_spec_dir().rglob("*") if path.is_file()):
        specification_hash.update(str(path.relative_to(draft_spec_dir())).encode())
        specification_hash.update(path.read_bytes())
    payload = {
        "format": CACHE_FORMAT_VERSION,
        "theta": np.asarray(theta, dtype=float).tolist(),
        "specification_sha256": specification_hash.hexdigest(),
        "physics": {
            "timestep": cfg.timestep,
            "solver_iterations": cfg.solver_iterations,
            "ls_iterations": cfg.ls_iterations,
            "joint_damping": cfg.joint_damping,
            "initial_clearance": cfg.initial_clearance,
        },
        "mass_calibration": {
            "target_mass_kg": cfg.draft_target_mass_kg,
            "motor_mass_scale": cfg.draft_motor_mass_scale,
        },
        "strict": cfg.draft_strict,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]


def _load_artifact(output_dir: Path) -> DraftArtifact:
    import yaml

    return DraftArtifact(
        output_dir=output_dir,
        xml_path=output_dir / "draft_lower_body.xml",
        feasibility_report=yaml.safe_load((output_dir / "feasibility_report.yaml").read_text()),
        resolved_parameters=yaml.safe_load((output_dir / "parameters_resolved.yaml").read_text()),
    )


def cached_draft_artifact(theta=None, cfg: RobotConfig = RobotConfig()) -> DraftArtifact:
    theta = DRAFT_NOMINAL_DESIGN if theta is None else np.asarray(theta, dtype=float)
    if theta.shape not in ((4,), (10,)) or not np.isfinite(theta).all() or np.any(theta <= 0):
        raise ValueError("DRAFT theta requires positive geometry and motor design values")
    cache_root = _cache_root(cfg)
    cache_root.mkdir(parents=True, exist_ok=True)
    output_dir = cache_root / _cache_key(theta, cfg)
    if (output_dir / "dial_adapter.json").is_file():
        return _load_artifact(output_dir)

    temporary = Path(tempfile.mkdtemp(prefix="generating-", dir=cache_root))
    try:
        artifact = generate_draft_artifact(temporary, theta, cfg, strict=cfg.draft_strict)
        provenance_path = artifact.output_dir / "dial_adapter.json"
        provenance = json.loads(provenance_path.read_text())
        provenance.update({"cache_key": output_dir.name, "cache_format": CACHE_FORMAT_VERSION})
        provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
        try:
            temporary.rename(output_dir)
        except FileExistsError:
            pass
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return _load_artifact(output_dir)


def draft_model_xml(artifact: DraftArtifact) -> str:
    root = ET.parse(artifact.xml_path).getroot()
    for mesh in root.findall("./asset/mesh"):
        path = Path(mesh.get("file"))
        if not path.is_absolute():
            mesh.set("file", str((artifact.output_dir / path).resolve()))
    return ET.tostring(root, encoding="unicode")


def export_draft_artifact(destination: str | Path, artifact: DraftArtifact) -> Path:
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for source in artifact.output_dir.iterdir():
        if source.is_file():
            shutil.copy2(source, destination / source.name)
    root = ET.parse(artifact.xml_path).getroot()
    for mesh in root.findall("./asset/mesh"):
        mesh.set("file", Path(mesh.get("file")).name)
    xml_path = destination / "model.xml"
    ET.ElementTree(root).write(xml_path, encoding="unicode")
    return xml_path


def describe_draft_model(model, theta, cfg: RobotConfig, artifact: DraftArtifact) -> dict:
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    actuator_checks = artifact.feasibility_report.get("actuator_checks", {})
    return {
        "backend": "draft",
        "source_xml": str(artifact.xml_path),
        "source_xml_sha256": hashlib.sha256(artifact.xml_path.read_bytes()).hexdigest(),
        "artifact_dir": str(artifact.output_dir),
        "theta": np.asarray(theta).tolist(),
        "nominal_theta": draft_nominal_design(
            include_hip_width=True, include_motor_design=len(np.asarray(theta)) == 10
        ).tolist(),
        "design_parameter_names": [
            "thigh_length_m", "shin_length_m", "density_proxy", "hip_width_m",
            "hip_torque_scale", "knee_torque_scale", "ankle_torque_scale",
            "hip_speed_scale", "knee_speed_scale", "ankle_speed_scale",
        ],
        "nq": model.nq,
        "nv": model.nv,
        "nu": model.nu,
        "total_mass_kg": float(model.body_mass.sum()),
        "rest_height_m": float(data.qpos[2]),
        "physics_timestep": model.opt.timestep,
        "control_mode": cfg.control_mode,
        "foot_contact_model": cfg.foot_contact_model,
        "draft_feasibility": {
            "errors": actuator_checks.get("errors", []),
            "warnings": actuator_checks.get("warnings", []),
            "report": artifact.feasibility_report,
        },
        "actuators": [
            {
                "joint": model.actuator(index).name,
                "control_range": model.actuator_ctrlrange[index].tolist(),
                "torque_range_Nm": model.actuator_forcerange[index].tolist(),
            }
            for index in range(model.nu)
        ],
    }


def load_draft_model(artifact: DraftArtifact) -> mujoco.MjModel:
    model = mujoco.MjModel.from_xml_path(str(artifact.xml_path))
    if (model.nq, model.nv, model.nu) != (19, 18, 12):
        raise ValueError(
            f"Expected DRAFT 12-joint lower body; got nq/nv/nu={model.nq}/{model.nv}/{model.nu}"
        )
    return model
