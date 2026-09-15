"""Nominal PPO robot physics, with real (optional) link/mass design changes.

Meshes stay in the zoo; no copied or simplified substitute robot is used.
"""

from copy import deepcopy
import hashlib
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from dial_lerobot.config import RobotConfig

JOINT_NAMES = (
    "hipz_right", "hipz_left", "hipx_right", "hipx_left",
    "hipy_right", "hipy_left", "knee_right", "knee_left",
    "ankley_right", "anklex_right", "ankley_left", "anklex_left",
)
FOOT_SITES = ("foot_left", "foot_right")  # Gait phase order, NOT qpos order.
THIGHS = ("tigh_subassembly", "tigh_subassembly_sym")
SHINS = ("shin_subassembly", "shin_subassembly_sym")
ANKLES = ("ankle_subassembly", "ankle_subassembly_2")
# Unitree H1 lower-body limits from the official DIAL model. The LeRobot
# model has the same 12 lower-body actuator slots, so these are retained as
# the initial limits while geometry and inertias come from LeRobot.
H1_TORQUE_LIMITS = {
    "hipz": 200.0, "hipx": 200.0, "hipy": 200.0,
    "knee": 300.0, "ankle": 40.0,
}
MODEL_RELATIVE = Path("models/bipedal_plateform_no_arms/mjcf/robot.xml")


def source_xml(cfg: RobotConfig = RobotConfig()) -> Path:
    default = Path(__file__).resolve().parents[3] / "lerobot-legged-zoo"
    zoo = Path(cfg.zoo_root or os.environ.get("LEROBOT_ZOO_ROOT", default)).expanduser()
    path = (zoo / MODEL_RELATIVE).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Zoo MJCF missing: {path}. Set LEROBOT_ZOO_ROOT or robot.zoo_root.")
    return path


def _numbers(value: str) -> np.ndarray:
    return np.fromstring(value, sep=" ")


def _fmt(values) -> str:
    return " ".join(f"{float(x):.17g}" for x in np.atleast_1d(values))


def nominal_design(cfg: RobotConfig = RobotConfig()) -> np.ndarray:
    root = ET.parse(source_xml(cfg)).getroot()
    # Anchor-to-anchor distances, including the CAD lateral offsets.
    thigh = np.linalg.norm(_numbers(root.find(f".//body[@name='{SHINS[0]}']").get("pos")))
    shin = np.linalg.norm(_numbers(root.find(f".//body[@name='{ANKLES[0]}']").get("pos")))
    return np.array([thigh, shin, 2.0])


def _scale_link(root: ET.Element, body_name: str, scale: float, mass_scale: float):
    body = root.find(f".//body[@name='{body_name}']")
    inertial = body.find("inertial")
    inertial.set("mass", _fmt(float(inertial.get("mass")) * mass_scale))
    for attr in ("fullinertia", "diaginertia"):
        if attr in inertial.attrib:
            inertial.set(attr, _fmt(_numbers(inertial.get(attr)) * mass_scale * scale**2))
    # Direct child anchors move; descendant segment geometry is scaled separately.
    for item in body:
        for attr in ("pos", "fromto"):
            if attr in item.attrib:
                item.set(attr, _fmt(_numbers(item.get(attr)) * scale))
        if item.tag in ("geom", "site") and "size" in item.attrib:
            item.set("size", _fmt(_numbers(item.get("size")) * scale))
        if item.tag == "geom" and item.get("mesh") and scale != 1.0:
            asset = root.find("asset")
            original_name = item.get("mesh")
            clone_name = f"{original_name}__{body_name}"
            if asset.find(f"mesh[@name='{clone_name}']") is None:
                original = asset.find(f"mesh[@name='{original_name}']")
                clone = deepcopy(original)
                clone.set("name", clone_name)
                clone.set("scale", _fmt(_numbers(clone.get("scale", "1 1 1")) * scale))
                asset.append(clone)
            item.set("mesh", clone_name)


def model_xml(theta=None, cfg: RobotConfig = RobotConfig()) -> str:
    path = source_xml(cfg)
    root = ET.parse(path).getroot()
    root.set("model", "lerobot_lower_body_dial_mpc")
    compiler = root.find("compiler")
    asset_dir = (path.parent / compiler.get("meshdir", "")).resolve()
    compiler.set("meshdir", str(asset_dir))
    compiler.set("inertiafromgeom", "false")  # Preserve the CAD inertial tensors.
    for mesh in root.findall("./asset/mesh"):
        mesh.set("name", mesh.get("name", Path(mesh.get("file")).stem))
        mesh.set("file", str(asset_dir / mesh.get("file")))
    nominal = nominal_design(cfg)
    theta = nominal if theta is None else np.asarray(theta, dtype=float)
    if theta.shape != (3,) or not np.isfinite(theta).all() or np.any(theta <= 0):
        raise ValueError("theta must contain three finite positive values: thigh, shin, density proxy")
    for names, length, reference in zip((THIGHS, SHINS), theta[:2], nominal[:2]):
        scale = float(length / reference)
        mass_scale = scale * float(theta[2] / nominal[2])
        for name in names:
            _scale_link(root, name, scale, mass_scale)

    # Settings from the zoo PPO articulation, independent of mjlab/Torch imports.
    natural_freq = 10 * 2 * 3.1415926535
    actuator = ET.SubElement(root, "actuator")
    for name in JOINT_NAMES:
        joint = root.find(f".//joint[@name='{name}']")
        armature = 0.02 if name.startswith("hip") else (0.015 if name.startswith("knee") else 0.01)
        effort = next(limit for prefix, limit in H1_TORQUE_LIMITS.items() if name.startswith(prefix))
        joint.set("armature", _fmt(armature))
        joint.set("frictionloss", "0")
        joint.set("damping", "0")
        limits = _numbers(joint.get("range"))
        if cfg.control_mode == "torque":
            ET.SubElement(actuator, "motor", name=name, joint=name,
                          ctrllimited="true", ctrlrange=_fmt([-effort, effort]),
                          forcelimited="true", forcerange=_fmt([-effort, effort]))
        else:
            midpoint = limits.mean()
            soft = midpoint + cfg.soft_joint_limit_factor * (limits - midpoint)
            ET.SubElement(actuator, "position", name=name, joint=name,
                          kp=_fmt(armature * natural_freq**2),
                          kv=_fmt(4 * armature * natural_freq),
                          ctrllimited="true", ctrlrange=_fmt(soft),
                          forcelimited="true", forcerange=_fmt([-effort, effort]))
    for geom in root.findall(".//geom"):
        name = geom.get("name", "")
        if name.endswith("_collision"):
            geom.set("condim", "1")
            if name.startswith(("left_foot", "right_foot")):
                geom.attrib.update(condim="3", priority="1", friction="0.6 0.005 0.0001",
                                   solref="0.005 1", solimp="0.995 0.9995 0.001 0.5 2")
    ET.SubElement(root, "option", timestep=_fmt(cfg.timestep), integrator="Euler",
                  solver="Newton", iterations=str(cfg.solver_iterations),
                  ls_iterations=str(cfg.ls_iterations), gravity="0 0 -9.81")
    world = root.find("worldbody")
    ET.SubElement(world, "geom", name="ground", type="plane", size="0 0 0.1",
                  friction="1.2 0.005 0.0001", solref="0.01 1",
                  solimp="0.99 0.999 0.001 0.5 2", rgba="0.25 0.3 0.35 1", condim="3")
    ET.SubElement(world, "light", name="mpc_light", pos="0 -3 4", dir="0 0 -1")
    # Set the neutral base height so the lower sole site clears the flat floor.
    temp_model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    temp_data = mujoco.MjData(temp_model)
    mujoco.mj_forward(temp_model, temp_data)
    clearance = min(temp_data.site_xpos[temp_model.site(s).id, 2] for s in FOOT_SITES)
    base = root.find(".//body[@name='torso_subassembly']")
    pos = _numbers(base.get("pos"))
    pos[2] += cfg.initial_clearance - clearance
    base.set("pos", _fmt(pos))
    return ET.tostring(root, encoding="unicode")


def build_model(theta=None, cfg: RobotConfig = RobotConfig()) -> mujoco.MjModel:
    model = mujoco.MjModel.from_xml_string(model_xml(theta, cfg))
    if (model.nq, model.nv, model.nu) != (19, 18, 12):
        raise ValueError(f"Expected the 12-joint lower body; got nq/nv/nu={model.nq}/{model.nv}/{model.nu}")
    return model


def initial_data(model: mujoco.MjModel, height_offset: float = 0.0) -> mujoco.MjData:
    data = mujoco.MjData(model)
    data.qpos[2] += height_offset
    mujoco.mj_forward(model, data)
    return data


def describe_model(model, theta, cfg):
    path = source_xml(cfg)
    data = initial_data(model)
    return {
        "source_xml": str(path), "source_xml_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "theta": np.asarray(theta).tolist(), "nominal_theta": nominal_design(cfg).tolist(),
        "nq": model.nq, "nv": model.nv, "nu": model.nu,
        "total_mass_kg": float(model.body_mass.sum()), "rest_height_m": float(data.qpos[2]),
        "foot_height_m": {s: float(data.site_xpos[model.site(s).id, 2]) for s in FOOT_SITES},
        "physics_timestep": model.opt.timestep,
        "control_mode": cfg.control_mode,
        "actuators": [{"joint": name, "type": int(model.actuator_biastype[i]),
                       "kp": float(model.actuator_gainprm[i, 0]),
                       "kd": float(-model.actuator_biasprm[i, 2]),
                       "control_range": model.actuator_ctrlrange[i].tolist(),
                       "torque_range_Nm": model.actuator_forcerange[i].tolist()}
                      for i, name in enumerate(JOINT_NAMES)],
    }
