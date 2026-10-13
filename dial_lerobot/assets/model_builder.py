"""Nominal PPO robot physics, with real (optional) link/mass design changes.

Meshes stay in the zoo. The optional physics-only build removes disabled
visual meshes while retaining all collision shapes and explicit inertials.
"""

from copy import deepcopy
import hashlib
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from dial_lerobot.config import RobotConfig
from dial_lerobot.assets.foot_geometry import foot_capsules, capsule_bottoms

JOINT_NAMES = (
    "hipz_right", "hipz_left", "hipx_right", "hipx_left",
    "hipy_right", "hipy_left", "knee_right", "knee_left",
    "ankley_right", "anklex_right", "ankley_left", "anklex_left",
)
FOOT_SITES = ("foot_left", "foot_right")  # Gait phase order, NOT qpos order.
THIGHS = ("tigh_subassembly", "tigh_subassembly_sym")
SHINS = ("shin_subassembly", "shin_subassembly_sym")
ANKLES = ("ankle_subassembly", "ankle_subassembly_2")
HIP_ROOTS = ("hipx_subassembly", "hipx_subassemby_sym")  # Right, left yaw anchors.
# Unitree H1 lower-body limits from the official DIAL model. The LeRobot
# limits are mapped by joint family to LeRobot's 12 lower-body joints.
# Geometry and inertias come from LeRobot.
H1_TORQUE_LIMITS = {
    "hipz": 200.0, "hipx": 200.0, "hipy": 200.0,
    "knee": 300.0, "ankle": 40.0,
}
MODEL_RELATIVE = Path("models/bipedal_plateform_no_arms/mjcf/robot.xml")


def source_xml(cfg: RobotConfig = RobotConfig()) -> Path:
    default = Path(__file__).resolve().parents[3] / "lerobot-legged-zoo"
    bundled = Path(__file__).resolve().parents[2] / "external" / "lerobot-legged-zoo"
    if (bundled / MODEL_RELATIVE).is_file():
        default = bundled
    zoo = Path(cfg.zoo_root or os.environ.get("LEROBOT_ZOO_ROOT", default)).expanduser()
    path = (zoo / MODEL_RELATIVE).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Zoo MJCF missing: {path}. Set LEROBOT_ZOO_ROOT or robot.zoo_root.")
    return path


def _numbers(value: str) -> np.ndarray:
    return np.fromstring(value, sep=" ")


def _fmt(values) -> str:
    return " ".join(f"{float(x):.17g}" for x in np.atleast_1d(values))


def nominal_design(cfg: RobotConfig = RobotConfig(), include_hip_width=False) -> np.ndarray:
    root = ET.parse(source_xml(cfg)).getroot()
    # Anchor-to-anchor distances, including the CAD lateral offsets.
    thigh = np.linalg.norm(_numbers(root.find(f".//body[@name='{SHINS[0]}']").get("pos")))
    shin = np.linalg.norm(_numbers(root.find(f".//body[@name='{ANKLES[0]}']").get("pos")))
    values = [thigh, shin, 2.0]
    if include_hip_width:
        hips = [_numbers(root.find(f".//body[@name='{name}']").get("pos")) for name in HIP_ROOTS]
        values.append(float(hips[1][1] - hips[0][1]))
    return np.array(values)


def _set_hip_width(root, width):
    """Translate complete legs in the torso's lateral frame about their midpoint.

    The torso CAD geometry/inertia is fixed. Segment masses and inertias remain
    local to their translated bodies; MuJoCo recomputes whole-body quantities.
    """
    bodies = [root.find(f".//body[@name='{name}']") for name in HIP_ROOTS]
    positions = [_numbers(body.get("pos")) for body in bodies]
    center = 0.5 * (positions[0][1] + positions[1][1])
    for body, position, sign in zip(bodies, positions, (-1, 1)):
        position[1] = center + sign * width / 2
        body.set("pos", _fmt(position))


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



def _replace_with_h1_pair_contacts(root: ET.Element):
    """Replace each LeRobot foot's six capsules with two H1-style rails.

    The H1 model uses two parallel capsules as its support points. The
    LeRobot foot frame has a different orientation and size, so the rails are
    placed in the LeRobot frame: they run along local ``x`` (the horizontal
    sole direction), are separated along local ``z`` (the forward/back
    direction after the LeRobot foot rotation), and sit at the same
    lowest local ``z`` as the native sole capsules. Their red, group-0 styling
    makes them visible in the MuJoCo viewer for geometry inspection.
    """
    parents = {child: parent for parent in root.iter() for child in parent}
    for side in ("left", "right"):
        site = root.find(f".//site[@name='foot_{side}']")
        if site is None or site not in parents:
            raise ValueError(f"Could not locate foot_{side} body in LeRobot XML")
        body = parents[site]
        old = [child for child in list(body)
               if child.tag == "geom"
               and child.get("name", "").startswith(f"{side}_foot")
               and child.get("name", "").endswith("_collision")]
        if len(old) != 6:
            raise ValueError(f"Expected six native {side} foot capsules, found {len(old)}")
        for geom in old:
            body.remove(geom)
        # The CAD foot is roughly 150 mm long by 48 mm wide in the
        # neutral world pose. Place rails near its front/rear ends.
        for name, x in (("front", 0.0), ("rear", -0.08)):
            ET.SubElement(
                body,
                "geom",
                name=f"{side}_foot_{name}_collision",
                **{
                    "class": "collision",
                    "type": "capsule",
                    "size": "0.015 0.025",
                    "fromto": _fmt([-0.009, -0.02, x, 0.009, -0.02, x]),
                    "group": "0",
                    "rgba": "1 0.08 0.03 1",
                },
            )


    # Disable the source XMLs misaligned foot/ankle capsule proxies in h1_pair mode.
    for geom in root.findall(".//geom"):
        name = geom.get("name", "")
        if (name.endswith("_collision") and not name.startswith(("left_foot_", "right_foot_"))
                and geom.get("type") == "capsule"
                and any(token in name.lower() for token in ("ankle", "foot", "femur"))):
            geom.set("contype", "0")
            geom.set("conaffinity", "0")

def model_xml(theta=None, cfg: RobotConfig = RobotConfig(), *, include_visual=True) -> str:
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
    if theta.shape not in ((3,), (4,)) or not np.isfinite(theta).all() or np.any(theta <= 0):
        raise ValueError("theta requires thigh, shin, density proxy, and optional hip width; all finite and positive")
    for names, length, reference in zip((THIGHS, SHINS), theta[:2], nominal[:2]):
        scale = float(length / reference)
        mass_scale = (1.0 if cfg.preserve_link_masses
                      else scale * float(theta[2] / nominal[2]))
        for name in names:
            _scale_link(root, name, scale, mass_scale)

    if len(theta) == 4:
        _set_hip_width(root, float(theta[3]))

    if cfg.foot_contact_model == "h1_pair":
        _replace_with_h1_pair_contacts(root)

    # Settings from the zoo PPO articulation, independent of mjlab/Torch imports.
    natural_freq = 10 * 2 * 3.1415926535
    actuator = ET.SubElement(root, "actuator")
    for name in JOINT_NAMES:
        joint = root.find(f".//joint[@name='{name}']")
        # Match the H1 MJCF actuator-side joint dynamics: every actuated
        # joint has 0.1 kg m^2 of armature. Geometry and joint limits remain
        # LeRobot-specific.
        armature = 0.1
        effort = next(limit for prefix, limit in H1_TORQUE_LIMITS.items() if name.startswith(prefix))
        joint.set("armature", _fmt(armature))
        joint.set("frictionloss", "0")
        joint.set("damping", _fmt(cfg.joint_damping))
        limits = _numbers(joint.get("range"))
        if cfg.control_mode == "torque":
            # Keep MuJoCo's actuator interface in physical Nm. The DIAL task
            # maps normalized actions through the H1-style PD controller.
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
    # Match the compiled defaults of DIAL's mjx_scene_h1_loco.xml.
    # Use the same parameters on both sides of ground contacts so mixing
    # cannot reintroduce the previous stiff, high-priority foot settings.
    contact_parameters = dict(condim="3", priority="0", friction="1 0.005 0.0001",
                              solref="0.02 1", solimp="0.9 0.95 0.001 0.5 2")
    for geom in root.findall(".//geom"):
        if geom.get("name", "").endswith("_collision"):
            geom.attrib.update(contact_parameters, conaffinity="0")
            # Retain contype=0 on disabled proxies. Enabled robot colliders
            # inherit contype=1 and collide with the ground (affinity=1),
            # but not with one another, as in H1.
    ET.SubElement(root, "option", timestep=_fmt(cfg.timestep), integrator="Euler",
                  solver="Newton", iterations=str(cfg.solver_iterations),
                  ls_iterations=str(cfg.ls_iterations), gravity="0 0 -9.81")
    world = root.find("worldbody")
    ET.SubElement(world, "geom", name="ground", type="plane", size="0 0 0.1",
                  contype="1", conaffinity="1", rgba="0.25 0.3 0.35 1",
                  **contact_parameters)
    ET.SubElement(world, "light", name="mpc_light", pos="0 -3 4", dir="0 0 -1")
    if not include_visual:
        # Preserve geom ordering while removing non-colliding CAD assets.
        # Inertia comes from explicit body inertials, never these placeholders.
        for geom in root.findall(".//geom"):
            if geom.get("mesh") is not None:
                if geom.get("class") != "visual":
                    raise ValueError("Only visual mesh geometry can be removed from planning")
                geom.attrib.pop("mesh")
                geom.attrib.update(type="sphere", size="0.001", contype="0", conaffinity="0")
        for mesh in list(root.find("asset").findall("mesh")):
            root.find("asset").remove(mesh)
    # Set neutral clearance using the lowest collision surface, not a marker.
    temp_model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    temp_data = mujoco.MjData(temp_model)
    mujoco.mj_forward(temp_model, temp_data)
    clearance = float(capsule_bottoms(temp_model, temp_data, foot_capsules(temp_model)).min())
    base = root.find(".//body[@name='torso_subassembly']")
    pos = _numbers(base.get("pos"))
    pos[2] += cfg.initial_clearance - clearance
    base.set("pos", _fmt(pos))
    return ET.tostring(root, encoding="unicode")


def build_model(theta=None, cfg: RobotConfig = RobotConfig(), *, include_visual=True) -> mujoco.MjModel:
    model = mujoco.MjModel.from_xml_string(model_xml(theta, cfg, include_visual=include_visual))
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
        "theta": np.asarray(theta).tolist(), "nominal_theta": nominal_design(cfg, len(theta) == 4).tolist(),
        "design_parameter_names": ["thigh_length_m", "shin_length_m", "density_proxy"]
                                  + (["hip_width_m"] if len(theta) == 4 else []),
        "hip_width_m": float(model.body_pos[model.body(HIP_ROOTS[1]).id, 1]
                              - model.body_pos[model.body(HIP_ROOTS[0]).id, 1]),
        "nq": model.nq, "nv": model.nv, "nu": model.nu,
        "total_mass_kg": float(model.body_mass.sum()), "rest_height_m": float(data.qpos[2]),
        "foot_height_m": {s: float(data.site_xpos[model.site(s).id, 2]) for s in FOOT_SITES},
        "physics_timestep": model.opt.timestep,
        "control_mode": cfg.control_mode,
        "foot_contact_model": cfg.foot_contact_model,
        "foot_collision_capsules": {
            side: [model.geom(i).name for i in foot_capsules(model)[index]]
            for index, side in enumerate(("left", "right"))
        },
        "actuators": [{"joint": name, "type": int(model.actuator_biastype[i]),
                       "kp": float(model.actuator_gainprm[i, 0]),
                       "kd": float(-model.actuator_biasprm[i, 2]),
                       "control_range": model.actuator_ctrlrange[i].tolist(),
                       "torque_range_Nm": model.actuator_forcerange[i].tolist()}
                      for i, name in enumerate(JOINT_NAMES)],
    }
