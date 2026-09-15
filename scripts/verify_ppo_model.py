"""Run in the zoo PPO environment to compare compiled nominal robot physics.

PYTHONPATH=/path/to/lerobot_mpc /path/to/zoo/.venv/bin/python scripts/verify_ppo_model.py
"""

import json
from pathlib import Path
import sys

import mujoco
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT.parent / "lerobot-legged-zoo"))

from mjlab.entity import Entity
from training_exemples.lerobot_humanoid_no_arms.lerobot_humanoid_no_arms_constants import get_lerobot_humanoid_no_arms_robot_cfg
from dial_lerobot.assets.model_builder import build_model, initial_data, JOINT_NAMES


def main():
    ppo = Entity(get_lerobot_humanoid_no_arms_robot_cfg()).spec.compile()
    mpc = build_model()
    checked = []
    for field in ("body_mass", "body_inertia", "body_ipos", "body_iquat", "body_quat",
                  "jnt_axis", "jnt_pos", "jnt_range", "dof_armature", "dof_damping",
                  "dof_frictionloss", "actuator_gainprm", "actuator_biasprm",
                  "actuator_forcerange", "actuator_trnid", "actuator_gear"):
        np.testing.assert_allclose(getattr(mpc, field), getattr(ppo, field), atol=1e-10)
        checked.append(field)
    # Compare robot geoms separately: the MPC scene also includes a floor.
    for field in ("geom_type", "geom_bodyid", "geom_pos", "geom_quat", "geom_size",
                  "geom_contype", "geom_conaffinity", "geom_condim", "geom_priority",
                  "geom_friction", "geom_solref", "geom_solimp"):
        robot_geoms = np.flatnonzero(mpc.geom_bodyid != 0)
        np.testing.assert_allclose(getattr(mpc, field)[robot_geoms], getattr(ppo, field), atol=1e-9)
        checked.append(field)
    assert [mpc.actuator(i).name for i in range(mpc.nu)] == list(JOINT_NAMES)
    ppo_data, mpc_data = mujoco.MjData(ppo), initial_data(mpc)
    ppo_data.qpos[2] = 0.75
    mujoco.mj_forward(ppo, ppo_data)
    np.testing.assert_allclose(mpc_data.xpos, ppo_data.xpos, atol=2e-8)
    np.testing.assert_allclose(mpc_data.site_xpos, ppo_data.site_xpos, atol=2e-8)
    # Same bounded target gives same actuator torque at the same joint state.
    target = np.random.default_rng(42).uniform(mpc.actuator_ctrlrange[:, 0], mpc.actuator_ctrlrange[:, 1])
    ppo_data.ctrl[:], mpc_data.ctrl[:] = target, target
    mujoco.mj_forward(ppo, ppo_data)
    mujoco.mj_forward(mpc, mpc_data)
    np.testing.assert_allclose(mpc_data.actuator_force, ppo_data.actuator_force, atol=1e-10)
    print(json.dumps({"passed": True, "compared_fields": checked,
                      "total_mass_kg": float(mpc.body_mass.sum()),
                      "rest_height_m": float(mpc_data.qpos[2]), "actuated_joints": mpc.nu}, indent=2))


if __name__ == "__main__":
    main()
