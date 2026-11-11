"""Check the lower-body model, GPU stepping, and optional checkpoint playback.

Run from the repository root with: uv run --no-sync python tools/check_lower_body.py
"""

import argparse
import json
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import mujoco
import numpy as np
import torch

import mjlab.tasks  # noqa: F401 -- discover the zoo's registered tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.scene import Scene
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

TASK = "Mjlab-Velocity-Flat-LeRobot-Humanoid-no-arms"


def check_model():
  cfg = load_env_cfg(TASK)
  cfg.scene.num_envs = 1
  scene = Scene(cfg.scene, device="cpu")
  model = scene.compile()
  data = mujoco.MjData(model)
  data.qpos[2] = cfg.scene.entities["robot"].init_state.pos[2]
  mujoco.mj_forward(model, data)
  hinges = model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE
  assert model.nu == int(hinges.sum()) == 12
  assert model.jnt_limited[hinges].all()
  assert np.all(model.jnt_range[hinges, 0] < model.jnt_range[hinges, 1])
  assert np.all(model.actuator_forcelimited)
  assert np.all(np.isfinite(model.actuator_forcerange))
  clearance = {}
  for i in range(model.ngeom):
    name = model.geom(i).name
    if "foot" not in name or "collision" not in name:
      continue
    assert model.geom_type[i] == mujoco.mjtGeom.mjGEOM_CAPSULE
    radius, half_length = model.geom_size[i, :2]
    clearance[name] = float(
      data.geom_xpos[i, 2] - radius
      - half_length * abs(data.geom_xmat[i].reshape(3, 3)[2, 2])
    )
  assert len(clearance) == 12 and min(clearance.values()) > 0
  assert not data.warning.number.any()
  return {
    "mass_kg": float(model.body_mass.sum()),
    "actions": model.nu,
    "limited_hinges": int(model.jnt_limited[hinges].sum()),
    "force_ranges_nm": model.actuator_forcerange.tolist(),
    "nominal_foot_clearance_m": clearance,
    "control_dt": cfg.decimation * cfg.sim.mujoco.timestep,
  }


def check_gpu(checkpoint, steps):
  assert torch.cuda.is_available(), "A CUDA GPU is required."
  cfg = load_env_cfg(TASK)
  cfg.scene.num_envs = 8
  cfg.seed = 42
  # Exercise timeout resets without needing a long rollout.
  cfg.episode_length_s = 0.4
  cfg.sim.nan_guard.enabled = True
  env = ManagerBasedRlEnv(cfg, device="cuda:0")
  try:
    obs, _ = env.reset(seed=42)
    robot = env.scene["robot"]
    limits = robot.data.soft_joint_pos_limits
    action = env.action_manager.get_term("joint_pos")
    # Explicitly stress the action mapping including encoder bias.
    for sign in (-1, 1):
      action.process_actions(torch.full((8, 12), sign * 100.0, device=env.device))
      action.apply_actions()
      targets = robot.data.joint_pos_target
      assert torch.all(targets >= limits[..., 0] - 1e-6)
      assert torch.all(targets <= limits[..., 1] + 1e-6)

    inference = None
    if checkpoint:
      wrapped = RslRlVecEnvWrapper(env, clip_actions=load_rl_cfg(TASK).clip_actions)
      runner_cls = load_runner_cls(TASK)
      runner = runner_cls(wrapped, asdict(load_rl_cfg(TASK)), None, "cuda:0")
      runner.load(str(checkpoint), load_optimizer=False)
      inference = runner.get_inference_policy(device="cuda:0")

    obs, _ = env.reset(seed=42)
    terminations = timeouts = 0
    contact_seen = torch.zeros(2, dtype=torch.bool, device=env.device)
    with torch.inference_mode():
      for _ in range(steps):
        commands = inference(obs) if inference else torch.rand((8, 12), device=env.device) * 2 - 1
        commands = commands.clamp(-1, 1)
        obs, rewards, terminated, truncated, _ = env.step(commands)
        assert all(torch.isfinite(value).all() for value in obs.values())
        assert torch.isfinite(rewards).all()
        assert torch.isfinite(env.sim.data.qpos).all()
        terminations += int(terminated.sum())
        timeouts += int(truncated.sum())
        contact_seen |= (env.scene["feet_ground_contact"].data.found > 0).any(dim=0)
        # Reset-time curricula must not silently replace the configured commands.
        ranges = env.command_manager.get_term("twist").cfg.ranges
        assert ranges.lin_vel_x == cfg.commands["twist"].ranges.lin_vel_x
        assert "command_vel" not in cfg.curriculum
    assert timeouts > 0, "No timeout resets exercised."
    assert contact_seen.all(), "Both feet must register ground contact."

    # Verify that a collapsed, still-upright robot terminates and resets.
    pose = robot.data.root_link_pose_w.clone()
    pose[:, 2] = 0.3
    robot.write_root_link_pose_to_sim(pose)
    _, _, fallen, _, _ = env.step(torch.zeros((8, 12), device=env.device))
    assert fallen.all(), "Low-base termination failed."
    return {
      "gpu": torch.cuda.get_device_name(),
      "steps_per_env": steps,
      "observation_shapes": {key: list(value.shape) for key, value in obs.items()},
      "terminations": terminations,
      "timeouts": timeouts,
      "both_feet_contact": bool(contact_seen.all()),
      "extreme_targets_bounded": True,
      "low_base_reset": True,
      "checkpoint": str(checkpoint) if checkpoint else None,
    }
  finally:
    env.close()


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path)
  parser.add_argument("--steps", type=int, default=100)
  parser.add_argument("--output", type=Path, default=Path("logs/lower_body_checks.json"))
  args = parser.parse_args()
  if args.steps < 40:
    parser.error("Use at least 40 steps to exercise contact and timeout resets.")
  result = {
    "task": TASK,
    "versions": {p: version(p) for p in ("mjlab", "mujoco", "mujoco-warp", "rsl-rl-lib", "torch")},
    "model": check_model(),
    "gpu": check_gpu(args.checkpoint, args.steps),
  }
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(result, indent=2) + "\n")
  print(f"Checks passed: {args.output}")


if __name__ == "__main__":
  main()
