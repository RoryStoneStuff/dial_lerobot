"""Measure forward locomotion under fixed commands without viewer overhead."""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--task", default="Mjlab-Velocity-Flat-LeRobot-Humanoid-no-arms-Gait")
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--speeds", type=float, nargs="+", default=[0.3, 0.5, 0.7])
  parser.add_argument("--seconds", type=float, default=20)
  parser.add_argument("--num-envs", type=int, default=8)
  parser.add_argument("--output", type=Path)
  parser.add_argument("--save-poses", action="store_true", help="Save first environment poses for offline video.")
  args = parser.parse_args()
  cfg = load_env_cfg(args.task, play=True)
  cfg.scene.num_envs = args.num_envs
  cfg.episode_length_s = args.seconds + 1
  cfg.seed = 42
  cfg.events["reset_base"].params["pose_range"]["yaw"] = (0, 0)
  command = cfg.commands["twist"]
  command.rel_standing_envs = 0.0
  command.init_velocity_prob = 0.0
  command.ranges.lin_vel_y = (0, 0)
  command.ranges.ang_vel_z = (0, 0)
  env = ManagerBasedRlEnv(cfg, device="cuda:0")
  results = []
  try:
    runner = load_runner_cls(args.task)(
      RslRlVecEnvWrapper(env, clip_actions=1), asdict(load_rl_cfg(args.task)), None, "cuda:0"
    )
    runner.load(str(args.checkpoint), load_optimizer=False)
    policy = runner.get_inference_policy(device="cuda:0")
    with torch.inference_mode():
      for speed in args.speeds:
        command.ranges.lin_vel_x = (speed, speed)
        obs, _ = env.reset(seed=42)
        robot = env.scene["robot"]
        start_x = robot.data.root_link_pos_w[:, 0].clone()
        fallen = torch.zeros(args.num_envs, dtype=torch.bool, device=env.device)
        sums = torch.zeros((args.num_envs, 4), device=env.device)
        counts = torch.zeros(args.num_envs, device=env.device)
        landings = torch.zeros((args.num_envs, 2), device=env.device)
        previous = None
        poses = []
        torch.cuda.synchronize()
        start = time.perf_counter()
        for step in range(round(args.seconds / env.step_dt)):
          obs, reward, done, truncated, _ = env.step(policy(obs).clamp(-1, 1))
          assert torch.isfinite(reward).all()
          assert all(torch.isfinite(x).all() for x in obs.values())
          fallen |= done | truncated
          contacts = env.scene["feet_ground_contact"].data.found > 0
          actual_command = env.command_manager.get_command("twist")
          assert torch.allclose(actual_command[:, 0], torch.full_like(actual_command[:, 0], speed))
          if step >= round(2 / env.step_dt):
            alive = ~fallen
            velocity = robot.data.root_link_lin_vel_b[:, 0]
            sums += torch.stack([
              velocity, (velocity - speed).abs(), contacts.all(-1).float(),
              (contacts.sum(-1) == 1).float(),
            ], dim=1) * alive[:, None]
            counts += alive
            if previous is not None:
              landings += (contacts & ~previous & alive[:, None]).float()
          previous = contacts.clone()
          if args.save_poses and step % 2 == 0:
            poses.append(env.sim.data.qpos[0].cpu().numpy().copy())
        torch.cuda.synchronize()
        duration = time.perf_counter() - start
        survivors = ~fallen
        dx = robot.data.root_link_pos_w[:, 0] - start_x
        means = (sums.sum(0) / counts.sum().clamp(min=1)).tolist()
        result = {
          "command_mps": speed,
          "survivors": int(survivors.sum()),
          "num_envs": args.num_envs,
          "actual_forward_mps_before_first_fall_after_2s": means[0],
          "mean_abs_speed_error_mps": means[1],
          "double_support_fraction": means[2],
          "single_support_fraction": means[3],
          "survivor_displacements_m": dx[survivors].tolist(),
          "landings_per_foot_before_first_fall": landings.tolist(),
          "wall_seconds": duration,
        }
        results.append(result)
        print(json.dumps(result), flush=True)
        if poses:
          np.savez_compressed(args.checkpoint.with_suffix(f".speed_{speed}.poses.npz"), qpos=np.array(poses), dt=env.step_dt * 2)
  finally:
    env.close()
  output = args.output or args.checkpoint.with_suffix(".locomotion.json")
  output.write_text(json.dumps({"task": args.task, "checkpoint": str(args.checkpoint), "seconds": args.seconds, "results": results}, indent=2) + "\n")
  print(f"Evaluation saved: {output}")


if __name__ == "__main__":
  main()
