"""Flat-ground gait experiment informed by the local Berkeley biped PPO task."""

import torch

from mjlab.envs import mdp as env_mdp
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat

from .env_cfgs import lerobot_humanoid_no_arms_flat_env_cfg
from .rl_cfg import lerobot_humanoid_no_arms_ppo_runner_cfg


def track_planar_velocity(env, std: float = 0.25):
  robot = env.scene["robot"]
  velocity = quat_apply_inverse(yaw_quat(robot.data.root_link_quat_w), robot.data.root_link_lin_vel_w)
  command = env.command_manager.get_command("twist")
  error = torch.sum((velocity[:, :2] - command[:, :2]) ** 2, dim=1)
  env.extras["log"]["Locomotion/actual_forward_mps"] = velocity[:, 0].mean()
  env.extras["log"]["Locomotion/command_forward_mps"] = command[:, 0].mean()
  env.extras["log"]["Locomotion/planar_error_mps"] = error.sqrt().mean()
  return torch.exp(-error / std**2)


def biped_air_time(env, threshold: float = 0.4):
  """Berkeley's positive biped air-time reward, adapted to MJLab contact data."""
  sensor = env.scene["feet_ground_contact"].data
  air = sensor.current_air_time
  contact = sensor.current_contact_time
  in_contact = contact > 0.0
  single = in_contact.sum(dim=1) == 1
  duration = torch.where(in_contact, contact, air)
  reward = torch.where(single, duration.min(dim=1).values, 0.0).clamp(max=threshold)
  command = env.command_manager.get_command("twist")
  moving = command[:, :2].norm(dim=1) > 0.1
  env.extras["log"]["Locomotion/single_support_fraction"] = single.float().mean()
  return reward * moving


def lateral_joint_deviation(env):
  robot = env.scene["robot"]
  ids = [i for i, n in enumerate(robot.joint_names) if n.startswith(("hipz_", "hipx_", "anklex_"))]
  return (robot.data.joint_pos[:, ids] - robot.data.default_joint_pos[:, ids]).abs().sum(dim=1)


def gait_env_cfg(play: bool = False):
  cfg = lerobot_humanoid_no_arms_flat_env_cfg(play)
  command = cfg.commands["twist"]
  command.ranges.lin_vel_x = (0.3, 0.7)
  command.ranges.ang_vel_z = (-0.3, 0.3)
  command.rel_standing_envs = 0.02
  command.resampling_time_range = (10.0, 10.0)
  # No externally injected command velocity: propulsion must come from the policy.
  command.init_velocity_prob = 0.0

  cfg.rewards["track_linear_velocity"] = RewardTermCfg(func=track_planar_velocity, weight=4.0)
  cfg.rewards["track_angular_velocity"].weight = 1.0
  cfg.rewards["track_angular_velocity"].params["std"] = 0.25
  cfg.rewards["upright"] = RewardTermCfg(func=env_mdp.flat_orientation_l2, weight=-2.0)
  cfg.rewards.pop("pose")
  cfg.rewards["lateral_joint_deviation"] = RewardTermCfg(func=lateral_joint_deviation, weight=-0.2)
  cfg.rewards["termination"] = RewardTermCfg(func=env_mdp.is_terminated, weight=-10.0)
  cfg.rewards["air_time"] = RewardTermCfg(func=biped_air_time, weight=1.0)
  cfg.rewards["action_rate_l2"].weight = -0.01
  cfg.rewards["angular_momentum"].weight = 0.0
  cfg.rewards["actuator_torque_l2"].weight = -2e-4
  cfg.rewards["ankle_torque_l2"].weight = -1e-4
  cfg.rewards["ankle_power_l1"].weight = -1e-4
  cfg.rewards.pop("ankle_torque_over_6nm_l2")
  cfg.rewards["foot_clearance"].weight = -0.5
  cfg.rewards["foot_clearance"].params["target_height"] = 0.06
  cfg.rewards["foot_swing_height"].weight = -0.1
  cfg.rewards["foot_swing_height"].params["target_height"] = 0.06
  cfg.terminations["fell_over"].params["limit_angle"] = 1.0471975512
  return cfg


def gait_rl_cfg():
  cfg = lerobot_humanoid_no_arms_ppo_runner_cfg()
  cfg.experiment_name = "lerobot_lower_body_gait"
  cfg.policy.actor_hidden_dims = (256, 128, 128)
  cfg.policy.critic_hidden_dims = (256, 128, 128)
  cfg.algorithm.learning_rate = 1e-3
  cfg.algorithm.entropy_coef = 0.008
  cfg.max_iterations = 3000
  cfg.save_interval = 250
  return cfg
