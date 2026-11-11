"""Behavioral checks for the adapted locomotion reward terms."""

import math
import unittest
from types import SimpleNamespace

import torch

from training_exemples.lerobot_humanoid_no_arms.gait import biped_air_time, track_planar_velocity


def environment(command, scene):
  return SimpleNamespace(
    scene=scene,
    command_manager=SimpleNamespace(get_command=lambda _: command),
    extras={"log": {}},
  )


class GaitRewardsTest(unittest.TestCase):
  def test_air_time_requires_single_support_and_motion_command(self):
    # Double stance, flight, left stance, right stance, stationary command.
    contact = torch.tensor([[.2, .2], [0., 0.], [.2, 0.], [0., .6], [.2, 0.]])
    air = torch.tensor([[0., 0.], [.2, .2], [0., .3], [.7, 0.], [0., .3]])
    command = torch.tensor([[.5, 0., 0.]] * 4 + [[0., 0., 0.]])
    env = environment(command, {"feet_ground_contact": SimpleNamespace(data=SimpleNamespace(
      current_contact_time=contact, current_air_time=air,
    ))})
    torch.testing.assert_close(biped_air_time(env), torch.tensor([0., 0., .2, .4, 0.]))

  def test_tracking_rewards_motion_in_robot_heading(self):
    # Stationary, correct forward motion, and the same motion after 90-degree yaw.
    robot = SimpleNamespace(data=SimpleNamespace(
      root_link_quat_w=torch.tensor([[1., 0., 0., 0.], [1., 0., 0., 0.],
                                   [math.sqrt(.5), 0., 0., math.sqrt(.5)]]),
      root_link_lin_vel_w=torch.tensor([[0., 0., 0.], [.5, 0., 0.], [0., .5, 0.]]),
    ))
    env = environment(torch.tensor([[.5, 0., 0.]] * 3), {"robot": robot})
    values = track_planar_velocity(env)
    self.assertLess(float(values[0]), .02)
    torch.testing.assert_close(values[1:], torch.ones(2))


if __name__ == "__main__":
  unittest.main()
