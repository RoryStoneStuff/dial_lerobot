"""Position targets constrained to the robot's soft joint travel limits."""

from dataclasses import dataclass

import torch
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.actions.actions import JointPositionAction


class LimitedJointPositionAction(JointPositionAction):
  def apply_actions(self) -> None:
    bias = self._entity.data.encoder_bias[:, self._target_ids]
    limits = self._entity.data.soft_joint_pos_limits[:, self._target_ids]
    # Bound after encoder bias, so the actual requested target stays inside limits.
    target = torch.clamp(self._processed_actions - bias, limits[..., 0], limits[..., 1])
    self._entity.set_joint_position_target(target, joint_ids=self._target_ids)


@dataclass(kw_only=True)
class LimitedJointPositionActionCfg(JointPositionActionCfg):
  def build(self, env) -> LimitedJointPositionAction:
    return LimitedJointPositionAction(self, env)
