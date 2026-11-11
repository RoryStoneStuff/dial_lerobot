# Berkeley-inspired lower-body gait experiment

Task: `Mjlab-Velocity-Flat-LeRobot-Humanoid-no-arms-Gait`.
The original `...no-arms` task is preserved for baseline checkpoint playback.
The robot model, physical joint travel, torque caps, 50 Hz control rate, and
position-target bounds are unchanged. This experiment trains a fresh network.

## Inspiration checked locally

Compared the Berkeley-Humanoid-Lite-main biped `config/biped/env_cfg.py`,
`mdp/rewards.py`, and `config/biped/agents/rsl_rl_ppo_cfg.py`, plus the saved
Berkeley-Humanoid-Lite-Old run `logs/rsl_rl/biped/2026-01-12_14-16-46/params/env.yaml`.
The saved run confirms tracking width 0.25, positive biped air-time reward,
action-rate penalty -0.01, tilt penalty -2, and termination penalty -10.
The user's successful Berkeley policies were inspiration; their weights are
not transferred to this different robot or simulation backend.

## Changes relative to the standing baseline

- Forward commands 0.3–0.7 m/s and yaw ±0.3 rad/s; 2% standing commands,
  resampled every 10 seconds. No injected command velocity.
- Planar velocity tracking in a yaw-aligned frame, width 0.25 instead of 0.5,
  weight 4. This sharply reduces the reward for standing during forward commands.
- Tilt penalty instead of an upright bonus; remove the all-joint posture bonus.
  Retain a small deviation penalty for hip yaw/roll and ankle roll.
- Berkeley's positive biped air-time term: reward single support for up to 0.4 s,
  zero for double support or flight, active only for nonzero motion commands.
- Reduce action-rate cost from -0.1 to -0.01. Relax the extra ankle effort costs
  and remove the soft 6 Nm ankle penalty; physical torque limits remain unchanged.
- Foot clearance target 0.06 m, retain slip and collision penalties, penalize
  non-timeout terminations, terminate tilt above 60 degrees.
- Berkeley-style actor/critic MLPs (256, 128, 128), learning rate 1e-3,
  entropy coefficient 0.008. Retain normalization and other existing PPO settings.

The custom reward and task definitions are in
`training_exemples/lerobot_humanoid_no_arms/gait.py`.

## Commands

```bash
cd /home/rory/coDesign/lerobot-legged-zoo
export PATH="/home/rory/.local/bin:$PATH"
TASK="Mjlab-Velocity-Flat-LeRobot-Humanoid-no-arms-Gait"

# Fresh run (do not run a second copy while the agent-started run is active).
uv run --no-sync train "$TASK" --env.scene.num-envs 1024 \
  --agent.max-iterations 3000 --agent.run-name berkeley_inspired_v1 \
  --enable-nan-guard True

uv run --no-sync tensorboard --logdir logs/rsl_rl/lerobot_lower_body_gait

# Set CHECKPOINT to a model_*.pt file from the gait experiment.
uv run --no-sync python tools/evaluate_locomotion.py --checkpoint "$CHECKPOINT"
uv run --no-sync play "$TASK" --checkpoint-file "$CHECKPOINT" --num-envs 1 --viewer native
```

`Locomotion/actual_forward_mps` measures actual motion;
`Locomotion/command_forward_mps` is the target;
`Locomotion/planar_error_mps` is tracking error;
`Locomotion/single_support_fraction` is the fraction on one foot.
These training metrics are batch averages and can include falling motion.
Use the evaluation script and video to establish sustained walking.

The headless evaluation uses eight environments, fixed commands at 0.3, 0.5,
and 0.7 m/s, zero turning, no standing sampling or injected velocity, and
20-second tests. It reports survival and displacement of survivors, actual
speed after a two-second settling period before each environment's first fall,
and foot support/landing statistics. JSON is saved next to the checkpoint.
Pass `--save-poses` to save the first environment's motion for offline rendering.

## Initial evaluation

Run: `logs/rsl_rl/lerobot_lower_body_gait/2026-09-06_19-07-42_berkeley_inspired_v1`.
At update 250 all eight environments survived each 20-second test. Measured
body-forward speeds were 0.341, 0.574, and 0.723 m/s for targets 0.3, 0.5,
and 0.7 m/s. This establishes movement, not yet a clean gait: contact transitions
are very frequent, foot clearance is small, and trajectories can turn despite
a zero yaw-rate command. Do not interpret single-support counts as clean steps.

The early evaluation JSON and a real-time 25 fps video are saved next to
`model_250.pt` as `model_250.locomotion.json` and `model_250_0.3mps.mp4`.
Training was started for 3000 updates; inspect later checkpoints before deciding
whether the shuffling improves or needs further reward/actuation changes.
The live console log is `/tmp/zoo-gait-training.log`.
