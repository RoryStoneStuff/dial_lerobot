# LeRobot lower-body PPO training

Use `Mjlab-Velocity-Flat-LeRobot-Humanoid-no-arms` from the zoo repository.
The robot is `models/bipedal_plateform_no_arms/mjcf/robot.xml`: 12 controlled
leg joints, the original fixed 4 kg torso body, and 14.3818 kg total mass.

## Environment

Run commands from the repository root. MJLab is pinned to commit
`a3f28e50fc85e0d28b9b198821683ed369f60eec` (1.0.0); `uv.lock` retains the
compatible MuJoCo 3.4.0 / MuJoCo Warp 0.0.2 / RSL-RL 3.1.0 stack.
Do not substitute the separate newer MJLab checkout without migrating the APIs.

For a fresh installation use `uv sync --locked`. The existing `.venv` has the
validated dependencies; use `uv run --no-sync` to run it without reinstalling.
On this workstation uv is at `/home/rory/.local/bin/uv` if it is not on PATH.
A CUDA GPU is required. All logging defaults to local TensorBoard files.

## Baseline behavior and assumptions

- Flat ground; 1024 environments by default, 50 Hz policy control and 200 Hz physics.
- Forward commands 0–0.5 m/s, zero lateral command, yaw commands ±0.2 rad/s;
  the command generator also samples standing. Playback uses the same ranges.
- The inherited speed curriculum is removed because it overrides these ranges
  and eventually requests 3 m/s. External pushes are disabled for the initial
  flat-ground baseline. Friction, encoder bias, COM, and reset randomization remain.
- The CAD zero pose already has bent legs. Reset root height is 0.75 m plus
  the inherited 0.01–0.05 m reset offset. Nominal foot clearance is 0.03 m.
- Falls reset on root tilt above 70 degrees or flat-ground root height below
  0.45 m. Training episodes time out at 20 s. Play retains the existing long episode.
- Joint ranges are provisional simulation values copied by name from
  `lerobot-humanoid/lerobot-humanoid-model/models/bipedal_plateform_no_arms/mjcf/robot.xml`
  in the adjacent checkout on 2026-09-06. Both models cite Onshape assembly
  `fb645318a27646d1d8840be6 / d1cae8805fb652b4d1614997 / 804a1da43f242001a05129b4`.
  The XML now contains all 12 ranges and does not need that adjacent checkout.
  These are not independently measured hardware travel limits.
- Existing zoo motor gains and torque limits are retained: hip yaw 15 Nm,
  other hips/knees 88 Nm, ankles 44 Nm. These are simulation assumptions pending
  confirmation of motor specifications. The 6 Nm ankle threshold in the reward
  is a soft penalty, not a hard actuator torque cap.
- Raw policy actions are clipped to ±1, scaled to position offsets, and added
  to the nominal pose. The custom action also clamps the target after encoder
  bias to the 90% soft joint range. Physics enforces the XML travel limits.
- Self-collision sensing now covers the complete robot subtree, including legs.
  Only the existing collision geometry participates; the torso mesh is visual.
- Ankle power is calculated as sum(abs(torque * joint velocity)), explicitly
  matching actuator names to joints because their array orders differ.

## Rewards and PPO

The existing velocity tracking, uprightness, posture, foot clearance/slip,
landing, action smoothness, joint-limit, momentum, and effort terms are retained.
Their weights are in `training_exemples/lerobot_humanoid_no_arms/env_cfgs.py`;
base reward definitions come from the pinned MJLab velocity task. No reference
motion or separate MPC cost function is required.

PPO defaults remain: actor and critic MLPs (512, 256, 128), ELU, normalized
observations, learning rate 3e-4 with adaptive KL schedule, discount 0.99,
GAE lambda 0.95, PPO clip 0.2, entropy coefficient 0.01, 24 rollout steps,
5 epochs and 4 minibatches per update. Edit `rl_cfg.py` or use CLI overrides.

## Validate, train, and inspect

```bash
# Physics, target limits, contacts, and fall/timeout reset checks.
uv run --no-sync python tools/check_lower_body.py

# Short PPO integration check, not enough to learn walking.
uv run --no-sync train Mjlab-Velocity-Flat-LeRobot-Humanoid-no-arms \
  --env.scene.num-envs 128 --agent.max-iterations 10 \
  --agent.save-interval 5 --agent.run-name preparation_smoke \
  --enable-nan-guard True

# First learning run: inspect the resulting gait and metrics before extending it.
uv run --no-sync train Mjlab-Velocity-Flat-LeRobot-Humanoid-no-arms \
  --env.scene.num-envs 1024 --agent.max-iterations 3000 \
  --agent.run-name flat_baseline --enable-nan-guard True

# View local training metrics.
uv run --no-sync tensorboard --logdir logs/rsl_rl/lerobot_humanoid_no_arms_velocity

# Substitute an actual checkpoint path for <checkpoint.pt>.
uv run --no-sync python tools/check_lower_body.py --checkpoint <checkpoint.pt>
uv run --no-sync play Mjlab-Velocity-Flat-LeRobot-Humanoid-no-arms \
  --checkpoint-file <checkpoint.pt>
```

Run directories contain `params/env.yaml`, `params/agent.yaml`, TensorBoard
events, `model_*.pt`, and ONNX policy exports. To resume with the train command,
pass `--agent.resume True --agent.load-run <run-directory-name>` and optionally
`--agent.load-checkpoint model_100.pt`; `max-iterations` is additional updates.

The ONNX file exports the policy network and observation normalization. A
controller using it outside MJLab must also reproduce joint ordering, action
clipping/scaling/offsets, encoder handling, and the soft-limit target clamp in
`actions.py`; those operations are not part of the exported neural network.

Successful checks establish that training and checkpoint playback run. They do
not establish a stable walking policy. Inspect episode length, velocity errors,
individual reward terms, torque demand, and video throughout the first learning run.

## Preparation validation (2026-09-06)

The RTX 4090 passed 100-step GPU checks with eight environments, including
both foot contacts, extreme position-target clamping, 40 timeout resets, and
forced low-base fall resets. Actor/critic observations are 48/60 values.
A 128-environment PPO run completed 10 updates (30,720 environment steps),
updated eight actor parameter tensors, saved checkpoints, and exported a valid
ONNX model. The final checkpoint passed the same 100-step playback checks.

Local results: `logs/lower_body_checks.json` and
`logs/lower_body_checkpoint_checks.json`. The smoke checkpoint is
`logs/rsl_rl/lerobot_humanoid_no_arms_velocity/2026-09-06_14-40-28_preparation_smoke/model_9.pt`.
The full 1024-environment learning run has not been started. These smoke-test
weights are for checking the pipeline, not a trained walking controller.
