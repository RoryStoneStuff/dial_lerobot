# Stance contact investigation — 2026-09-16

## Scope and corrections

The goal is to establish physically coherent stance behavior before further walking tuning. Experiments use the LeRobot model, HydraX DIAL, and the existing joint-family torque limits (hips 200 Nm, knees 300 Nm, ankles 40 Nm). The PD gains and all other cost weights are unchanged in these comparisons.

An independent CPU MuJoCo check held all normalized targets at zero, starting with no random height offset. At 2 ms and 5 ms physics steps it completed 2 seconds at about 0.712 m base height. At 10 ms it crossed below 0.5 m after 1.0 seconds, with peak generalized velocity about 84; at 20 ms it did so after 0.22 seconds, with peak velocity about 2,409. These are short numerical sanity checks, not MPC walking tests, and do not establish long-term stability. The MPC comparisons consequently use **5 ms physics, 10 ms replanning** (control decimation 2).

The previous 30 mm “marker offset” was initial clearance, not an offset between the foot site and the physical sole. Actual neutral site-to-lowest-capsule offsets are approximately 30–36 nanometers. Reset placement now uses actual collision surfaces. The cost's resting height excludes initial airborne clearance. The tests disable random height jitter; the old ±20 mm reset jitter could start a robot either penetrating the floor or airborne even when `initial_clearance=0`.

Both planning and execution now refresh position/velocity kinematics after each physics step. Previously execution refreshed them, while planning evaluated integrated joint positions with foot positions from before integration. Refreshing `kinematics`, `com_pos`, and `com_vel` avoids a redundant second contact solve.

## Exact cost changes

Each foot has six collision capsules resolved by name from the loaded model. For capsule g, its signed lowest-surface height above the world z=0 floor is

`h_g = geom_z - abs(R_zz) * capsule_half_length - capsule_radius`.

This calculation uses the actual geometry and rotation, including rolled or pitched feet. It is checked against MuJoCo's `mj_geomDistance` for a tilted robot with changed leg lengths. It currently supports the flat ground plane in this model. These distances measure proximity, **not contact force or guaranteed load-bearing support**.

| Term | Before | After |
|---|---|---|
| Gait height, weight 5 | Foot-site z minus initial site z, compared to the sinusoidal height reference | Lowest collision-surface height compared to the same reference; still squared meters, weight 5 |
| Stance contact | `w_contact * foot_site_height²` | `w_contact * mean_g((max(abs(h_g)-0.002,0)/0.02)²)` per scheduled stance foot |
| Swing contact | `w_contact * max(0.01-foot_site_height,0)²` | `w_contact * (max(min(0.01,reference_height)-min_g(h_g),0)/0.02)²` per swing foot |
| Stance slip (new) | Absent | `w_slip * sum(||foot_site_world_velocity_xy / 0.2||²)` for scheduled stance feet within 2 mm of the floor |

The six-capsule mean encourages an approximately flat sole instead of accepting toe/edge contact alone. The 2 mm deadband accommodates soft contact. Normalization by 20 mm is a deliberate increase in contact-term scale; reusing the number 0.05 does **not** preserve the old term's strength. For example, one flat foot 22 mm above the floor now contributes `w_contact`, versus roughly `0.000484*w_contact` previously (at zero initial clearance). Swing clearance tapers to zero at lift-off and touchdown rather than abruptly requesting 10 mm.

The slip measurement is velocity of the sole site, including rotation of its body. It matches native MuJoCo `mj_objectVelocity` in a nonzero-velocity regression test. It is not an estimate of friction force or of the instantaneous velocity of every possible contact point.

An optional `gait_start_delay` requests both feet in stance with zero gait/joint references during startup. A zero-amplitude standing reference also treats both feet as stance feet. Duty changes, if tested, change the scheduled support fraction rather than a cost coefficient. The forward-velocity ramp retains its original timing.

Height and fall thresholds now derive from the grounded neutral pose even if the initial pose has clearance. The energy term remains squared normalized action; it is not mechanical energy or the source DIAL repository's normalized-torque effort term. Velocity, upright, yaw, angular-rate, joint-reference, lateral-joint, fall-penalty and terminal weights were not retuned here.

Raw accumulated costs are retained for compatibility; `integrated_running_cost` additionally multiplies by physics dt. Total objective values from different cost definitions must not be treated as a common performance ranking.

## Experiment protocol

- RTX 4090, JAX CUDA device confirmed in each saved metadata file.
- Physics dt 0.005 s, replan dt 0.010 s, horizon 0.3 s, 5 knots, 128 samples, noise 0.03, temperature 0.05, one DIAL iteration.
- Desired forward speed 0.3 m/s with the existing 2 s ramp.
- Identical direct episode seeds 45 and 46; reset clearance and height jitter both zero.
- Short comparisons last up to 3 s and terminate on the existing fall criterion (base height or tilt). A non-finite first update now saves an empty prefix and failure reason instead of crashing during summary generation. The live viewer also pauses on a detected fall.
- Trajectories, physical model XML, configuration, git revision/diff, diagnostics and results are under `results/stance_contact_audit/`.
- Foot metrics are recomputed with CPU MuJoCo kinematics from saved qpos/qvel. “Near ground” means lowest capsule height between -10 and +2 mm; it does not certify support force. Slip RMS is measured only during scheduled stance and near-ground samples. Swing clearance is the fraction of scheduled swing observations whose lowest capsule exceeds 15 mm.

Results and the selected configuration are recorded below after the comparisons finish.

## Paper objective coverage (corrected during the paper-weight comparison)

The ordinary H1 walking reward in `dial-mpc/dial_mpc/envs/unitree_h1_env.py`
has active gait-height, upright, yaw, body-frame velocity, yaw-rate, base-height,
and normalized-torque effort terms. Air-time, root-position, and alive have
zero weight. A separate contact reward appears in another H1 task variant;
it is not part of this walking reward. Our sole-contact and contact-event
penalties are additional LeRobot objectives, not equivalent source terms.

The prior claim that all terms were faithfully implemented was too strong.
The current air-time placeholder depends on scheduled phase, not measured
flight/contact history, and cannot distinguish candidate actions. It remains
disabled. Root-position currently uses a world-XY analytic ramp integral;
that differs from the source position target and is also disabled. An exact
source air-time implementation would require carrying per-foot contact history
through planner and evaluator scans.

The original stance trials above used normalized-action effort. A later effort
change accidentally divided normalized actions by torque limits. The new
paper-weight comparisons correct that input to `state.ctrl`, the clipped PD
torque command used for each physics step with unit motor gears. Effort is
`sum((torque_command / torque_limit)^2)`; it is not mechanical work in joules.
Both new comparison arms use this correction. See `paper_weight_trials.md`.
