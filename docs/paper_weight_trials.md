# H1 paper-weight screening — 2026-09-16

## Protocol

Source: the ordinary H1 task reward in
`../dial-mpc/dial_mpc/envs/unitree_h1_env.py`, not the contact reward in its
separate task variant. Local source digest, worktree diff, runner, full configs,
models, runtime versions and GPU identity are saved with the experiment.

Use HydraX DIAL unchanged, 1024 samples, 5 knots, 0.3 s horizon, one iteration,
noise 0.03, temperature 0.05, physics dt 0.005 s and replanning dt 0.010 s.
Target speed is 0.3 m/s with a 2 s ramp. Each screening trial lasts at most
3 simulated seconds, stopping immediately on the existing height/tilt fall
criterion or nonfinite values. Direct seeds: current weights 45; paper weights
45 and 46. Zero reset jitter. These are short screens, not sustained-walking
validation. Earlier CLI seed 45 produced a different episode seed, so only
these freshly matched runs are the direct weight comparison.

## Weights

| Term | Current comparison | Paper preset |
|---|---:|---:|
| Foot gait height | 25 | 5 |
| Joint reference | 12 | 0 |
| Forward/lateral velocity | 4 | 1 |
| Upright | 5 | 0.5 |
| Height | 1 | 0.5 |
| Yaw | 0.1 | 0.1 |
| Yaw rate | 1 | 1 |
| Torque effort | 0.01 | 0.01 |
| Lateral hip posture | 6 | 0 |
| Sole contact | 0.05 | 0 |
| Terminal height/velocity/upright | 5 / 2 / 10 | 0 / 0 / 0 |
| Fall penalty | 10 | 10 |

Air-time, position, contact-event, alive and slip remain zero. Fall protection
is a deliberate retained LeRobot safeguard. Geometry comes from LeRobot's
collision capsules and grounded neutral height, with unchanged torque limits.
The gait still uses our 5 cm swing target, not H1's 15 cm, and its reference
waveform and controller settings are not claimed to reproduce the paper.

## Input correction and limitations

Before these comparisons, effort incorrectly received normalized actions
and divided them by torque limits. Both arms now use the actual clipped PD
torque commands (`state.ctrl`, motor gears one). The regression check verifies
that changing the action argument alone does not change torque effort, and
half-limit torque on every actuator yields `0.01 * 12 * 0.5^2 = 0.03`.

The air-time placeholder is time-only, not a measured flight-time reward, and
is disabled. Root-position is also an approximation and disabled. These were
previously described too strongly as equivalent source implementations.
The source also converts angular velocity by pi/180; we retain rad/s units,
so identical coefficients do not imply identical effective angular penalties.

The source planner standardizes sampled mean rewards by their standard
deviation before softmax; HydraX uses summed/integrated costs and a fixed
temperature. A fixed numerical temperature is therefore not interchangeable.
We log sampled plan-cost standard deviation and effective sample size
`1 / sum(plan_weight^2)`. Values near 1024 mean nearly uniform averaging;
values near one mean one candidate dominates. These diagnostics describe
plan selection, not walking success.

Compare displacement, post-ramp velocity and actual foot lift. Do not compare
raw fitness or total costs across different objectives as a performance score.

## Results

Trials in progress; no walking result established yet.
