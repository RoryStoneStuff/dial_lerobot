# DRAFT Integration

The DRAFT backend is an opt-in, 12-DOF generator that uses the joint and
foot-site interface expected by the existing DIAL task. The current LeRobot
model remains the default and is not modified by this backend.

## Current boundary

`dial_lerobot/assets/draft_lower_body/` is the declarative DRAFT robot. Its
design inputs are thigh length, shin length, and hip width. DRAFT derives
structural masses, motor envelopes, reflected inertias, and feasibility data.

`dial_lerobot/assets/draft_adapter.py` translates DRAFT's mirrored joint names
to the established LeRobot names, adds the 12 torque actuators DIAL requires,
adds the established two-capsule foot contacts, filters unsupported robot
self-collisions, and sets floor clearance.

The generated artifact contains:

- `draft_lower_body.xml`
- `feasibility_report.yaml`
- `parameters_resolved.yaml`
- `dial_adapter.json`
- the generated torso mesh and inertia report

Artifacts are cached by design vector, DRAFT specification contents, relevant
physics settings, and strictness. Set `robot.draft_cache_dir` to override the
default user cache. Evaluation folders receive a self-contained copy of the
generated XML, mesh, resolved parameters, feasibility report, and provenance.

## Setup

Install the pinned optional dependency:

```bash
pip install -e ".[draft]"
```

For the existing local clone, an offline editable installation is also valid:

```bash
pip install -e /path/to/draft --no-deps --no-build-isolation
```

## Generate

```bash
python -m scripts.generate_draft_lower_body \
  --output /tmp/draft_lerobot_nominal
```

The nominal model currently compiles to `nq=19`, `nv=18`, and `nu=12`, and the
existing locomotion task can construct its MJX model from it.

## Optimization

`configs/codesign_draft_moderate_15gen.toml` runs the established CMA-ES
protocol with DRAFT as the model backend:

```bash
./run_mpc.sh inspect --config configs/codesign_draft_moderate_15gen.toml
./run_mpc.sh optimize --config configs/codesign_draft_moderate_15gen.toml \
  --output results/codesign_draft_01
```

The DRAFT optimization vector is
`[thigh, shin, density_proxy, hip_width, hip_torque, knee_torque,
ankle_torque, hip_speed, knee_speed, ankle_speed]`, where motor entries are
scales relative to the nominal requirements. The path is
`theta -> cached DRAFT artifact -> DIAL evaluation -> MPC demand feedback ->
motor regeneration -> completion/survival/motor feasibility/CoT ranking`. A DRAFT
`FeasibilityViolation` is a `ValueError`, so infeasible CMA-ES candidates are
recorded as failed candidates and receive the dominated rank without entering
the MPC rollout. Dependency and programming errors still propagate.

The legacy density slot remains fixed at `2.0` for saved-vector compatibility;
DRAFT currently consumes thigh length, shin length, and hip width and derives
the physical masses and actuator armatures. After the first rollout, observed
per-joint torque and speed peaks are aggregated by motor class and multiplied
by a 20% safety margin. If those requirements exceed the proposed motor
scales, DRAFT regenerates the robot and the final rollout is scored. DRAFT
warnings are preserved in
`metadata.json` and `feasibility_report.yaml`; allowed warnings do not reject a
candidate. Hard DRAFT errors reject it when `draft_strict = true`.

Motor-design candidates retain shared JAX compilation. DRAFT's custom speed
annotations are normalized because the current MPC task reads actuator limits
from dynamic model arrays; actuator limits, armature, mass, and inertia remain
runtime inputs to the shared kernel.

## Parity report

```bash
python -m scripts.compare_draft_nominal \
  --output /tmp/draft_lerobot_comparison \
  --json /tmp/draft_lerobot_comparison.json
```

The nominal kinematics now reproduce the source LeRobot body transforms. The
maximum joint-anchor error is approximately `0.001 mm`, the maximum joint-axis
error is below `0.000001 degrees`, and neutral root height matches. Thigh and
shin changes scale the original three-dimensional translation vectors rather
than replacing the CAD chain with vertical links.

Dynamics are intentionally not matched yet. The DRAFT model weighs about
40.5 kg versus 14.4 kg for the current model because it applies fitted
structural and actuator trends. DRAFT also reports motor-packing and
structural-density warnings. These discrepancies are inputs to the dynamics
and feasibility phase, not accepted final parameters.

## Mass calibration

`configs/codesign_draft_mass_calibrated_15gen.toml` enables an opt-in mass
calibration layer. It scales motor geom density separately, then scales the
remaining structural density so the generated nominal model matches the
legacy LeRobot nominal mass of `14.3817786 kg`. The current motor scale of
`0.23` is an effective provisional value; it must be replaced by measured
actuator masses before treating the calibrated model as hardware-valid.

The calibration preserves dimensions, torque limits, and motor geometry while
matching total mass. It does not yet match the legacy center of mass or full
inertia tensor.

## Current limitation and next milestones

1. Decide which legacy mass properties are calibration targets and which
   should be replaced by DRAFT's population trends.
2. Remove the temporary packing allowances by correcting actuator placement.
3. Replace the provisional motor scale with a measured actuator catalog and
   calibrate center of mass and inertia in addition to total mass.
4. Remove the temporary packing allowances by correcting actuator placement.
5. Validate and, if needed, retune the nominal controller for calibrated DRAFT
   dynamics before interpreting a long morphology search.
6. Replace peak-demand sizing with a full torque-speed envelope and motor
   catalog selection.
