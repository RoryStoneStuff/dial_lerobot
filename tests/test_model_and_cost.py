from dataclasses import replace

import mujoco
import numpy as np
import pytest

from dial_lerobot.assets.model_builder import build_model, initial_data, nominal_design, SHINS, ANKLES, THIGHS
from dial_lerobot.config import Config, CostConfig, load_config
from dial_lerobot.tasks.lerobot_task import COST_NAMES, LeRobotLocomotionTask


@pytest.fixture(scope="module")
def model():
    return build_model()


@pytest.fixture(scope="module")
def task(model):
    return LeRobotLocomotionTask(model)


def test_nominal_and_morphology_actually_change_dynamics(model):
    nominal = nominal_design()
    changed = build_model(nominal * [1.05, 0.95, 1.2])
    assert (model.nq, model.nv, model.nu) == (19, 18, 12)
    assert model.body_mass.sum() == pytest.approx(14.3817786)
    assert changed.body_mass.sum() != pytest.approx(model.body_mass.sum())
    for body, scale in zip((SHINS[0], ANKLES[0]), (1.05, 0.95)):
        i = model.body(body).id
        np.testing.assert_allclose(changed.body_pos[i], model.body_pos[i] * scale)
    thigh = model.body(THIGHS[0]).id
    assert changed.body_mass[thigh] == pytest.approx(model.body_mass[thigh] * 1.05 * 1.2)
    np.testing.assert_allclose(changed.body_inertia[thigh], model.body_inertia[thigh] * 1.05**3 * 1.2)
    assert changed.nmesh > model.nmesh  # Scaled visual meshes are cloned, not shared with fixed links.
    i = model.geom("femur_12_collision").id
    np.testing.assert_allclose(changed.geom_size[i], model.geom_size[i] * 1.05)
    assert changed.body_mass[model.body("torso_mesh").id] == 4.0


def test_h1_walk_phase_and_lift_curve_with_lerobot_joint_signs(task):
    # With H1's centered swing window, at phase .1 the left foot is in stance
    # while the half-cycle-offset right foot is already in swing. Half a cycle
    # later their roles reverse. Swing peaks at phase .5, not at phase .75.
    right_swing = task.gait_reference(0.1)
    left_swing = task.gait_reference(0.6)
    assert float(right_swing[0][1]) > float(right_swing[0][0])
    assert float(left_swing[0][0]) > float(left_swing[0][1])
    peak = task.gait_reference(0.5)[0]
    np.testing.assert_allclose(peak, [task.cfg.swing_height, 0.0], atol=1e-6)

    # The optional joint-reference target starts at the configured standing
    # pose translated from Berkeley signs into the LeRobot CAD coordinates.
    np.testing.assert_allclose(task.gait_reference(0.0)[1], [0.1, -0.1], atol=1e-6)
    # Left is at its standing knee target; the right knee is in swing.
    assert float(task.gait_reference(0.0)[2][0]) == pytest.approx(0.05)
    assert float(task.gait_reference(0.0)[2][1]) == pytest.approx(-0.3)
    assert np.max(np.abs(np.asarray(left_swing[2]) - [0.05, 0.05])) <= 0.35 + 1e-6
    for t in np.linspace(0, 1, 25):
        _, hips, knees = task.gait_reference(t)
        assert np.all(hips >= np.asarray(task.hip_bounds)[:, 0])
        assert np.all(hips <= np.asarray(task.hip_bounds)[:, 1])
        assert np.all(knees >= np.asarray(task.knee_bounds)[:, 0])
        assert np.all(knees <= np.asarray(task.knee_bounds)[:, 1])


def test_cost_velocity_frame_and_fall_penalty(model, task):
    state = initial_data(model)
    state.time = 3.0
    # Facing +world y and moving +world y is forward in the robot frame.
    state.qpos[3:7] = [np.sqrt(0.5), 0, 0, np.sqrt(0.5)]
    state.qvel[:3] = [0, 0.5, 0]
    mujoco.mj_forward(model, state)
    terms = np.asarray(task.cost_terms(state, state.ctrl))
    assert terms[2] < 1e-10
    # At t=3 the H1 walk phase has the right foot at peak clearance.
    assert float(terms[0]) == pytest.approx(
        task.cfg.gait_weight * task.cfg.swing_height**2, rel=1e-5
    )
    assert terms[-1] == 0
    state.qpos[2] = task.healthy_z_min - 0.01
    mujoco.mj_forward(model, state)
    assert float(task.cost_terms(state, state.ctrl)[-1]) == 10


def test_terminal_source_weights_and_effort_cost(model, task):
    state = initial_data(model)
    assert float(task.terminal_cost(state)) == pytest.approx(0.5)
    zero_cost = float(task.running_cost(state, np.zeros(12)))
    # Changing the optimizer action argument must not masquerade as Nm.
    assert float(task.running_cost(state, np.ones(12))) == pytest.approx(zero_cost)
    state.ctrl[:] = 0.5 * np.asarray(task.torque_limits)
    # H1's effort term is normalized torque times joint speed. Static support
    # torque has no mechanical-power penalty; moving under the same torque does.
    mujoco.mj_forward(model, state)
    stationary_energy = np.asarray(task.cost_terms(state, np.zeros(12)))[COST_NAMES.index("energy")]
    assert stationary_energy == pytest.approx(0.0, abs=1e-12)
    state.qvel[np.asarray(task.dof_indices)] = 80.0
    mujoco.mj_forward(model, state)
    terms = np.asarray(task.cost_terms(state, np.zeros(12)))
    assert terms[COST_NAMES.index("energy")] == pytest.approx(
        task.cfg.energy_weight * 12 * (0.5 * 80.0 / 160.0) ** 2
    )
    assert float(task.running_cost(state, np.zeros(12))) > zero_cost


def test_bad_knobs_fail_before_compilation():
    cfg = Config()
    with pytest.raises(ValueError):
        replace(cfg, mpc=replace(cfg.mpc, num_samples=0)).validate()
    with pytest.raises(ValueError):
        replace(cfg, cost=replace(CostConfig(), duty=1)).validate()
    with pytest.raises(ValueError):
        build_model([0, 0.26, 2])
    baseline = load_config("configs/baseline.toml")
    assert baseline.robot.control_mode == "torque"
    assert baseline.mpc.num_samples == 2048
    assert baseline.mpc.num_knots == 5
