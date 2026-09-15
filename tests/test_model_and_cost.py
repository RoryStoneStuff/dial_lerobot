from dataclasses import replace

import mujoco
import numpy as np
import pytest

from dial_lerobot.assets.model_builder import build_model, initial_data, nominal_design, SHINS, ANKLES, THIGHS
from dial_lerobot.config import Config, CostConfig, load_config
from dial_lerobot.tasks.lerobot_task import LeRobotLocomotionTask


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


def test_feet_alternate_and_references_respect_real_limits(task):
    left = task.gait_reference(0.775)
    right = task.gait_reference(0.275)
    np.testing.assert_allclose(left[0], [0.06, 0.0], atol=1e-6)
    np.testing.assert_allclose(right[0], [0.0, 0.06], atol=1e-6)
    assert float(left[2][0]) == pytest.approx(-0.5625)
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
    assert terms[-1] == 0
    state.qpos[2] = task.healthy_z_min - 0.01
    mujoco.mj_forward(model, state)
    assert float(task.cost_terms(state, state.ctrl)[-1]) == 10


def test_terminal_source_weights_and_no_effort_penalty(model, task):
    state = initial_data(model)
    assert float(task.terminal_cost(state)) == pytest.approx(0.5)
    assert float(task.running_cost(state, np.zeros(12))) == pytest.approx(
        float(task.running_cost(state, np.ones(12))))


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
