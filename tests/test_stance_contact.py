from dataclasses import replace
import mujoco
import numpy as np
import pytest
from dial_lerobot.assets.model_builder import build_model, initial_data, nominal_design
from dial_lerobot.assets.foot_geometry import foot_capsules, capsule_bottoms
from dial_lerobot.config import CostConfig, RobotConfig
from dial_lerobot.tasks.lerobot_task import COST_NAMES, LeRobotLocomotionTask


def test_capsule_distance_matches_mujoco_for_tilted_and_resized_robot():
    m = build_model(nominal_design() * [1.07, .96, 1.])
    d = initial_data(m)
    d.qpos[2] += .10
    d.qpos[3:7] = [np.cos(.12), 0, np.sin(.12), 0]
    mujoco.mj_forward(m, d)
    ids = foot_capsules(m)
    actual = capsule_bottoms(m, d, ids)
    expected = np.array([[mujoco.mj_geomDistance(m, d, int(g), m.geom('ground').id,
                                                 2., None) for g in foot] for foot in ids])
    np.testing.assert_allclose(actual, expected, atol=1e-9)


def test_initial_air_gap_is_not_mistaken_for_site_offset():
    m = build_model(cfg=replace(RobotConfig(), initial_clearance=.03))
    task = LeRobotLocomotionTask(m)
    d = initial_data(m)
    assert float(task.foot_bottoms(d).min()) == pytest.approx(.03, abs=1e-6)
    assert np.max(np.abs(task.foot_height_offset)) < 1e-5
    assert float(task.cost_terms(d, np.zeros(12))[COST_NAMES.index('contact')]) > .05
    assert task.rest_height == pytest.approx(d.qpos[2] - .03)


def test_stance_slip_velocity_matches_native_mujoco_and_is_contact_gated():
    m = build_model()
    task = LeRobotLocomotionTask(m, replace(CostConfig(), gait_start_delay=.3, stance_slip_weight=.1))
    d = initial_data(m)
    d.qvel[:] = np.linspace(-.4, .4, m.nv)
    mujoco.mj_forward(m, d)
    for i, sid in enumerate(task.foot_sites):
        v = np.zeros(6)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_SITE, int(sid), v, 0)
        np.testing.assert_allclose(task.foot_velocity(d)[i], v[3:], atol=1e-6)
    idx = COST_NAMES.index('stance_slip')
    assert float(task.cost_terms(d, np.zeros(12))[idx]) > 0
    d.qpos[2] += .03
    mujoco.mj_forward(m, d)
    assert float(task.cost_terms(d, np.zeros(12))[idx]) == 0


def test_double_support_start_and_continuous_clearance_at_swing_boundary():
    m = build_model()
    task = LeRobotLocomotionTask(m, replace(CostConfig(), gait_start_delay=.3, duty=.6))
    assert np.all(task.stance_mask(.1))
    np.testing.assert_allclose(task.gait_reference(.1)[0], 0)
    np.testing.assert_allclose(task.gait_reference(.1)[1], [.1, -.1])
    np.testing.assert_allclose(task.gait_reference(.1)[2], [.05, .05])
    d = initial_data(m)
    idx = COST_NAMES.index('contact')
    d.time = .6 - 1e-5  # left foot phase approaches duty / 2 = .3
    before = float(task.cost_terms(d, np.zeros(12))[idx])
    d.time = .6 + 1e-5
    after = float(task.cost_terms(d, np.zeros(12))[idx])
    assert abs(after-before) < 1e-6
    d.qpos[3:7] = [np.cos(.05), np.sin(.05), 0, 0]
    mujoco.mj_forward(m, d)
    d.qpos[2] -= float(task.foot_bottoms(d).min())
    mujoco.mj_forward(m, d)
    d.time = .1
    # Merely touching with an edge does not count as a flat supported stance.
    assert float(task.cost_terms(d, np.zeros(12))[idx]) > .01
