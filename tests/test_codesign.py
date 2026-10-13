from dataclasses import replace
import json

import pytest

from dial_lerobot.config import Config, load_config
from dial_lerobot.design_objective import design_objective, minimization_ranks


def result(cfg, updates, cost, fell=False, seconds=None):
    seconds = updates * cfg.robot.timestep if seconds is None else seconds
    return {"fitness": -cost, "model": {}, "episodes": [{
        "updates": updates, "sim_seconds": seconds,
        "integrated_running_cost": cost, "fell": fell,
        "failure_reason": "fall" if fell else None,
    }]}


def config():
    cfg = Config()
    return replace(cfg, evaluation=replace(cfg.evaluation, num_episodes=1, episode_steps=100))


def test_fixed_masses_preserve_original_weight_and_resize_inertia():
    import numpy as np
    from dial_lerobot.assets.model_builder import build_model, nominal_design, THIGHS, SHINS
    cfg = load_config("configs/codesign_walking.toml")
    robot = replace(cfg.robot, preserve_link_masses=True)
    theta = nominal_design(robot, include_hip_width=True)
    original = build_model(theta, robot, include_visual=False)
    theta[:2] *= [1.5, 0.5]
    theta[2] = 4.0  # Ignored by the fixed-mass physics policy.
    resized = build_model(theta, robot, include_visual=False)
    np.testing.assert_array_equal(resized.body_mass, original.body_mass)
    for names, ratio in ((THIGHS, 1.5), (SHINS, 0.5)):
        for name in names:
            body = original.body(name).id
            np.testing.assert_allclose(resized.body_inertia[body],
                                       original.body_inertia[body] * ratio**2)


def test_fall_cannot_win_by_accumulating_less_cost():
    cfg = config()
    survived = design_objective(result(cfg, 100, 10000), cfg)[0]
    fell = design_objective(result(cfg, 99, 0, True), cfg)[0]
    assert survived > fell
    assert minimization_ranks([fell, survived, survived]) == [1, 0, 0]


def test_completed_designs_compare_cost_not_float_clock_drift():
    cfg = config()
    good = design_objective(result(cfg, 100, 5, seconds=1.999999), cfg)[0]
    bad = design_objective(result(cfg, 100, 10, seconds=2.000001), cfg)[0]
    assert good > bad
    with pytest.raises(FloatingPointError):
        design_objective(result(cfg, 100, float("nan")), cfg)


def test_outer_loop_shared_seeds_selection_and_saved_design(monkeypatch, tmp_path):
    import dial_lerobot.optimize as module

    cfg = config()
    cfg = replace(cfg, design=replace(cfg.design, maxiter=1, popsize=2,
                                     lower_bounds=(*cfg.design.lower_bounds, 0.24),
                                     upper_bounds=(*cfg.design.upper_bounds, 0.293)))
    calls = []

    class FakeEvaluator:
        def __init__(self, cfg, theta, **kwargs):
            self.shared_kernel = object()
            self.cfg = cfg

        def warmup(self):
            return 0.1

        def evaluate(self, seed, output):
            calls.append((seed, output))
            # The survivor has worse legacy fitness, but must be selected.
            if len(calls) == 1:
                return result(cfg, 100, 2000)  # Initial morphology.
            return result(cfg, 100, 1000) if len(calls) == 2 else result(cfg, 10, 1, True)

    monkeypatch.setattr(module, "Evaluator", FakeEvaluator)
    monkeypatch.setattr(module, "runtime_versions", lambda: {})
    monkeypatch.setattr(module, "nominal_design", lambda robot, **kwargs: __import__("numpy").array(
        [0.3, 0.3, 2.0, 0.266] if kwargs.get("include_hip_width") else [0.3, 0.3, 2.0]))
    log = module.optimize_design(cfg, tmp_path)
    assert [seed for seed, _ in calls] == [cfg.evaluation.seed] * 3
    assert log["best_objective"]["completion_fraction"] == 1.0
    best = json.loads((tmp_path / "best_design.json").read_text())
    assert best["fitness"] == -1000
    assert len(best["theta"]) == 4
    assert best["trajectory_folder"] == str(calls[1][1].resolve())
    evaluations = log["generations"][0]["evaluations"]
    assert [item["selection_rank"] for item in evaluations] == [0, 1]


def test_codesign_profile_keeps_working_controller_and_geometry_based_height():
    cfg = load_config("configs/codesign_walking.toml")
    assert (cfg.robot.solver_iterations, cfg.robot.ls_iterations) == (2, 2)
    assert cfg.cost.ref_tracking_weight == 0
    assert cfg.cost.rest_height is None
    assert cfg.cost.healthy_z_min is None
    assert cfg.evaluation.episode_steps * cfg.robot.timestep == 10


@pytest.mark.parametrize("profile", ["codesign_walking", "codesign_extreme_512"])
def test_design_bounds_build_grounded_models_with_fixed_actuators(profile):
    from itertools import product
    import numpy as np
    from dial_lerobot.assets.model_builder import build_model, initial_data
    from dial_lerobot.assets.foot_geometry import foot_capsules, capsule_bottoms
    from dial_lerobot.tasks.lerobot_task import LeRobotLocomotionTask

    cfg = load_config(f"configs/{profile}.toml")
    nominal = build_model(cfg=cfg.robot)
    heights, masses = [], []
    for theta in product(*zip(cfg.design.lower_bounds, cfg.design.upper_bounds)):
        model = build_model(theta, cfg.robot)
        data = initial_data(model)
        bottoms = capsule_bottoms(model, data, foot_capsules(model))
        assert np.min(bottoms) >= -1e-6
        assert np.max(np.min(bottoms, axis=1)) < 0.001
        task = LeRobotLocomotionTask(model, cfg.cost)
        assert 0 < task.healthy_z_min < task.rest_height
        assert task.rest_height == pytest.approx(data.qpos[2], abs=0.001)
        np.testing.assert_array_equal(model.actuator_forcerange, nominal.actuator_forcerange)
        np.testing.assert_array_equal(model.jnt_range, nominal.jnt_range)
        assert model.nu == 12
        assert np.all(model.body_inertia[1:] > 0)
        heights.append(task.rest_height)
        masses.append(model.body_mass.sum())
    assert max(heights) - min(heights) > 0.05
    assert max(masses) > min(masses)


def test_hip_width_translates_complete_legs_symmetrically_and_preserves_legacy():
    import numpy as np
    from dial_lerobot.assets.model_builder import build_model, initial_data, nominal_design, HIP_ROOTS

    cfg = load_config("configs/codesign_walking.toml")
    theta = nominal_design(cfg.robot, include_hip_width=True)
    original = build_model(theta, cfg.robot)
    changed_theta = theta.copy()
    changed_theta[3] += 0.04
    changed = build_model(changed_theta, cfg.robot)
    before, after = initial_data(original), initial_data(changed)
    for name, sign in zip(HIP_ROOTS, (-1, 1)):
        delta = changed.body_pos[changed.body(name).id] - original.body_pos[original.body(name).id]
        np.testing.assert_allclose(delta, [0, sign * 0.02, 0], atol=1e-10)
    for site, sign in (("foot_right", -1), ("foot_left", 1)):
        delta = after.site_xpos[changed.site(site).id] - before.site_xpos[original.site(site).id]
        np.testing.assert_allclose(delta, [0, sign * 0.02, 0], atol=1e-10)
    np.testing.assert_array_equal(changed.body_mass, original.body_mass)
    np.testing.assert_array_equal(changed.body_inertia, original.body_inertia)
    legacy = build_model(theta[:3], cfg.robot)
    np.testing.assert_allclose(original.body_pos, legacy.body_pos, atol=1e-14)
    np.testing.assert_allclose(initial_data(legacy).qpos, before.qpos, atol=1e-14)
