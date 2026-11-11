from pathlib import Path
from dataclasses import replace

import numpy as np
import pytest

from dial_lerobot.assets.draft_adapter import (
    DRAFT_JOINT_NAMES,
    motor_feedback_theta,
    generate_draft_artifact,
    load_draft_model,
)
from dial_lerobot.assets.foot_geometry import capsule_bottoms, foot_capsules
from dial_lerobot.assets.model_builder import JOINT_NAMES, build_model, initial_data
from dial_lerobot.config import Config, RobotConfig


def test_draft_dependency_error_is_actionable(tmp_path, monkeypatch):
    import dial_lerobot.assets.draft_adapter as adapter

    def missing():
        raise RuntimeError("DRAFT support is optional; install with `pip install -e '.[draft]'`")

    monkeypatch.setattr(adapter, "_robot_generator", missing)
    with pytest.raises(RuntimeError, match="install.*draft"):
        adapter.generate_draft_artifact(tmp_path)


def test_draft_backend_requires_supported_control_and_contacts():
    design = replace(Config().design, lower_bounds=(0.25, 0.20, 0.8, 0.22),
                     upper_bounds=(0.43, 0.33, 4.0, 0.31))
    Config(robot=RobotConfig(model_backend="draft", foot_contact_model="h1_pair"),
           design=design).validate()
    with pytest.raises(ValueError, match="torque control"):
        Config(robot=RobotConfig(
            model_backend="draft", control_mode="position", foot_contact_model="h1_pair"
        )).validate()
    with pytest.raises(ValueError, match="h1_pair"):
        Config(robot=RobotConfig(model_backend="draft")).validate()


def test_draft_motor_feedback_uses_mpc_demand_and_preserves_motor_floor():
    theta = np.asarray((0.34, 0.26, 2.0, 0.266, 0.75, 0.8, 0.9, 0.8, 0.9, 1.0))
    result = {"episodes": [{
        "max_torque_by_joint": {"hipz_left": 190.0, "knee_left": 290.0, "ankley_left": 35.0},
        "max_speed_by_joint": {"hipz_left": 17.0, "knee_left": 17.0, "ankley_left": 24.0},
    }]}
    updated = motor_feedback_theta(theta, result, safety_margin=1.2)
    np.testing.assert_allclose(updated[:4], theta[:4])
    np.testing.assert_allclose(updated[4:7], (1.14, 1.16, 1.05))
    np.testing.assert_allclose(updated[7:], (1.1333333333, 1.1333333333, 1.152))


@pytest.mark.integration
def test_draft_mass_calibration_hits_legacy_target(tmp_path):
    pytest.importorskip("draft")
    cfg = RobotConfig(
        model_backend="draft", foot_contact_model="h1_pair",
        draft_target_mass_kg=14.3817786, draft_motor_mass_scale=0.23,
    )
    artifact = generate_draft_artifact(tmp_path / "draft_model", cfg=cfg, strict=True)
    model = load_draft_model(artifact)
    assert float(model.body_mass.sum()) == pytest.approx(14.3817786, abs=1e-6)
    provenance = __import__("json").loads((artifact.output_dir / "dial_adapter.json").read_text())
    assert provenance["mass_calibration"]["post_calibration_mass_kg"] == pytest.approx(14.3817786)


@pytest.mark.integration
def test_nominal_draft_lower_body_compiles(tmp_path):
    pytest.importorskip("draft")
    artifact = generate_draft_artifact(tmp_path / "draft_model", strict=True)
    model = load_draft_model(artifact)
    assert (model.nq, model.nv, model.nu) == (19, 18, 12)
    assert tuple(model.actuator(index).name for index in range(model.nu)) == DRAFT_JOINT_NAMES
    assert artifact.feasibility_report["total_mass_kg"] > 0
    data = __import__("mujoco").MjData(model)
    __import__("mujoco").mj_forward(model, data)
    bottoms = capsule_bottoms(model, data, foot_capsules(model))
    assert np.min(bottoms) == pytest.approx(RobotConfig().initial_clearance, abs=1e-7)


@pytest.mark.integration
def test_draft_backend_caches_and_exports_self_contained_artifact(tmp_path):
    pytest.importorskip("draft")
    from dial_lerobot.assets.model_builder import describe_model, export_model_artifacts, nominal_design

    cfg = RobotConfig(
        model_backend="draft", foot_contact_model="h1_pair",
        draft_cache_dir=str(tmp_path / "cache"),
    )
    theta = nominal_design(cfg, include_hip_width=True)
    first = build_model(theta, cfg)
    second = build_model(theta, cfg, include_visual=False)
    assert (first.nq, first.nv, first.nu) == (second.nq, second.nv, second.nu) == (19, 18, 12)
    assert len(list((tmp_path / "cache").glob("*/dial_adapter.json"))) == 1
    metadata = describe_model(first, theta, cfg)
    assert metadata["backend"] == "draft"
    assert "draft_feasibility" in metadata
    export_model_artifacts(tmp_path / "result", theta, cfg)
    assert (tmp_path / "result" / "model.xml").is_file()
    assert (tmp_path / "result" / "feasibility_report.yaml").is_file()
    exported = __import__("mujoco").MjModel.from_xml_path(str(tmp_path / "result" / "model.xml"))
    assert (exported.nq, exported.nv, exported.nu) == (19, 18, 12)


@pytest.mark.integration
def test_nominal_draft_joint_frames_match_lerobot(tmp_path):
    pytest.importorskip("draft")
    artifact = generate_draft_artifact(tmp_path / "draft_model", strict=True)
    legacy = build_model()
    draft = load_draft_model(artifact)
    legacy_data = initial_data(legacy)
    draft_data = initial_data(draft)
    assert draft_data.qpos[2] == pytest.approx(legacy_data.qpos[2], abs=2e-6)
    for name in JOINT_NAMES:
        legacy_id = legacy.joint(name).id
        draft_id = draft.joint(name).id
        assert draft_data.xanchor[draft_id] == pytest.approx(
            legacy_data.xanchor[legacy_id], abs=2e-6
        )
        assert draft_data.xaxis[draft_id] == pytest.approx(
            legacy_data.xaxis[legacy_id], abs=2e-7
        )


@pytest.mark.integration
def test_draft_link_geoms_follow_child_joint_vectors(tmp_path):
    pytest.importorskip("draft")
    import mujoco

    artifact = generate_draft_artifact(tmp_path / "draft_model", strict=True)
    model = load_draft_model(artifact)
    data = initial_data(model)
    for body_id in range(1, model.nbody):
        body_name = model.body(body_id).name
        link_name = f"{body_name}_link"
        link_id = next((i for i in range(model.ngeom)
                        if model.geom(i).name == link_name), -1)
        child_ids = [i for i in range(model.nbody) if model.body_parentid[i] == body_id]
        if link_id < 0 or len(child_ids) != 1:
            continue
        child_id = child_ids[0]
        body_to_child = data.xpos[child_id] - data.xpos[body_id]
        body_to_geom = data.geom_xpos[link_id] - data.xpos[body_id]
        assert np.linalg.norm(np.cross(body_to_child, body_to_geom)) < 1e-6


@pytest.mark.integration
def test_remote_motor_mounts_scale_with_supporting_links(tmp_path):
    pytest.importorskip("draft")
    import mujoco
    from dial_lerobot.assets.model_builder import nominal_design

    cfg = RobotConfig(model_backend="draft", foot_contact_model="h1_pair")
    nominal = nominal_design(cfg, include_hip_width=True, include_motor_design=True)
    stretched = nominal.copy()
    stretched[0] *= 1.1
    stretched[1] *= 1.1
    nominal_model = load_draft_model(generate_draft_artifact(tmp_path / "nominal", nominal, cfg))
    stretched_model = load_draft_model(generate_draft_artifact(tmp_path / "stretched", stretched, cfg))
    neutral_data = mujoco.MjData(stretched_model)
    mujoco.mj_forward(stretched_model, neutral_data)
    nominal_data = initial_data(nominal_model)
    stretched_data = initial_data(stretched_model)
    for motor_name, body_name, scale in (
        ("left_thigh_link_motor", "left_thigh_link", 1.1),
        ("left_shin_link_motor", "left_shin_link", 1.1),
    ):
        geom_id = stretched_model.geom(motor_name).id
        body_id = stretched_model.body(body_name).id
        nominal_local = nominal_model.geom(motor_name).pos
        expected = stretched_data.xpos[body_id] + stretched_data.xmat[body_id].reshape(3, 3) @ (nominal_local * scale)
        np.testing.assert_allclose(stretched_data.geom_xpos[geom_id], expected, atol=1e-6)
    marker_names = [stretched_model.geom(i).name for i in range(stretched_model.ngeom)]
    assert not any(name.endswith("_detail") for name in marker_names)
    for side in ("left", "right"):
        foot_box = stretched_model.geom(f"{side}_foot_box")
        assert np.allclose(foot_box.rgba, [0.0, 0.0, 0.0, 0.0])
        axis = neutral_data.geom_xmat[foot_box.id].reshape(3, 3)
        assert abs(axis[2, 0]) < 1e-6
        assert abs(axis[2, 1]) < 1e-6
        assert abs(axis[2, 2]) > 1.0 - 1e-6
    assert all(f"{joint_name}_marker" in marker_names for joint_name in DRAFT_JOINT_NAMES)
