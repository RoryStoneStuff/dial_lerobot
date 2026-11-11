"""Procedural box torso for the DRAFT lower-body parity model."""

from pathlib import Path

import trimesh
import yaml

from draft.generation.layered_mesh import inertial_properties


class LowerBodyTorsoMesh:
    name = "torso"
    density_param = "torso_rho"

    def get_layers(self, params):
        return []

    def build_and_save(self, params, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        length = float(params["torso_length"])
        mesh = trimesh.creation.box(
            extents=(float(params["torso_depth"]), float(params["torso_width"]), length)
        )
        mesh.apply_translation((0.0, 0.0, 0.5 * length))
        props = inertial_properties(mesh, float(params[self.density_param]))
        mesh.export(output_dir / "torso_mesh.stl")
        inertia = props["inertia_com"]
        center = props["center_of_mass"]
        (output_dir / "torso_inertial_properties.yaml").write_text(
            yaml.safe_dump({
                "mass": float(props["mass"]),
                "com_x": float(center[0]),
                "com_y": float(center[1]),
                "com_z": float(center[2]),
                "I_xx": float(inertia[0, 0]),
                "I_yy": float(inertia[1, 1]),
                "I_zz": float(inertia[2, 2]),
                "I_xy": float(inertia[0, 1]),
                "I_xz": float(inertia[0, 2]),
                "I_yz": float(inertia[1, 2]),
            }, sort_keys=False)
        )
        return mesh, props
