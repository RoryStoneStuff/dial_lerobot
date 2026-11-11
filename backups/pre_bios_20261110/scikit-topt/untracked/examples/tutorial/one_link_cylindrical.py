"""Topology optimization on a nonrectangular, link-like design domain."""

import argparse
import json
from pathlib import Path

import numpy as np
import skfem
import sktopt


PLA = {
    "youngs_modulus_pa": 3.5e9,
    "poisson_ratio": 0.36,
    "nominal_tensile_strength_pa": 50e6,
}


def make_link_mesh(length, core_radius, boss_radius, mesh_size):
    outer = sktopt.mesh.toy_problem.create_box_hex(
        length, 2.0 * boss_radius, 2.0 * boss_radius, mesh_size
    )
    centers = outer.p[:, outer.t].mean(axis=1)
    x = centers[0]
    y = centers[1] - boss_radius
    z = centers[2] - boss_radius
    boss_length = max(3.0 * mesh_size, 0.12 * length)
    end_factor = np.minimum(x / boss_length, (length - x) / boss_length)
    radius = core_radius + (boss_radius - core_radius) * np.clip(1.0 - end_factor, 0.0, 1.0)
    inside = (y / radius) ** 2 + (z / radius) ** 2 <= 1.0
    mesh = skfem.MeshHex(outer.p, outer.t[:, inside]).remove_unused_nodes()
    return mesh


def make_task(length, core_radius, boss_radius, mesh_size):
    mesh = make_link_mesh(length, core_radius, boss_radius, mesh_size)
    eps = mesh_size * 1.1
    fixed = sktopt.mesh.utils.get_points_in_range(
        (0.0, eps), (0.0, 2.0 * boss_radius), (0.0, 2.0 * boss_radius)
    )
    end = sktopt.mesh.utils.get_points_in_range(
        (length - eps, length), (0.0, 2.0 * boss_radius), (0.0, 2.0 * boss_radius)
    )
    mesh = mesh.with_boundaries({
        "dirichlet": mesh.facets_satisfying(fixed),
        "neumann_0": mesh.facets_satisfying(end),
        "neumann_1": mesh.facets_satisfying(end),
    })
    mesh = mesh.with_subdomains({"design": np.arange(mesh.nelements)})
    basis = skfem.Basis(
        mesh,
        skfem.ElementVector(skfem.ElementHex1()),
        intorder=1,
    )
    return sktopt.mesh.LinearElasticity.from_mesh_tags(
        basis,
        "all",
        ["u^2", "u^1"],
        [1.0, -1.0],
        PLA["youngs_modulus_pa"],
        PLA["poisson_ratio"],
    )


def make_config(output, iterations, volume_fraction):
    return sktopt.core.optimizers.OC_Config(
        dst_path=str(output),
        export_img=False,
        p=sktopt.tools.SchedulerConfig.constant(target_value=3.0),
        vol_frac=sktopt.tools.SchedulerConfig.constant(target_value=volume_fraction),
        beta=sktopt.tools.SchedulerConfig.constant(target_value=1.0),
        filter_radius=sktopt.tools.SchedulerConfig.constant(target_value=0.012),
        move_limit=sktopt.tools.SchedulerConfig.constant(target_value=0.1),
        eta=sktopt.tools.SchedulerConfig.constant(target_value=0.3),
        lambda_lower=1e-14,
        lambda_upper=1e10,
        max_iters=iterations,
        record_times=iterations,
        filter_type="spacial",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--length", type=float, default=0.206)
    parser.add_argument("--core-radius", type=float, default=0.0185)
    parser.add_argument("--boss-radius", type=float, default=0.028)
    parser.add_argument("--mesh-size", type=float, default=0.006)
    parser.add_argument("--volume-fraction", type=float, default=0.35)
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    task = make_task(args.length, args.core_radius, args.boss_radius, args.mesh_size)
    config = make_config(args.output, args.iterations, args.volume_fraction)
    (args.output / "material_assumptions.json").write_text(json.dumps(PLA, indent=2))
    optimizer = sktopt.core.OC_Optimizer(config, task)
    optimizer.parameterize()
    optimizer.optimize()


if __name__ == "__main__":
    main()
