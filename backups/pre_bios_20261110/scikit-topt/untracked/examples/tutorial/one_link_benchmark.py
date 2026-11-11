"""Small 3-D link topology-optimization benchmark.

This is intentionally a load-case prototype, not yet a MuJoCo importer.
"""

import argparse
from pathlib import Path

import numpy as np
import skfem
import sktopt


def make_task(length, width, mesh_size, output_dir):
    mesh = sktopt.mesh.toy_problem.create_box_hex(
        length, width, width, mesh_size
    )
    eps = mesh_size * 1.1
    fixed = sktopt.mesh.utils.get_points_in_range(
        (0.0, eps), (0.0, width), (0.0, width)
    )
    end = sktopt.mesh.utils.get_points_in_range(
        (length - eps, length), (0.0, width), (0.0, width)
    )
    mesh = mesh.with_boundaries({
        "dirichlet": fixed,
        "neumann_0": end,
        "neumann_1": end,
    })
    mesh = mesh.with_subdomains({"design": np.arange(mesh.nelements)})
    basis = skfem.Basis(
        mesh, skfem.ElementVector(skfem.ElementHex1()), intorder=1
    )
    return sktopt.mesh.LinearElasticity.from_mesh_tags(
        basis,
        "all",
        ["u^2", "u^1"],
        [1.0, -1.0],
        70e9,
        0.33,
    )


def make_config(output_dir, iterations):
    return sktopt.core.optimizers.OC_Config(
        dst_path=str(output_dir),
        export_img=False,
        p=sktopt.tools.SchedulerConfig.constant(target_value=3.0),
        vol_frac=sktopt.tools.SchedulerConfig.constant(target_value=0.35),
        beta=sktopt.tools.SchedulerConfig.constant(target_value=1.0),
        filter_radius=sktopt.tools.SchedulerConfig.constant(target_value=0.012),
        move_limit=sktopt.tools.SchedulerConfig.constant(target_value=0.1),
        eta=sktopt.tools.SchedulerConfig.constant(target_value=0.3),
        lambda_lower=1e-14,
        lambda_upper=1e10,
        max_iters=iterations,
        record_times=iterations,
        filter_type="helmholtz",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--length", type=float, default=0.206)
    parser.add_argument("--width", type=float, default=0.037)
    parser.add_argument("--mesh-size", type=float, default=0.006)
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    task = make_task(args.length, args.width, args.mesh_size, args.output)
    config = make_config(args.output, args.iterations)
    optimizer = sktopt.core.OC_Optimizer(config, task)
    optimizer.parameterize()
    optimizer.optimize()


if __name__ == "__main__":
    main()
