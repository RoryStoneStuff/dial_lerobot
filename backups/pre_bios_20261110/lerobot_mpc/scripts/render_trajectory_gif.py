"""Render a saved MPC trajectory as an accelerated animated GIF."""

import argparse
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image


def main():
    p = argparse.ArgumentParser()
    p.add_argument("trajectory", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--every", type=int, default=5,
                   help="Render every Nth saved state")
    p.add_argument("--duration-ms", type=int, default=10,
                   help="Display duration per frame")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    args = p.parse_args()

    root = args.trajectory.parent
    model = mujoco.MjModel.from_xml_path(str(root / "model.xml"))
    data = mujoco.MjData(model)
    with np.load(args.trajectory, allow_pickle=False) as archive:
        time = archive["time"]
        qpos = archive["qpos"]
        qvel = archive["qvel"]
        ctrl = archive["ctrl"]

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = model.body("torso_subassembly").id
    camera.distance = 2.4
    camera.azimuth = 125
    camera.elevation = -15
    model.vis.global_.offwidth = args.width
    model.vis.global_.offheight = args.height

    frames = []
    with mujoco.Renderer(model, height=args.height, width=args.width) as renderer:
        for i in range(0, len(time), max(1, args.every)):
            data.qpos[:] = qpos[i]
            data.qvel[:] = qvel[i]
            data.ctrl[:] = ctrl[i]
            data.time = float(time[i])
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            frames.append(Image.fromarray(renderer.render()).convert("P", palette=Image.Palette.ADAPTIVE))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(args.output, save_all=True, append_images=frames[1:],
                   duration=max(1, args.duration_ms), loop=0, optimize=False)
    print(f"saved {args.output} ({len(frames)} frames, {float(time[-1]):.2f}s simulated)")


if __name__ == "__main__":
    main()
