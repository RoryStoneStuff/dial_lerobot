"""Create a synchronized morphology-and-optimization-graph video."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def best_evaluation(generation):
    ranked = [e for e in generation.get("evaluations", []) if e.get("selection_rank") == 0]
    return ranked[0] if ranked else min(generation["evaluations"], key=lambda e: e.get("fitness", 0))


def label(image, generation, evaluation):
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 28)
        small = ImageFont.truetype("DejaVuSans.ttf", 22)
    except OSError:
        font = small = ImageFont.load_default()
    objective = evaluation["objective"]
    draw.rectangle((18, 18, 450, 122), fill=(0, 0, 0))
    draw.text((32, 28), f"CMA-ES generation {generation + 1}", fill="white", font=font)
    draw.text((32, 65), f"CoT: {objective['mean_positive_mechanical_cot']:.3f}", fill="white", font=small)
    draw.text((210, 65), f"Velocity: {objective['mean_vx_m_s']:.3f} m/s", fill="white", font=small)
    return image


def graph_frame(points, generation_count, current_generation):
    visible = [point for point in points if point[0] <= current_generation]
    generations = sorted({point[0] for point in visible})
    grouped = {g: [point[2] for point in visible if point[0] == g] for g in generations}
    means = [float(np.mean(grouped[g])) for g in generations]
    minima = [min(grouped[g]) for g in generations]
    running_best = np.minimum.accumulate(minima)
    fig, axis = plt.subplots(figsize=(9.6, 7.2), dpi=100, constrained_layout=True)
    colors = plt.cm.viridis((np.asarray(generations) - 1) / max(1, generation_count - 1))
    color_map = dict(zip(generations, colors))
    for generation, candidate, cot in visible:
        axis.scatter(generation, cot, color=color_map[generation], edgecolor="white",
                     linewidth=0.5, s=48, zorder=3)
    axis.plot(generations, means, "o--", color="0.35", linewidth=2,
              label="Generation mean", zorder=2)
    axis.plot(generations, running_best, "s-", color="#d62728", linewidth=2,
              label="Best CoT so far", zorder=4)
    axis.set_xlim(0.5, generation_count + 0.5)
    all_cot = [point[2] for point in points]
    axis.set_ylim(min(all_cot) - 0.03, max(all_cot) + 0.03)
    axis.set_xticks(range(1, generation_count + 1))
    axis.set_xlabel("CMA-ES generation")
    axis.set_ylabel("Positive mechanical cost of transport")
    axis.set_title(f"Optimization progress through generation {current_generation}")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="upper right")
    fig.canvas.draw()
    image = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3])
    plt.close(fig)
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("optimization", type=Path)
    parser.add_argument("--morphology-frames", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seconds-per-generation", type=float, default=1.0)
    parser.add_argument("--crossfade-seconds", type=float, default=0.6)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    args = parser.parse_args()

    data = json.loads(args.optimization.read_text())
    generations = data["generations"]
    points = []
    for generation in generations:
        for candidate, evaluation in enumerate(generation["evaluations"]):
            cot = evaluation["objective"].get("mean_positive_mechanical_cot")
            if cot is not None:
                points.append((generation["generation"] + 1, candidate + 1, float(cot)))

    raw_images = []
    for generation in generations:
        raw_path = args.morphology_frames.parent / (
            args.morphology_frames.name.replace("_video_frames", "_frames")
            + f"/generation_{generation['generation']:03d}_raw.png"
        )
        if not raw_path.exists():
            raise FileNotFoundError(raw_path)
        raw_images.append(label(Image.open(raw_path), generation["generation"], best_evaluation(generation)))

    graph_images = [graph_frame(points, len(generations), generation + 1) for generation in range(len(generations))]
    output_frames = args.output.parent / (args.output.stem + "_frames")
    output_frames.mkdir(parents=True, exist_ok=True)
    hold_count = max(1, round(args.seconds_per_generation * args.fps))
    fade_count = max(1, round(args.crossfade_seconds * args.fps))
    frame_index = 0
    for index, (morphology, graph) in enumerate(zip(raw_images, graph_images)):
        next_morphology = raw_images[index + 1] if index + 1 < len(raw_images) else morphology
        next_graph = graph_images[index + 1] if index + 1 < len(graph_images) else graph
        current = Image.new("RGB", (morphology.width + graph.width, morphology.height), "white")
        current.paste(morphology, (0, 0)); current.paste(graph, (morphology.width, 0))
        for _ in range(hold_count):
            current.save(output_frames / f"frame_{frame_index:06d}.png"); frame_index += 1
        if index + 1 < len(raw_images):
            for fade_index in range(1, fade_count + 1):
                amount = fade_index / fade_count
                blended_morphology = Image.blend(morphology, next_morphology, amount)
                blended_graph = Image.blend(graph, next_graph, amount)
                blended = Image.new("RGB", current.size, "white")
                blended.paste(blended_morphology, (0, 0)); blended.paste(blended_graph, (morphology.width, 0))
                blended.save(output_frames / f"frame_{frame_index:06d}.png"); frame_index += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(args.ffmpeg), "-y", "-framerate", str(args.fps),
                    "-i", str(output_frames / "frame_%06d.png"), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "18", str(args.output)], check=True)
    print(args.output)
    print(f"Rendered {len(generations)} generations and {frame_index} synchronized frames")


if __name__ == "__main__":
    main()
