"""Create a cross-faded morphology video from a saved CMA-ES run."""

import argparse
import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def best_evaluation(generation):
    evaluations = generation.get("evaluations", [])
    ranked = [evaluation for evaluation in evaluations if evaluation.get("selection_rank") == 0]
    if ranked:
        return ranked[0]
    return min(evaluations, key=lambda evaluation: evaluation.get("fitness", float("inf")))


def label(image, generation, evaluation):
    result = image.convert("RGB")
    draw = ImageDraw.Draw(result)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 28)
        small = ImageFont.truetype("DejaVuSans.ttf", 22)
    except OSError:
        font = small = ImageFont.load_default()
    objective = evaluation["objective"]
    text = [
        f"CMA-ES generation {generation + 1}",
        f"CoT: {objective['mean_positive_mechanical_cot']:.3f}",
        f"Average velocity: {objective['mean_vx_m_s']:.3f} m/s",
    ]
    draw.rectangle((18, 18, 440, 122), fill=(0, 0, 0, 175))
    draw.text((32, 28), text[0], fill="white", font=font)
    draw.text((32, 65), text[1], fill="white", font=small)
    draw.text((210, 65), text[2], fill="white", font=small)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("optimization", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seconds-per-generation", type=float, default=1.0)
    parser.add_argument("--crossfade-seconds", type=float, default=0.6)
    parser.add_argument("--fps", type=int, default=24)
    args = parser.parse_args()

    data = json.loads(args.optimization.read_text())
    generations = data.get("generations", [])
    if not generations:
        raise ValueError("No completed generations found")

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from dial_lerobot.config import from_dict
    from dial_lerobot.assets.model_builder import build_model
    from dial_lerobot.visualize_lerobot import snapshot

    config = from_dict(data["config"])
    frame_dir = args.output.parent / (args.output.stem + "_frames")
    frame_dir.mkdir(parents=True, exist_ok=True)
    rendered = []
    for generation in generations:
        evaluation = best_evaluation(generation)
        theta = evaluation.get("evaluated_theta", evaluation["theta"])
        raw_path = frame_dir / f"generation_{generation['generation']:03d}_raw.png"
        snapshot(build_model(theta, config.robot), raw_path)
        rendered.append(label(Image.open(raw_path), generation["generation"], evaluation))

    frames_dir = args.output.parent / (args.output.stem + "_video_frames")
    frames_dir.mkdir(parents=True, exist_ok=True)
    hold_count = max(1, round(args.seconds_per_generation * args.fps))
    fade_count = max(1, round(args.crossfade_seconds * args.fps))
    frame_index = 0
    for index, image in enumerate(rendered):
        next_image = rendered[index + 1] if index + 1 < len(rendered) else image
        for _ in range(hold_count):
            image.save(frames_dir / f"frame_{frame_index:06d}.png")
            frame_index += 1
        if index + 1 < len(rendered):
            for fade_index in range(1, fade_count + 1):
                amount = fade_index / fade_count
                Image.blend(image, next_image, amount).save(
                    frames_dir / f"frame_{frame_index:06d}.png"
                )
                frame_index += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-y", "-framerate", str(args.fps),
        "-i", str(frames_dir / "frame_%06d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        str(args.output),
    ], check=True)
    print(args.output)
    print(f"Rendered {len(rendered)} generations and {frame_index} video frames")


if __name__ == "__main__":
    main()
