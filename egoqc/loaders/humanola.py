# Humanola loader (sample and labeled sets share the LeRobot layout)
import json
from pathlib import Path

import cv2
import yaml

from egoqc.clip import Clip


def load_humanola(dataset: Path, config_path: Path = Path("config.yaml")):
    """Yield one Clip per episode of a Humanola (LeRobot-layout) dataset folder."""
    # Step 1: camera name for this dataset, from config.yaml
    name = dataset.name  # e.g. "humanola_labeled"
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)  # nested dict

    if name not in config["datasets"]:
        known = ", ".join(config["datasets"])
        raise ValueError(f"Dataset '{name}' not found in config.yaml. Known datasets: {known}")

    camera = config["datasets"][name]["camera"]  # e.g. "observation.images.head"

    # Step 2: path to meta/info.json (fixed LeRobot layout, so no searching)
    meta = dataset / "meta" / "info.json"
    if not meta.exists():
        raise FileNotFoundError(f"{meta} not found. Is '{dataset}' a LeRobot dataset folder?")

    with open(meta, encoding="utf-8") as g:
        info = json.load(g)  # dict

    # template, e.g. "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    video_path = info["video_path"]
    chunks_size = info["chunks_size"]  # episodes per chunk folder (1000)

    # Step 3: per-episode metadata, one JSON object per line
    meta_jsonl = dataset / "meta" / "episodes.jsonl"
    if not meta_jsonl.exists():
        raise FileNotFoundError(f"{meta_jsonl} not found")

    episodes = []
    with open(meta_jsonl, encoding="utf-8") as h:
        for line in h:
            if not line.strip():  # skip blank lines
                continue
            episodes.append(json.loads(line))

    # Step 4: video file path for each episode
    for episode in episodes:
        index = episode["episode_index"]
        chunk = index // chunks_size  # whole-number division: 0-999 -> 0, 1000-1999 -> 1
        relative = video_path.format(episode_chunk=chunk, video_key=camera, episode_index=index)
        video_file = dataset / relative

        # reset per episode, so a bad clip never inherits the previous clip's numbers
        error = None
        video_fps = float("nan")  # unmeasured is NaN, never 0
        duration = float("nan")
        resolution = None

        # Step 5-7: missing / unopenable / invalid clips are recorded, not raised
        if not video_file.exists():
            error = "video file missing"
        else:
            cap = cv2.VideoCapture(str(video_file))
            if not cap.isOpened():
                error = "could not open video"
            else:
                cap_fps = cap.get(cv2.CAP_PROP_FPS)
                cap_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                cap_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if cap_fps <= 0 or cap_frames <= 0:
                    error = "video has invalid properties"
                else:
                    video_fps = cap_fps  # measured from the video, not info.json
                    duration = cap_frames / cap_fps
                    resolution = (cap_w, cap_h)  # (width, height)
            cap.release()

        # Step 8: hand out one Clip at a time (pose_path stays None for now)
        yield Clip(
            path=video_file,
            source_dataset=name,
            fps=video_fps,
            duration=duration,
            resolution=resolution,
            error=error,
        )
