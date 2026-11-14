# Clip object: path, fps, resolution, duration, source dataset, optional ground-truth pose
from pathlib import Path

from pydantic import BaseModel


class Clip(BaseModel):
    path: Path
    source_dataset: str
    # Measured from the decoded video. NaN / None means "not measured" (e.g. corrupt file), never 0.
    fps: float = float("nan")
    duration: float = float("nan")
    resolution: tuple[int, int] | None = None  # (width, height), same order as OpenCV
    # Location of the dataset's ground-truth hand pose (parquet), loaded only when needed.
    pose_path: Path | None = None
    # Why the clip could not be read (e.g. "corrupt file", "file missing"); None for a healthy clip.
    error: str | None = None
