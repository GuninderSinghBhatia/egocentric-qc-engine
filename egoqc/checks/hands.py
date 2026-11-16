# Hand visibility check (Phase 3, minimal): are the hands in view? A manipulation clip
# without hands teaches a robot nothing. Uses MediaPipe's hand landmarker on sampled frames.
from functools import lru_cache
from pathlib import Path

import cv2

from egoqc.checks.result import CheckResult
from egoqc.checks.sampling import sample_frames
from egoqc.checks.summary import summarize_marks


@lru_cache(maxsize=2)
def hand_detector(model_path: str):
    """One MediaPipe HandLandmarker per model file, reused for every frame and clip."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"hand model not found: {path} (see Phase 0 setup)")
    import mediapipe as mp  # slow import, so only when the check actually runs
    from mediapipe.tasks.python import BaseOptions, vision

    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(path)), num_hands=2,
    )  # IMAGE mode: every sample is judged on its own
    return mp, vision.HandLandmarker.create_from_options(options)


def hands_score(frame, model_path: str, width: int = 640) -> float:
    """Number of hands MediaPipe finds in the frame (0, 1 or 2), as a float."""
    mp, detector = hand_detector(model_path)
    height = int(frame.shape[0] * width / frame.shape[1])  # keep the aspect ratio
    small = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    return float(len(result.hand_landmarks))


def summarize_hands(scores, max_bad_share) -> CheckResult:
    """Hands' own part: a sample is bad when no hand is visible."""
    bad = [count == 0 for _, count in scores]
    return summarize_marks("hands", scores, bad, max_bad_share, what="handless")


def hands_check(clip, model_path, width, max_bad_share) -> CheckResult:
    scores = []  # (time in seconds, hands found), nothing dropped
    for time, frame in sample_frames(clip):
        scores.append((time, hands_score(frame, model_path, width)))
    return summarize_hands(scores, max_bad_share)


def hands_check_with_config(clip, config: dict) -> CheckResult:
    """Run the hand visibility check with settings from an already loaded config dict."""
    settings = config.get("checks", {}).get("hands")
    if settings is None:
        raise ValueError("config.yaml has no 'checks:' -> 'hands:' section")
    return hands_check(
        clip,
        model_path=settings["model_path"],
        width=settings["width"],
        max_bad_share=settings["max_bad_share"],
    )
