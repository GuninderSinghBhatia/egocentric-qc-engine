# Frozen frames check: a sampled frame that is (almost) identical to the next frame = video stood still
import cv2
import numpy as np

from egoqc.checks.result import CheckResult
from egoqc.checks.sampling import sample_frame_pairs
from egoqc.checks.summary import summarize_marks


def _small_gray(frame, width):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height = int(gray.shape[0] * width / gray.shape[1])  # keep the aspect ratio
    return cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA).astype(np.int16)


def frozen_score(frame, next_frame, width: int = 160) -> float:
    """Mean absolute gray difference between two neighbouring frames (0 = identical).

    Frames are shrunk first, so sensor noise averages out and the score does not depend
    on the video's resolution. int16 so that the subtraction cannot wrap around.
    """
    return float(np.abs(_small_gray(frame, width) - _small_gray(next_frame, width)).mean())


def summarize_frozen(scores, difference_threshold, max_bad_share) -> CheckResult:
    """Frozen's own part: a sample is bad when it barely differs from the next frame."""
    bad = [score < difference_threshold for _, score in scores]
    return summarize_marks("frozen", scores, bad, max_bad_share, what="frozen")


def frozen_check(clip, difference_threshold, max_bad_share, width) -> CheckResult:
    scores = []  # (time in seconds, difference to the next frame), nothing dropped
    for time, frame, next_frame in sample_frame_pairs(clip):
        scores.append((time, frozen_score(frame, next_frame, width)))
    return summarize_frozen(scores, difference_threshold, max_bad_share)


def frozen_check_with_config(clip, config: dict) -> CheckResult:
    """Run the frozen check with settings from an already loaded config dict."""
    settings = config.get("checks", {}).get("frozen")
    if settings is None:
        raise ValueError("config.yaml has no 'checks:' -> 'frozen:' section")
    return frozen_check(
        clip,
        difference_threshold=settings["difference_threshold"],
        max_bad_share=settings["max_bad_share"],
        width=settings["width"],
    )
