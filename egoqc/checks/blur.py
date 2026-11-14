# Blur check: variance of the Laplacian on sampled frames (low score = blurry)
import cv2

from egoqc.checks.result import CheckResult
from egoqc.checks.sampling import sample_frames
from egoqc.checks.summary import summarize_marks


def blur_score(frame, width: int = 640) -> float:
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    old_height, old_width = gray_frame.shape  # numpy shape is (height, width)
    new_height = int(old_height * width / old_width)  # same scale factor, no stretching
    resize_frame = cv2.resize(gray_frame, (width, new_height))  # cv2 wants (width, height)
    laplacian_call = cv2.Laplacian(resize_frame, cv2.CV_64F)
    variance = float(laplacian_call.var())
    return variance


def summarize_scores(scores, score_threshold, max_bad_share) -> CheckResult:
    """Blur's own part: a sample is bad when its score is below the threshold."""
    bad = [score < score_threshold for _, score in scores]  # one True/False per sample
    return summarize_marks("blur", scores, bad, max_bad_share, what="blurry")


def blur_check(clip, score_threshold, max_bad_share, width) -> CheckResult:
    scores = []  # (time in seconds, blur score), nothing dropped
    for time, frame in sample_frames(clip):
        scores.append((time, blur_score(frame, width)))
    return summarize_scores(scores, score_threshold, max_bad_share)


def blur_check_with_config(clip, config: dict) -> CheckResult:
    """Run the blur check with settings from an already loaded config dict (no file is read here)."""
    settings = config.get("checks", {}).get("blur")
    if settings is None:
        raise ValueError("config.yaml has no 'checks:' -> 'blur:' section")
    return blur_check(
        clip,
        score_threshold=settings["score_threshold"],
        max_bad_share=settings["max_bad_share"],
        width=settings["width"],
    )


if __name__ == "__main__":
    from pathlib import Path

    from egoqc.config import load_config
    from egoqc.loaders.humanola import load_humanola

    config = load_config()  # read the file once
    clip = next(load_humanola(Path("data/humanola_sample")))
    result = blur_check_with_config(clip, config)
    print(result.passed, result.note)