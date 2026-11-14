# Brightness check: mean gray value of sampled frames (too dark or blown out = bad)
import cv2

from egoqc.checks.result import CheckResult
from egoqc.checks.sampling import sample_frames
from egoqc.checks.summary import summarize_marks


def brightness_score(frame) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(gray.mean())  # 0 = black, 255 = white


def summarize_brightness(
    scores, minimum_brightness, maximum_brightness, max_bad_share
) -> CheckResult:
    """Brightness's own part: a sample is bad when it is outside the allowed range."""
    bad = [score < minimum_brightness or score > maximum_brightness for _, score in scores]
    return summarize_marks("brightness", scores, bad, max_bad_share, what="badly exposed")


def brightness_check(clip, minimum_brightness, maximum_brightness, max_bad_share) -> CheckResult:
    scores = []  # (time in seconds, mean gray value), nothing dropped
    for time, frame in sample_frames(clip):
        scores.append((time, brightness_score(frame)))
    return summarize_brightness(scores, minimum_brightness, maximum_brightness, max_bad_share)


def brightness_check_with_config(clip, config: dict) -> CheckResult:
    """Run the brightness check with settings from an already loaded config dict."""
    settings = config.get("checks", {}).get("brightness")
    if settings is None:
        raise ValueError("config.yaml has no 'checks:' -> 'brightness:' section")
    return brightness_check(
        clip,
        minimum_brightness=settings["minimum_brightness"],
        maximum_brightness=settings["maximum_brightness"],
        max_bad_share=settings["max_bad_share"],
    )


if __name__ == "__main__":
    from pathlib import Path

    from egoqc.config import load_config
    from egoqc.loaders.humanola import load_humanola

    config = load_config()  # read the file once
    clip = next(load_humanola(Path("data/humanola_sample")))
    result = brightness_check_with_config(clip, config)
    print(result.passed, result.note)
