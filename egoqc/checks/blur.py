# Blur check: variance of the Laplacian on sampled frames (low score = blurry)
import cv2

from egoqc.checks.result import CheckResult
from egoqc.checks.sampling import sample_frames


def blur_score(frame, width: int = 640) -> float:
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    old_height, old_width = gray_frame.shape  # numpy shape is (height, width)
    new_height = int(old_height * width / old_width)  # same scale factor, no stretching
    resize_frame = cv2.resize(gray_frame, (width, new_height))  # cv2 wants (width, height)
    laplacian_call = cv2.Laplacian(resize_frame, cv2.CV_64F)
    variance = float(laplacian_call.var())
    return variance


def summarize_scores(scores, score_threshold, max_bad_share) -> CheckResult:
    """Turn a list of (time, score) into the facts and the verdict. Needs no video."""
    if not scores:  # nothing could be scored, for example the clip did not open
        return CheckResult(
            check="blur", passed=False, bad_share=float("nan"), longest_bad_s=float("nan"),
            longest_bad_start_s=None, note="no frames could be scored", scores=[],
        )

    interval = scores[1][0] - scores[0][0] if len(scores) > 1 else 1.0  # seconds per sample
    bad = [score < score_threshold for _, score in scores]  # one True/False per sample
    bad_share = sum(bad) / len(scores)

    # Walk through the samples once, remembering where the current bad run started
    longest = 0  # length of the longest run, in samples
    longest_start = None  # index where that run started
    run = 0  # length of the run we are in right now
    for i, is_bad in enumerate(bad):
        if is_bad:
            run += 1
            if run > longest:
                longest = run
                longest_start = i - run + 1
        else:
            run = 0

    longest_bad_s = longest * interval
    start_s = scores[longest_start][0] if longest_start is not None else None
    if longest == 0:
        note = "no blurry frames"
    else:
        note = (
            f"{bad_share:.1%} of sampled frames are blurry; "
            f"longest blurry stretch {longest_bad_s:.0f} s starting at {start_s:.0f} s"
        )
    return CheckResult(
        check="blur", passed=bad_share <= max_bad_share, bad_share=bad_share,
        longest_bad_s=longest_bad_s, longest_bad_start_s=start_s, note=note, scores=scores,
    )


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