# Frame drops check: gaps between frame timestamps that are larger than 1 / fps
from pathlib import Path

import cv2

from egoqc.clip import Clip
from egoqc.checks.result import CheckResult



def read_timestamps(clip: Clip) -> list[float]:
    """Return the stored timestamp (in seconds) of EVERY frame in the clip."""
    if clip.error:  # nothing to read from a clip that did not open
        return []

    timestamps = []  # fresh list on every call
    cap = cv2.VideoCapture(str(clip.path))
    try:
        while cap.grab():  # grab() skips the pixel conversion; we only need the time
            timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)
    finally:
        cap.release()  # runs even if something above fails
    return timestamps



def expected_gap_of(timestamps, fps) -> float:
    """The clip's real frame interval: the median gap between timestamps.

    Not 1 / fps: OpenCV reports the AVERAGE frame rate, and a clip that lost frames has a
    lower average (a 30 fps demo clip with 90 frames lost reads as 25.5 fps), which would
    miscount the very drops we look for. Drops are rare, so the median ignores them.
    """
    if len(timestamps) < 3:
        return 1 / fps
    gaps = sorted(later - earlier for earlier, later in zip(timestamps, timestamps[1:]))
    return gaps[len(gaps) // 2]


def find_drops(timestamps, fps):
    expected_gap = expected_gap_of(timestamps, fps)
    drops = []
    for earlier, later in zip(timestamps, timestamps[1:]):
        gap = later - earlier

        if gap > 1.5 * expected_gap:
            frames_lost = round(gap / expected_gap) - 1
            drops.append((earlier, frames_lost))

    return drops


def summarize_drops(timestamps, fps, max_dropped_share) -> CheckResult:
    """Turn a clip's frame timestamps into the frame drops verdict (no video needed)."""
    if not timestamps:  # nothing read, for example a corrupt file: unmeasured, so NaN
        return CheckResult(
            check="frame_drops", passed=False, bad_share=float("nan"),
            longest_bad_s=float("nan"), longest_bad_start_s=None,
            note="no frames could be read", scores=[],
        )

    drops = find_drops(timestamps, fps)
    frames_lost_total = sum(lost for _, lost in drops)
    # The clip should have had the lost frames too, so they count in the total
    bad_share = frames_lost_total / (len(timestamps) + frames_lost_total)

    if not drops:
        longest_bad_s, start_s = 0.0, None  # measured and found nothing, so 0 is honest
        note = "no dropped frames"
    else:
        start_s, biggest = max(drops, key=lambda drop: drop[1])  # drop with most frames lost
        longest_bad_s = biggest * expected_gap_of(timestamps, fps)  # seconds of video missing
        note = (
            f"{len(drops)} drops, {frames_lost_total} frames lost ({bad_share:.2%} of the clip); "
            f"biggest gap {longest_bad_s:.2f} s at {start_s:.1f} s"
        )

    return CheckResult(
        check="frame_drops", passed=bad_share <= max_dropped_share, bad_share=bad_share,
        longest_bad_s=longest_bad_s, longest_bad_start_s=start_s, note=note,
        scores=[(time, float(lost)) for time, lost in drops],
    )


def frame_drops_check(clip: Clip, max_dropped_share) -> CheckResult:
    return summarize_drops(read_timestamps(clip), clip.fps, max_dropped_share)


def frame_drops_check_with_config(clip: Clip, config: dict) -> CheckResult:
    """Run the frame drops check with settings from an already loaded config dict."""
    settings = config.get("checks", {}).get("frame_drops")
    if settings is None:
        raise ValueError("config.yaml has no 'checks:' -> 'frame_drops:' section")
    return frame_drops_check(clip, max_dropped_share=settings["max_dropped_share"])


if __name__ == "__main__":
    from egoqc.loaders.humanola import load_humanola

    DATASET = Path("data/humanola_sample")
    clips = [c for c in load_humanola(DATASET) if not c.error]  # skip clips that did not open
    clip = clips[0]

    timestamps = read_timestamps(clip)
    gaps = [b - a for a, b in zip(timestamps, timestamps[1:])]
    print(f"clip: {clip.path.name}  expected gap: {1 / clip.fps:.4f} s")
    print(f"timestamps read: {len(timestamps)}")
    print(f"first 5: {[round(t, 4) for t in timestamps[:5]]}")
    print(f"smallest gap: {min(gaps):.4f} s  largest gap: {max(gaps):.4f} s")

    # Real clip: should have no drops
    print(f"drops in real clip: {find_drops(timestamps, clip.fps)}")

    # Fake damage: remove 1 frame at index 100 and 3 frames at index 500
    broken = list(timestamps)
    del broken[100]
    del broken[500:503]
    print(f"drops in broken copy: {[(round(t, 2), n) for t, n in find_drops(broken, clip.fps)]}")

    # Full verdicts, threshold from config.yaml
    from egoqc.config import load_config

    config = load_config()
    share = config["checks"]["frame_drops"]["max_dropped_share"]
    result = frame_drops_check_with_config(clip, config)
    print(f"real clip:   passed={result.passed}  {result.note}")
    result = summarize_drops(broken, clip.fps, share)
    print(f"broken copy: passed={result.passed}  {result.note}")
