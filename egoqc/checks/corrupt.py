# Corrupt file check: can every Nth frame up to the promised frame count actually be decoded?
#
# Found by experiment (2026-11-17): garbage written into the middle of an mp4 does NOT make
# OpenCV report an error. The file opens, the header still promises every frame, and reading
# just stops at the damage. So the signal is "frames the header promises that we never reach",
# plus any frame whose pixels fail to decode. Files that do not open at all are already
# flagged by the loader (clip.error) and fail here with NaN.
import cv2

from egoqc.checks.result import CheckResult
from egoqc.checks.summary import summarize_marks
from egoqc.clip import Clip


def decode_marks(clip: Clip, every_nth: int) -> list[tuple[float, float]]:
    """(time, 1.0 if decoded else 0.0) for frames 0, N, 2N, ... up to the promised count."""
    if clip.error:
        return []

    decoded = {}  # frame index -> did its pixels decode
    cap = cv2.VideoCapture(str(clip.path))
    index = 0
    try:
        while cap.grab():
            if index % every_nth == 0:
                ok, frame = cap.retrieve()
                decoded[index] = bool(ok) and frame is not None and frame.size > 0
            index += 1
    finally:
        cap.release()
    return marks_from_decoded(clip, decoded, every_nth)


def marks_from_decoded(clip: Clip, decoded: dict[int, bool], every_nth: int):
    """Turn {frame index: decoded?} into marks for every Nth frame the header promises."""
    if clip.error:
        return []
    expected = round(clip.duration * clip.fps)  # the loader measured duration as frames / fps
    # A position we never reached is as bad as one that would not decode
    return [
        (i / clip.fps, 1.0 if decoded.get(i, False) else 0.0)
        for i in range(0, expected, every_nth)
    ]


def summarize_decoding(marks, max_bad_share) -> CheckResult:
    bad = [ok == 0.0 for _, ok in marks]
    result = summarize_marks("corrupt", marks, bad, max_bad_share, what="undecodable")
    if not marks:
        result.note = "file could not be opened"
    return result


def corrupt_check(clip: Clip, every_nth, max_bad_share) -> CheckResult:
    return summarize_decoding(decode_marks(clip, every_nth), max_bad_share)


def corrupt_check_with_config(clip: Clip, config: dict) -> CheckResult:
    """Run the corrupt file check with settings from an already loaded config dict."""
    settings = config.get("checks", {}).get("corrupt")
    if settings is None:
        raise ValueError("config.yaml has no 'checks:' -> 'corrupt:' section")
    return corrupt_check(
        clip, every_nth=settings["every_nth"], max_bad_share=settings["max_bad_share"]
    )
