# One pass through a clip that collects what every check needs, so the video is decoded once
#
# Each check also works on its own (blur_check, frozen_check, ...), but each of those walks
# the whole video. Running all of them that way took ~80 s per sample clip; this pass gives
# the same numbers with one walk. The per-sample rules match sampling.py exactly.
from dataclasses import dataclass, field

import cv2

from egoqc.checks.blur import blur_score
from egoqc.checks.brightness import brightness_score
from egoqc.checks.frozen import frozen_score
from egoqc.checks.hands import hands_score
from egoqc.clip import Clip


@dataclass
class ClipScan:
    timestamps: list[float] = field(default_factory=list)  # every frame, seconds (frame drops)
    decoded: dict[int, bool] = field(default_factory=dict)  # frame index -> decoded? (corrupt)
    blur: list[tuple[float, float]] = field(default_factory=list)  # (time, score) per sample
    brightness: list[tuple[float, float]] = field(default_factory=list)
    frozen: list[tuple[float, float]] = field(default_factory=list)
    hands: list[tuple[float, float]] = field(default_factory=list)  # (time, hands found)


def scan_clip(clip: Clip, every_nth: int, blur_width: int, frozen_width: int,
              samples_per_second: int = 1, hand_model: str | None = None,
              hand_width: int = 640) -> ClipScan:
    """hand_model: path to MediaPipe's hand model; None skips hand detection."""
    scan = ClipScan()
    if clip.error:  # nothing to read; every check turns the empty scan into NaN
        return scan

    gap = max(1, round(clip.fps / samples_per_second))  # same sample spacing as sample_frames
    cap = cv2.VideoCapture(str(clip.path))
    index = 0
    pending = None  # (time, frame) of the last sample, waiting for the frame after it
    try:
        while cap.grab():
            scan.timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)
            is_sample = index % gap == 0
            is_decode_check = index % every_nth == 0
            is_after_sample = pending is not None

            frame = None
            ok = False
            if is_sample or is_decode_check or is_after_sample:
                ok, frame = cap.retrieve()  # decode pixels only when someone needs them
                ok = bool(ok) and frame is not None and frame.size > 0

            if is_decode_check:
                scan.decoded[index] = ok
            if is_after_sample:
                sample_time, sample_frame = pending
                if ok:
                    scan.frozen.append((sample_time, frozen_score(sample_frame, frame, frozen_width)))
                pending = None
            if is_sample and ok:
                time = index / clip.fps
                scan.blur.append((time, blur_score(frame, blur_width)))
                scan.brightness.append((time, brightness_score(frame)))
                if hand_model is not None:
                    scan.hands.append((time, hands_score(frame, hand_model, hand_width)))
                pending = (time, frame)
            index += 1
    finally:
        cap.release()
    return scan
