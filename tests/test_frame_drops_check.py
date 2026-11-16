"""Tests for the frame drops check, using made-up timestamp lists and a tiny fake video."""
import math

import cv2
import numpy as np
import pytest

from egoqc.checks.frame_drops import (
    find_drops,
    frame_drops_check,
    frame_drops_check_with_config,
    read_timestamps,
    summarize_drops,
)
from egoqc.clip import Clip

FPS = 30.0
MAX_SHARE = 0.01  # the starting value from config.yaml


def even_timestamps(count, fps=FPS):
    """Perfect timestamps, as a clip with no drops would have."""
    return [i / fps for i in range(count)]


def without(timestamps, *indexes):
    """Copy of the list with the given frame indexes removed (fake dropped frames)."""
    gone = set(indexes)
    return [t for i, t in enumerate(timestamps) if i not in gone]


# --- find_drops --------------------------------------------------------------------

def test_clean_timestamps_have_no_drops():
    assert find_drops(even_timestamps(300), FPS) == []


def test_one_missing_frame_is_one_drop_of_one_frame():
    drops = find_drops(without(even_timestamps(300), 100), FPS)
    assert len(drops) == 1
    time, lost = drops[0]
    assert time == pytest.approx(99 / FPS)  # the gap starts at the frame before the hole
    assert lost == 1


def test_three_missing_frames_in_a_row_are_one_drop_of_three():
    drops = find_drops(without(even_timestamps(300), 50, 51, 52), FPS)
    assert [lost for _, lost in drops] == [3]


def test_separate_holes_are_separate_drops():
    drops = find_drops(without(even_timestamps(300), 10, 200, 201), FPS)
    assert [lost for _, lost in drops] == [1, 2]


def test_small_jitter_is_not_a_drop():
    timestamps = even_timestamps(300)
    timestamps[100] += 0.4 / FPS  # 40% late: the gaps become 1.4x and 0.6x, below 1.5x
    assert find_drops(timestamps, FPS) == []


def test_works_at_other_frame_rates():
    fps = 29.97
    drops = find_drops(without(even_timestamps(300, fps), 150), fps)
    assert [lost for _, lost in drops] == [1]


def test_too_few_timestamps_give_no_drops():
    assert find_drops([], FPS) == []
    assert find_drops([0.0], FPS) == []


# --- summarize_drops (verdict and facts) ---------------------------------------------

def test_clean_clip_passes_with_zero_share():
    result = summarize_drops(even_timestamps(300), FPS, MAX_SHARE)
    assert result.check == "frame_drops"
    assert result.passed
    assert result.bad_share == 0.0
    assert result.longest_bad_s == 0.0
    assert result.longest_bad_start_s is None
    assert result.scores == []
    assert result.note == "no dropped frames"


def test_share_counts_lost_frames_against_the_full_length():
    # 1000 frames, 5 removed: 995 read + 5 lost = 1000 expected, share 0.005
    timestamps = without(even_timestamps(1000), 100, 300, 500, 700, 900)
    result = summarize_drops(timestamps, FPS, MAX_SHARE)
    assert result.bad_share == pytest.approx(5 / 1000)
    assert result.passed  # 0.5% is under the 1% limit


def test_too_many_lost_frames_fail():
    timestamps = without(even_timestamps(1000), *range(400, 420))  # 20 of 1000 = 2%
    result = summarize_drops(timestamps, FPS, MAX_SHARE)
    assert result.bad_share == pytest.approx(0.02)
    assert not result.passed


def test_share_exactly_at_the_limit_passes():
    timestamps = without(even_timestamps(1000), *range(400, 410))  # 10 of 1000 = 1%
    assert summarize_drops(timestamps, FPS, MAX_SHARE).passed


def test_biggest_drop_is_reported_with_its_length_and_start():
    # a 1-frame hole early, a 6-frame hole later
    timestamps = without(even_timestamps(1000), 50, *range(600, 606))
    result = summarize_drops(timestamps, FPS, MAX_SHARE)
    assert result.longest_bad_s == pytest.approx(6 / FPS)
    assert result.longest_bad_start_s == pytest.approx(599 / FPS)
    assert len(result.scores) == 2
    assert all(type(lost) is float for _, lost in result.scores)
    assert "2 drops" in result.note and "7 frames lost" in result.note


def test_nothing_read_fails_with_nan_not_zero():
    result = summarize_drops([], FPS, MAX_SHARE)
    assert not result.passed
    assert math.isnan(result.bad_share)
    assert math.isnan(result.longest_bad_s)
    assert result.longest_bad_start_s is None
    assert result.note == "no frames could be read"


# --- read_timestamps and the full check on real (tiny) video files --------------------

def make_clip(path, frame_count, fps=10):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (160, 120))
    for _ in range(frame_count):
        writer.write(np.full((120, 160, 3), 128, dtype=np.uint8))
    writer.release()
    return Clip(path=path, source_dataset="fake", fps=float(fps),
                duration=frame_count / fps, resolution=(160, 120))


def test_read_timestamps_gives_one_even_timestamp_per_frame(tmp_path):
    clip = make_clip(tmp_path / "clip.mp4", 25)
    timestamps = read_timestamps(clip)
    assert len(timestamps) == 25
    assert timestamps[0] == pytest.approx(0.0)
    gaps = [b - a for a, b in zip(timestamps, timestamps[1:])]
    assert all(gap == pytest.approx(0.1, abs=1e-3) for gap in gaps)


def test_full_check_on_a_clean_video_passes(tmp_path):
    clip = make_clip(tmp_path / "clip.mp4", 25)
    result = frame_drops_check(clip, MAX_SHARE)
    assert result.passed
    assert result.bad_share == 0.0


def test_clip_with_error_reads_nothing_and_fails(tmp_path):
    clip = Clip(path=tmp_path / "missing.mp4", source_dataset="fake", error="file missing")
    assert read_timestamps(clip) == []
    result = frame_drops_check(clip, MAX_SHARE)
    assert not result.passed
    assert math.isnan(result.bad_share)


def test_unreadable_file_without_error_flag_also_fails(tmp_path):
    path = tmp_path / "garbage.mp4"
    path.write_bytes(b"not a video")
    clip = Clip(path=path, source_dataset="fake", fps=FPS)
    result = frame_drops_check(clip, MAX_SHARE)
    assert not result.passed
    assert math.isnan(result.bad_share)


# --- config -------------------------------------------------------------------------

def test_with_config_uses_the_threshold_from_config(tmp_path):
    clip = make_clip(tmp_path / "clip.mp4", 25)
    config = {"checks": {"frame_drops": {"max_dropped_share": MAX_SHARE}}}
    assert frame_drops_check_with_config(clip, config).passed


def test_with_config_without_section_raises(tmp_path):
    clip = make_clip(tmp_path / "clip.mp4", 25)
    with pytest.raises(ValueError, match="frame_drops"):
        frame_drops_check_with_config(clip, {"checks": {}})


def test_drops_are_counted_from_the_real_frame_gap_not_the_average_fps():
    # 600 frames at 30 fps with 90 removed: the AVERAGE rate falls to 25.5 fps, which is what
    # OpenCV reports. Using it as the expected gap would miscount the drops.
    timestamps = without(even_timestamps(600), *range(150, 180), *range(400, 460))
    drops = find_drops(timestamps, fps=25.5)
    assert [lost for _, lost in drops] == [30, 60]
    result = summarize_drops(timestamps, 25.5, MAX_SHARE)
    assert result.longest_bad_s == pytest.approx(60 / FPS)
