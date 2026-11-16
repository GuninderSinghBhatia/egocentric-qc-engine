"""Tests for the spec, frozen and corrupt checks, the pair sampler and the results table."""
import math
import os

import cv2
import numpy as np
import pytest

from egoqc.checks.corrupt import corrupt_check, corrupt_check_with_config, decode_marks
from egoqc.checks.frozen import frozen_check, frozen_check_with_config, frozen_score
from egoqc.checks.runner import CHECKS, clip_row, results_table, run_all_checks
from egoqc.checks.sampling import sample_frame_pairs, sample_frames
from egoqc.checks.spec import spec_check, spec_check_with_config
from egoqc.clip import Clip

HAND_MODEL = "models/hand_landmarker.task"  # git-ignored; tests that run the full pipeline need it
needs_hand_model = pytest.mark.skipif(
    not os.path.exists(HAND_MODEL), reason="MediaPipe hand model not downloaded"
)

WIDTH, HEIGHT, FPS = 160, 120, 10
SPEC = dict(min_width=1920, min_height=1080, fps=30, fps_tolerance=0.1)

rng = np.random.default_rng(0)


def noise_frame():
    """A frame of random pixels, so every frame clearly differs from the next."""
    return rng.integers(0, 256, (HEIGHT, WIDTH, 3), dtype=np.uint8)


def smooth_frame():
    """A random but smooth image, closer to real video than pure noise.

    Pure noise compresses so badly that the encoder's periodic keyframes re-encode a
    repeated noise image visibly differently, which would hide a freeze in the test video.
    """
    small = rng.integers(0, 256, (12, 16, 3), dtype=np.uint8)
    return cv2.resize(small, (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)


def write_clip(path, frames, fps=FPS):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (WIDTH, HEIGHT))
    for frame in frames:
        writer.write(frame)
    writer.release()
    return Clip(path=path, source_dataset="fake", fps=float(fps),
                duration=len(frames) / fps, resolution=(WIDTH, HEIGHT))


def spec_clip(width, height, fps):
    return Clip(path="x.mp4", source_dataset="fake", fps=fps, duration=10.0,
                resolution=(width, height))


# --- spec ------------------------------------------------------------------------------

def test_spec_passes_for_full_hd_30fps():
    result = spec_check(spec_clip(1920, 1080, 30.0), **SPEC)
    assert result.passed and result.bad_share == 0.0


def test_spec_accepts_2997_fps_and_bigger_frames():
    assert spec_check(spec_clip(3840, 3360, 29.97), **SPEC).passed


def test_spec_fails_low_resolution():
    result = spec_check(spec_clip(1280, 720, 30.0), **SPEC)
    assert not result.passed
    assert result.bad_share == 1.0
    assert "1280x720" in result.note


def test_spec_fails_if_only_one_side_is_too_small():
    assert not spec_check(spec_clip(1920, 1000, 30.0), **SPEC).passed


def test_spec_fails_wrong_fps():
    result = spec_check(spec_clip(1920, 1080, 25.0), **SPEC)
    assert not result.passed and "25.00 fps" in result.note


def test_spec_reports_both_problems():
    note = spec_check(spec_clip(640, 480, 15.0), **SPEC).note
    assert "640x480" in note and "15.00 fps" in note


def test_spec_unmeasured_clip_fails_with_nan():
    clip = Clip(path="x.mp4", source_dataset="fake", error="could not open video")
    result = spec_check(clip, **SPEC)
    assert not result.passed and math.isnan(result.bad_share)


def test_spec_with_config():
    config = {"checks": {"spec": SPEC}}
    assert spec_check_with_config(spec_clip(1920, 1080, 30.0), config).passed
    with pytest.raises(ValueError, match="spec"):
        spec_check_with_config(spec_clip(1920, 1080, 30.0), {"checks": {}})


# --- sampler -------------------------------------------------------------------------------

def test_sample_frames_still_one_per_second(tmp_path):
    clip = write_clip(tmp_path / "c.mp4", [noise_frame() for _ in range(35)])
    times = [t for t, _ in sample_frames(clip)]
    assert times == pytest.approx([0.0, 1.0, 2.0, 3.0])


def test_pairs_are_a_sample_and_the_frame_right_after_it(tmp_path):
    # each frame is a flat gray whose value is its index * 5, so we can tell frames apart
    # (steps of 10, tolerance 4.5: compression shifts flat grays by a few levels)
    frames = [np.full((HEIGHT, WIDTH, 3), i * 10, dtype=np.uint8) for i in range(25)]
    clip = write_clip(tmp_path / "c.mp4", frames)
    pairs = list(sample_frame_pairs(clip))
    assert [t for t, _, _ in pairs] == pytest.approx([0.0, 1.0, 2.0])
    for t, frame, next_frame in pairs:
        index = round(t * FPS)
        assert frame.mean() == pytest.approx(index * 10, abs=4.5)
        assert next_frame.mean() == pytest.approx((index + 1) * 10, abs=4.5)


def test_last_frame_as_sample_yields_no_pair(tmp_path):
    clip = write_clip(tmp_path / "c.mp4", [noise_frame() for _ in range(11)])  # sample at 10 is last
    assert [t for t, _, _ in sample_frame_pairs(clip)] == pytest.approx([0.0])


# --- frozen ---------------------------------------------------------------------------------

def test_frozen_score_is_zero_for_identical_frames():
    frame = noise_frame()
    assert frozen_score(frame, frame.copy()) == 0.0


def test_frozen_score_is_large_for_different_frames():
    assert frozen_score(noise_frame(), noise_frame()) > 10


def test_frozen_score_does_not_wrap_around():
    black = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    white = np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)
    assert frozen_score(black, white) == pytest.approx(255)
    assert frozen_score(white, black) == pytest.approx(255)


def test_moving_video_is_not_frozen(tmp_path):
    clip = write_clip(tmp_path / "c.mp4", [smooth_frame() for _ in range(50)])
    result = frozen_check(clip, difference_threshold=0.1, max_bad_share=0.05, width=160)
    assert result.passed and result.bad_share == 0.0


def test_video_that_stands_still_is_frozen(tmp_path):
    moving = [smooth_frame() for _ in range(20)]
    still = [moving[-1]] * 30  # the last image repeated for 3 seconds
    clip = write_clip(tmp_path / "c.mp4", moving + still)
    result = frozen_check(clip, difference_threshold=0.1, max_bad_share=0.05, width=160)
    assert not result.passed
    assert result.longest_bad_start_s == pytest.approx(2.0)
    assert result.bad_share == pytest.approx(3 / 5)


def test_frozen_with_config(tmp_path):
    clip = write_clip(tmp_path / "c.mp4", [noise_frame() for _ in range(20)])
    config = {"checks": {"frozen": {"difference_threshold": 0.1, "max_bad_share": 0.05, "width": 160}}}
    assert frozen_check_with_config(clip, config).passed


# --- corrupt -----------------------------------------------------------------------------------

def test_healthy_file_decodes_everywhere(tmp_path):
    clip = write_clip(tmp_path / "c.mp4", [noise_frame() for _ in range(40)])
    marks = decode_marks(clip, every_nth=10)
    assert [ok for _, ok in marks] == [1.0, 1.0, 1.0, 1.0]
    assert corrupt_check(clip, every_nth=10, max_bad_share=0.0).passed


def test_file_that_stops_before_its_promised_length_fails(tmp_path):
    clip = write_clip(tmp_path / "c.mp4", [noise_frame() for _ in range(40)])
    clip.duration = 8.0  # the header promised 80 frames, only 40 exist
    result = corrupt_check(clip, every_nth=10, max_bad_share=0.0)
    assert not result.passed
    assert result.bad_share == pytest.approx(0.5)
    assert result.longest_bad_start_s == pytest.approx(4.0)  # where the readable part ends


def test_garbage_in_the_middle_is_caught(tmp_path):
    clip = write_clip(tmp_path / "c.mp4", [noise_frame() for _ in range(100)])
    data = bytearray(clip.path.read_bytes())
    start = len(data) // 3
    data[start:start + len(data) // 4] = os.urandom(len(data) // 4)
    clip.path.write_bytes(bytes(data))
    assert not corrupt_check(clip, every_nth=10, max_bad_share=0.0).passed


def test_unopened_clip_fails_with_nan(tmp_path):
    clip = Clip(path=tmp_path / "gone.mp4", source_dataset="fake", error="video file missing")
    result = corrupt_check(clip, every_nth=10, max_bad_share=0.0)
    assert not result.passed and math.isnan(result.bad_share)
    assert result.note == "file could not be opened"


def test_corrupt_with_config(tmp_path):
    clip = write_clip(tmp_path / "c.mp4", [noise_frame() for _ in range(20)])
    config = {"checks": {"corrupt": {"every_nth": 10, "max_bad_share": 0.0}}}
    assert corrupt_check_with_config(clip, config).passed


# --- results table --------------------------------------------------------------------------------

def full_config():
    return {"checks": {
        "spec": dict(min_width=100, min_height=100, fps=10, fps_tolerance=0.1),
        "corrupt": {"every_nth": 10, "max_bad_share": 0.0},
        "frame_drops": {"max_dropped_share": 0.01},
        "frozen": {"difference_threshold": 0.1, "max_bad_share": 0.05, "width": 160},
        "blur": {"score_threshold": 100, "max_bad_share": 0.10, "width": 160},
        "brightness": {"minimum_brightness": 40, "maximum_brightness": 215, "max_bad_share": 0.10},
        # max_bad_share 1.0: the synthetic test frames have no hands, so hands must not fail them
        "hands": {"model_path": HAND_MODEL, "width": 160, "max_bad_share": 1.0},
    }}


@needs_hand_model
def test_every_check_runs_on_a_clean_clip(tmp_path):
    clip = write_clip(tmp_path / "c.mp4", [noise_frame() for _ in range(30)])
    results = run_all_checks(clip, full_config())
    assert list(results) == list(CHECKS)
    assert all(r.passed for r in results.values()), {n: r.note for n, r in results.items()}


@needs_hand_model
def test_single_pass_gives_the_same_results_as_each_check_alone(tmp_path):
    from egoqc.checks.blur import blur_check_with_config
    from egoqc.checks.brightness import brightness_check_with_config
    from egoqc.checks.frame_drops import frame_drops_check_with_config
    from egoqc.checks.hands import hands_check_with_config

    moving = [smooth_frame() for _ in range(25)]
    frames = moving + [moving[-1]] * 20 + [np.zeros((HEIGHT, WIDTH, 3), np.uint8)] * 15
    clip = write_clip(tmp_path / "c.mp4", frames)
    clip.duration = 7.0  # promise more frames than exist, so corrupt has something to find
    config = full_config()
    together = run_all_checks(clip, config)
    alone = {
        "spec": spec_check_with_config(clip, config),
        "corrupt": corrupt_check_with_config(clip, config),
        "frame_drops": frame_drops_check_with_config(clip, config),
        "frozen": frozen_check_with_config(clip, config),
        "blur": blur_check_with_config(clip, config),
        "brightness": brightness_check_with_config(clip, config),
        "hands": hands_check_with_config(clip, config),
    }
    for name in CHECKS:
        assert together[name] == alone[name], name
    # and the test clip really exercises the failure paths
    assert not together["corrupt"].passed
    assert not together["frozen"].passed
    assert not together["brightness"].passed


def test_broken_clip_gets_a_failing_row_not_a_crash(tmp_path):
    clip = Clip(path=tmp_path / "gone.mp4", source_dataset="fake", error="video file missing")
    row = clip_row(clip, run_all_checks(clip, full_config()))
    assert row["passed"] is False
    assert row["error"] == "video file missing"
    assert set(row["failed_checks"].split(", ")) == set(CHECKS)
    assert all(math.isnan(row[f"{name}_bad_share"]) for name in CHECKS)


@needs_hand_model
def test_table_has_one_row_per_clip_and_a_column_set_per_check(tmp_path):
    good = write_clip(tmp_path / "good.mp4", [noise_frame() for _ in range(30)])
    gone = Clip(path=tmp_path / "gone.mp4", source_dataset="fake", error="video file missing")
    config = full_config()
    table = results_table([clip_row(c, run_all_checks(c, config)) for c in (good, gone)])
    assert list(table["clip"]) == ["good.mp4", "gone.mp4"]
    assert list(table["passed"]) == [True, False]
    for name in CHECKS:
        assert {f"{name}_bad_share", f"{name}_passed", f"{name}_note"} <= set(table.columns)
