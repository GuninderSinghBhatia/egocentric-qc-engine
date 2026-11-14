"""Tests for the brightness check, using synthetic images and a tiny fake video."""
import math

import cv2
import numpy as np
import pytest

from egoqc.checks.brightness import (
    brightness_check,
    brightness_check_with_config,
    brightness_score,
    summarize_brightness,
)
from egoqc.clip import Clip

WIDTH, HEIGHT, FPS = 160, 120, 10
MIN_B, MAX_B, MAX_SHARE = 40, 215, 0.10  # the starting values from config.yaml


def flat_image(value):
    return np.full((HEIGHT, WIDTH, 3), value, dtype=np.uint8)


def write_video(path, frames):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    for frame in frames:
        writer.write(frame)
    writer.release()


def make_clip(path, frames):
    write_video(path, frames)
    return Clip(
        path=path, source_dataset="fake", fps=float(FPS),
        duration=len(frames) / FPS, resolution=(WIDTH, HEIGHT),
    )


# --- brightness_score on single images -------------------------------------------

def test_score_is_the_mean_gray_value():
    assert brightness_score(flat_image(0)) == pytest.approx(0)
    assert brightness_score(flat_image(128)) == pytest.approx(128)
    assert brightness_score(flat_image(255)) == pytest.approx(255)


def test_dark_normal_and_bright_images_are_ordered():
    assert brightness_score(flat_image(10)) < brightness_score(flat_image(128)) < brightness_score(flat_image(245))


def test_score_does_not_depend_on_image_size():
    small = flat_image(100)
    big = cv2.resize(small, (WIDTH * 4, HEIGHT * 4))
    assert brightness_score(big) == pytest.approx(brightness_score(small))


def test_score_is_a_plain_float():
    assert type(brightness_score(flat_image(50))) is float


# --- summarize_brightness: the "outside a range" rule ----------------------------

def samples(values):
    return [(float(t), v) for t, v in enumerate(values)]  # 1 s apart


def test_frames_inside_the_range_pass():
    r = summarize_brightness(samples([100] * 10), MIN_B, MAX_B, MAX_SHARE)
    assert r.passed and r.bad_share == 0 and r.note == "no badly exposed frames"


def test_too_dark_and_too_bright_are_both_bad():
    r = summarize_brightness(samples([10, 128, 245, 128]), MIN_B, MAX_B, max_bad_share=0.0)
    assert r.bad_share == pytest.approx(0.5)  # one dark + one bright out of four
    assert not r.passed


def test_values_exactly_on_the_limits_are_good():
    r = summarize_brightness(samples([MIN_B, MAX_B, 100]), MIN_B, MAX_B, max_bad_share=0.0)
    assert r.passed and r.bad_share == 0


def test_values_just_outside_the_limits_are_bad():
    r = summarize_brightness(samples([MIN_B - 0.1, MAX_B + 0.1]), MIN_B, MAX_B, max_bad_share=0.0)
    assert r.bad_share == 1.0


def test_dark_stretch_is_reported_with_start_and_length():
    values = [100] * 300
    values[110:135] = [5] * 25  # a 25 s dark stretch, like a sleeve over the lens
    r = summarize_brightness(samples(values), MIN_B, MAX_B, MAX_SHARE)
    assert r.passed  # 25 of 300 = 8.3%, under the 10% limit
    assert r.longest_bad_s == 25 and r.longest_bad_start_s == 110
    assert "25 s starting at 110 s" in r.note  # passes, but the report still says it happened


def test_dark_then_bright_stretches_count_as_one_bad_run():
    # both directions are "bad", so a dark stretch followed directly by a bright one is one run
    r = summarize_brightness(samples([100, 5, 5, 250, 250, 100]), MIN_B, MAX_B, MAX_SHARE)
    assert r.longest_bad_s == 4 and r.longest_bad_start_s == 1


def test_every_score_is_kept():
    data = samples([100, 5, 250])
    assert summarize_brightness(data, MIN_B, MAX_B, MAX_SHARE).scores == data


def test_no_samples_fails_with_nan():
    r = summarize_brightness([], MIN_B, MAX_B, MAX_SHARE)
    assert not r.passed and math.isnan(r.bad_share)


# --- brightness_check on a tiny fake video ---------------------------------------

def test_normal_video_passes(tmp_path):
    clip = make_clip(tmp_path / "normal.mp4", [flat_image(128)] * 30)
    r = brightness_check(clip, MIN_B, MAX_B, MAX_SHARE)
    assert r.passed and r.bad_share == 0
    assert len(r.scores) == 3  # 30 frames at 10 fps = 3 s = 3 samples
    assert all(abs(score - 128) < 10 for _, score in r.scores)


def test_dark_video_fails(tmp_path):
    clip = make_clip(tmp_path / "dark.mp4", [flat_image(10)] * 30)
    r = brightness_check(clip, MIN_B, MAX_B, MAX_SHARE)
    assert not r.passed and r.bad_share == 1.0


def test_blown_out_video_fails(tmp_path):
    clip = make_clip(tmp_path / "bright.mp4", [flat_image(250)] * 30)
    r = brightness_check(clip, MIN_B, MAX_B, MAX_SHARE)
    assert not r.passed and r.bad_share == 1.0


def test_half_dark_video_reports_where_the_dark_starts(tmp_path):
    frames = [flat_image(128)] * 50 + [flat_image(10)] * 50  # 10 s, dark from 5 s on
    clip = make_clip(tmp_path / "half.mp4", frames)
    r = brightness_check(clip, MIN_B, MAX_B, MAX_SHARE)
    assert not r.passed
    assert r.bad_share == pytest.approx(0.5)
    assert r.longest_bad_s == 5 and r.longest_bad_start_s == 5


def test_clip_with_error_is_not_scored(tmp_path):
    clip = Clip(path=tmp_path / "missing.mp4", source_dataset="fake", error="video file missing")
    r = brightness_check(clip, MIN_B, MAX_B, MAX_SHARE)
    assert not r.passed and r.scores == []


# --- settings from config ---------------------------------------------------------

def test_settings_come_from_the_config_dict(tmp_path):
    clip = make_clip(tmp_path / "normal.mp4", [flat_image(128)] * 30)
    inside = {"checks": {"brightness": {
        "minimum_brightness": 40, "maximum_brightness": 215, "max_bad_share": 0.10}}}
    too_high_min = {"checks": {"brightness": {
        "minimum_brightness": 200, "maximum_brightness": 250, "max_bad_share": 0.10}}}
    assert brightness_check_with_config(clip, inside).passed
    assert not brightness_check_with_config(clip, too_high_min).passed  # same clip, only config changed


def test_missing_brightness_section_gives_a_clear_error():
    with pytest.raises(ValueError, match="brightness"):
        brightness_check_with_config(None, {"checks": {"blur": {}}})
