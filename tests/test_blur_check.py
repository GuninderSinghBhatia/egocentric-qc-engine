"""Tests for the blur check, using synthetic images and a tiny fake video (no real data needed)."""
import math

import cv2
import numpy as np
import pytest

from egoqc.checks.blur import blur_check, blur_check_with_config, blur_score, summarize_scores
from egoqc.clip import Clip

WIDTH, HEIGHT, FPS = 160, 120, 10


def sharp_image():
    """Vertical black/white stripes, 2 px wide: lots of strong edges."""
    row = np.tile(np.array([0, 0, 255, 255], dtype=np.uint8), WIDTH // 4)
    gray = np.tile(row, (HEIGHT, 1))
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def blurry_image():
    return cv2.GaussianBlur(sharp_image(), (31, 31), 0)


def dark_image():
    return np.full((HEIGHT, WIDTH, 3), 10, dtype=np.uint8)  # nearly black, no edges


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


# --- blur_score on single images -------------------------------------------------

def test_sharp_image_scores_higher_than_blurry_one():
    assert blur_score(sharp_image()) > 10 * blur_score(blurry_image())


def test_sharp_image_is_above_and_blurry_below_the_starting_threshold():
    assert blur_score(sharp_image()) > 100
    assert blur_score(blurry_image()) < 100


def test_dark_flat_image_scores_as_blurry():
    # no edges at all, so the blur check cannot tell "dark" from "blurry" (brightness check's job)
    assert blur_score(dark_image()) < 100


def test_score_works_on_different_frame_sizes():
    big = cv2.resize(sharp_image(), (WIDTH * 4, HEIGHT * 4), interpolation=cv2.INTER_NEAREST)
    assert blur_score(big, width=640) > 100  # resized to a common width before scoring


# --- summarize_scores: facts and verdict, numbers only ----------------------------

def samples(values):
    return [(float(t), v) for t, v in enumerate(values)]  # 1 s apart


def test_all_good_samples_pass_with_no_bad_stretch():
    r = summarize_scores(samples([500] * 10), score_threshold=100, max_bad_share=0.10)
    assert r.passed and r.bad_share == 0
    assert r.longest_bad_s == 0 and r.longest_bad_start_s is None
    assert r.note == "no blurry frames"


def test_longest_run_and_start_are_found_among_several_runs():
    values = [500] * 20
    for t in range(7, 12):  # a 5-sample bad run starting at t=7
        values[t] = 20
    values[15] = 20  # a stray single bad sample
    r = summarize_scores(samples(values), score_threshold=100, max_bad_share=0.10)
    assert r.bad_share == pytest.approx(6 / 20)
    assert r.longest_bad_s == 5
    assert r.longest_bad_start_s == 7
    assert not r.passed


def test_short_glitch_does_not_fail_the_clip_but_is_reported():
    values = [500] * 100
    values[40:43] = [20, 20, 20]  # 3 bad samples out of 100
    r = summarize_scores(samples(values), score_threshold=100, max_bad_share=0.10)
    assert r.passed  # 3% is under the 10% limit
    assert r.longest_bad_s == 3 and r.longest_bad_start_s == 40
    assert "3 s starting at 40 s" in r.note  # the report still says it happened


def test_share_exactly_at_the_limit_passes():
    r = summarize_scores(samples([20] + [500] * 9), score_threshold=100, max_bad_share=0.10)
    assert r.bad_share == pytest.approx(0.10)
    assert r.passed


def test_score_equal_to_threshold_is_not_bad():
    r = summarize_scores(samples([100.0] * 5), score_threshold=100, max_bad_share=0.0)
    assert r.passed and r.bad_share == 0


def test_run_at_the_very_end_is_counted():
    r = summarize_scores(samples([500] * 5 + [20] * 3), score_threshold=100, max_bad_share=0.0)
    assert r.longest_bad_s == 3 and r.longest_bad_start_s == 5


def test_every_score_is_kept_in_the_result():
    data = samples([500, 20, 500])
    assert summarize_scores(data, 100, 0.10).scores == data


def test_no_samples_fails_with_nan_not_zero():
    r = summarize_scores([], score_threshold=100, max_bad_share=0.10)
    assert not r.passed
    assert math.isnan(r.bad_share)
    assert r.note == "no frames could be scored"


# --- blur_check on a tiny fake video -----------------------------------------------

def test_sharp_video_passes(tmp_path):
    clip = make_clip(tmp_path / "sharp.mp4", [sharp_image()] * 30)
    r = blur_check(clip, score_threshold=100, max_bad_share=0.10, width=160)
    assert r.passed and r.bad_share == 0
    assert len(r.scores) == 3  # 30 frames at 10 fps = 3 s = 3 samples


def test_blurry_video_fails(tmp_path):
    clip = make_clip(tmp_path / "blurry.mp4", [blurry_image()] * 30)
    r = blur_check(clip, score_threshold=100, max_bad_share=0.10, width=160)
    assert not r.passed and r.bad_share == 1.0


def test_half_blurry_video_reports_where_the_blur_starts(tmp_path):
    frames = [sharp_image()] * 50 + [blurry_image()] * 50  # 10 s, blur from 5 s on
    clip = make_clip(tmp_path / "half.mp4", frames)
    r = blur_check(clip, score_threshold=100, max_bad_share=0.10, width=160)
    assert not r.passed
    assert r.bad_share == pytest.approx(0.5)
    assert r.longest_bad_s == 5 and r.longest_bad_start_s == 5


def test_clip_with_error_is_not_scored(tmp_path):
    clip = Clip(path=tmp_path / "missing.mp4", source_dataset="fake", error="video file missing")
    r = blur_check(clip, score_threshold=100, max_bad_share=0.10, width=160)
    assert not r.passed and r.scores == []


# --- settings from config ----------------------------------------------------------

def test_settings_come_from_the_config_dict(tmp_path):
    clip = make_clip(tmp_path / "sharp.mp4", [sharp_image()] * 30)
    lenient = {"checks": {"blur": {"score_threshold": 100, "max_bad_share": 0.10, "width": 160}}}
    strict = {"checks": {"blur": {"score_threshold": 10**9, "max_bad_share": 0.10, "width": 160}}}
    assert blur_check_with_config(clip, lenient).passed
    assert not blur_check_with_config(clip, strict).passed  # same clip, only the config changed


def test_missing_blur_section_gives_a_clear_error():
    with pytest.raises(ValueError, match="blur"):
        blur_check_with_config(None, {"datasets": {}})
