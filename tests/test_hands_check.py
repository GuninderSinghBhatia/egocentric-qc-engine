"""Tests for the hand visibility check (MediaPipe on sampled frames)."""
import math
import os

import numpy as np
import pytest

from egoqc.checks.hands import hands_check_with_config, hands_score, summarize_hands
from egoqc.clip import Clip

HAND_MODEL = "models/hand_landmarker.task"
needs_hand_model = pytest.mark.skipif(
    not os.path.exists(HAND_MODEL), reason="MediaPipe hand model not downloaded"
)


# --- summarize_hands (no model needed) ----------------------------------------------------------

def test_hands_in_every_sample_passes():
    result = summarize_hands([(t, 2.0) for t in range(10)], max_bad_share=0.5)
    assert result.passed and result.bad_share == 0.0 and result.check == "hands"


def test_one_hand_counts_as_visible():
    assert summarize_hands([(t, 1.0) for t in range(10)], max_bad_share=0.5).bad_share == 0.0


def test_mostly_handless_clip_fails_and_reports_the_stretch():
    scores = [(t, 2.0) for t in range(3)] + [(t, 0.0) for t in range(3, 10)]
    result = summarize_hands(scores, max_bad_share=0.5)
    assert not result.passed
    assert result.bad_share == pytest.approx(0.7)
    assert result.longest_bad_start_s == 3
    assert "handless" in result.note


def test_exactly_at_the_limit_passes():
    scores = [(t, 0.0) for t in range(5)] + [(t, 1.0) for t in range(5, 10)]
    assert summarize_hands(scores, max_bad_share=0.5).passed


def test_nothing_sampled_fails_with_nan():
    result = summarize_hands([], max_bad_share=0.5)
    assert not result.passed and math.isnan(result.bad_share)


# --- with the real model --------------------------------------------------------------------------

@needs_hand_model
def test_blank_frame_has_no_hands():
    assert hands_score(np.zeros((400, 640, 3), np.uint8), HAND_MODEL) == 0.0


@needs_hand_model
def test_score_is_a_count_between_0_and_2():
    frame = np.random.default_rng(0).integers(0, 256, (400, 640, 3), dtype=np.uint8)
    assert hands_score(frame, HAND_MODEL) in (0.0, 1.0, 2.0)


def test_missing_model_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="hand model"):
        hands_score(np.zeros((40, 64, 3), np.uint8), str(tmp_path / "nope.task"))


def test_with_config_without_section_raises(tmp_path):
    clip = Clip(path=tmp_path / "x.mp4", source_dataset="fake", error="video file missing")
    with pytest.raises(ValueError, match="hands"):
        hands_check_with_config(clip, {"checks": {}})


def test_with_config_on_unreadable_clip_fails_with_nan(tmp_path):
    clip = Clip(path=tmp_path / "x.mp4", source_dataset="fake", error="video file missing")
    config = {"checks": {"hands": {"model_path": HAND_MODEL, "width": 640, "max_bad_share": 0.5}}}
    result = hands_check_with_config(clip, config)
    assert not result.passed and math.isnan(result.bad_share)
