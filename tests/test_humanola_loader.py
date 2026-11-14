"""Tests for the Humanola loader, using tiny fake datasets built in a temp folder.

The real data/ folder is git-ignored, so nothing here depends on it.
"""
import json
import math

import cv2
import numpy as np
import pytest
import yaml

from egoqc.loaders.humanola import load_humanola

CAMERA = "observation.images.cam"
TEMPLATE = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
WIDTH, HEIGHT, FPS, FRAMES = 64, 48, 10, 10


def write_video(path):
    """Write a tiny real mp4 (64x48, 10 fps, 10 frames) so OpenCV can open it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT)
    )
    for i in range(FRAMES):
        writer.write(np.full((HEIGHT, WIDTH, 3), i * 20, dtype=np.uint8))
    writer.release()


def make_dataset(tmp_path, name="fake_ds", episodes=3, chunks_size=1000):
    """Build a fake LeRobot-layout dataset plus a config.yaml that knows about it."""
    root = tmp_path / name
    (root / "meta").mkdir(parents=True)
    info = {
        "fps": FPS,
        "total_episodes": episodes,
        "video_path": TEMPLATE,
        "chunks_size": chunks_size,
    }
    (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    lines = [
        json.dumps({"episode_index": i, "duration_s": FRAMES / FPS, "length": FRAMES})
        for i in range(episodes)
    ]
    (root / "meta" / "episodes.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for i in range(episodes):
        write_video(video_file(root, i, chunks_size))

    config = {"datasets": {name: {"camera": CAMERA}}}
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return root, config_path


def video_file(root, index, chunks_size=1000):
    rel = TEMPLATE.format(
        episode_chunk=index // chunks_size, video_key=CAMERA, episode_index=index
    )
    return root / rel


def test_yields_one_clip_per_episode_with_measured_values(tmp_path):
    root, config = make_dataset(tmp_path, episodes=3)
    clips = list(load_humanola(root, config))
    assert len(clips) == 3
    for clip in clips:
        assert clip.error is None
        assert clip.source_dataset == "fake_ds"
        assert clip.resolution == (WIDTH, HEIGHT)  # (width, height)
        assert clip.fps == pytest.approx(FPS)
        assert clip.duration == pytest.approx(FRAMES / FPS)
        assert clip.pose_path is None
    assert [c.path.name for c in clips] == [f"episode_{i:06d}.mp4" for i in range(3)]


def test_missing_video_is_yielded_with_error_and_nan(tmp_path):
    root, config = make_dataset(tmp_path, episodes=3)
    video_file(root, 1).unlink()
    clips = list(load_humanola(root, config))
    assert len(clips) == 3  # the loader does not stop
    bad = clips[1]
    assert bad.error == "video file missing"
    assert math.isnan(bad.duration)
    assert math.isnan(bad.fps)
    assert bad.resolution is None


def test_good_clip_after_a_bad_one_does_not_inherit_its_values(tmp_path):
    root, config = make_dataset(tmp_path, episodes=3)
    video_file(root, 0).unlink()
    clips = list(load_humanola(root, config))
    assert clips[0].error == "video file missing"
    assert clips[1].error is None
    assert clips[1].duration == pytest.approx(FRAMES / FPS)
    assert clips[1].resolution == (WIDTH, HEIGHT)


def test_corrupt_video_is_yielded_with_error(tmp_path):
    root, config = make_dataset(tmp_path, episodes=2)
    video_file(root, 0).write_text("this is not a video", encoding="utf-8")
    clips = list(load_humanola(root, config))
    assert len(clips) == 2
    assert clips[0].error == "could not open video"
    assert math.isnan(clips[0].duration)
    assert clips[0].resolution is None
    assert clips[1].error is None


def test_episodes_use_the_chunk_folder(tmp_path):
    # chunks_size=2: episodes 0-1 in chunk-000, episode 2 in chunk-001
    root, config = make_dataset(tmp_path, episodes=3, chunks_size=2)
    clips = list(load_humanola(root, config))
    assert all(c.error is None for c in clips)
    assert "chunk-001" in clips[2].path.parts[-3]


def test_blank_line_at_end_of_episodes_file_is_ignored(tmp_path):
    root, config = make_dataset(tmp_path, episodes=2)
    path = root / "meta" / "episodes.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + "\n\n", encoding="utf-8")
    assert len(list(load_humanola(root, config))) == 2


def test_unknown_dataset_name_lists_known_ones(tmp_path):
    root, config = make_dataset(tmp_path, name="fake_ds")
    other = tmp_path / "not_registered"
    other.mkdir()
    with pytest.raises(ValueError, match="fake_ds"):
        list(load_humanola(other, config))


def test_missing_info_json_raises(tmp_path):
    root, config = make_dataset(tmp_path)
    (root / "meta" / "info.json").unlink()
    with pytest.raises(FileNotFoundError, match="info.json"):
        list(load_humanola(root, config))


def test_missing_episodes_file_raises(tmp_path):
    root, config = make_dataset(tmp_path)
    (root / "meta" / "episodes.jsonl").unlink()
    with pytest.raises(FileNotFoundError, match="episodes.jsonl"):
        list(load_humanola(root, config))
