# Build a small demo dataset: one real 20 s clip plus copies with one known defect each,
# so the QC report and dashboard show what each check catches.
#
#   python -m scripts.make_demo_set            (needs ffmpeg on PATH)
#   python -m egoqc check data/demo
#
# Writes data/demo in the same LeRobot layout as Humanola, so the normal loader reads it.
import json
import os
import shutil
import subprocess
from pathlib import Path

SOURCE = Path("data/humanola_sample/videos/chunk-000/observation.images.main_stereo_left/episode_000004.mp4")
OUT = Path("data/demo")
CAMERA = "observation.images.main_stereo_left"
SECONDS = 20

# (label shown in the dashboard, ffmpeg video filter or None, extra ffmpeg args)
CLIPS = [
    ("clean: untouched recording", None, []),
    ("blurred: lens smear / out of focus", "gblur=sigma=12", []),
    ("dark: underexposed", "lutyuv=y=val*0.25", []),
    ("frozen: camera stalled 8 s", "loop=loop=240:size=1:start=240,setpts=N/30/TB", []),
    ("dropped frames: 90 frames lost", r"select='not(between(n\,150\,179)+between(n\,400\,459))'",
     ["-fps_mode", "passthrough"]),
    ("corrupt: damaged in the middle", None, []),  # bytes overwritten after encoding
    ("low resolution: 1280x720", "scale=1280:720", []),
    ("no hands: camera aimed above the task", "crop=iw/2:ih/2:iw/4:0,scale=1920:1200", []),
]


def encode(target: Path, video_filter, extra):
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(SOURCE), "-t", str(SECONDS), "-an"]
    if video_filter:
        cmd += ["-vf", video_filter]
    cmd += extra + ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                    "-pix_fmt", "yuv420p", str(target)]
    subprocess.run(cmd, check=True)


def damage_middle(path: Path):
    """Overwrite a slice of the middle of the file with random bytes."""
    data = bytearray(path.read_bytes())
    start, length = len(data) // 2, len(data) // 10
    data[start:start + length] = os.urandom(length)
    path.write_bytes(bytes(data))


def main():
    if not SOURCE.exists():
        raise SystemExit(f"source clip not found: {SOURCE} (download humanola_sample first)")
    if OUT.exists():
        shutil.rmtree(OUT)
    videos = OUT / "videos" / "chunk-000" / CAMERA
    videos.mkdir(parents=True)
    (OUT / "meta").mkdir()

    episodes = []
    for index, (label, video_filter, extra) in enumerate(CLIPS):
        target = videos / f"episode_{index:06d}.mp4"
        encode(target, video_filter, extra)
        if label.startswith("corrupt"):
            damage_middle(target)
        episodes.append({"episode_index": index, "tasks": [label]})
        print(f"episode_{index:06d}.mp4  {label}")

    info = {
        "fps": 30,
        "chunks_size": 1000,
        "total_episodes": len(CLIPS),
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    }
    (OUT / "meta" / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    with open(OUT / "meta" / "episodes.jsonl", "w", encoding="utf-8") as f:
        for episode in episodes:
            f.write(json.dumps(episode) + "\n")
    print(f"wrote {len(CLIPS)} clips to {OUT}")


if __name__ == "__main__":
    main()
