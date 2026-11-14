# Exploration only (not part of the engine): look at blur scores to choose a threshold
# Run from the repo root:  uv run python -m scripts.explore_blur
from pathlib import Path

import cv2
import numpy as np

from egoqc.checks.blur import blur_score
from egoqc.checks.sampling import sample_frames
from egoqc.loaders.humanola import load_humanola

DATASET = Path("data/humanola_sample")
OUT = Path("outputs")

clips = [c for c in load_humanola(DATASET) if not c.error]  # skip clips that did not open

# Pass 1: score every sampled frame. Frames are NOT kept (1,600 frames would not fit in memory).
results = []  # (clip index, clip name, time in s, score)
for i, clip in enumerate(clips):
    for time, frame in sample_frames(clip):
        results.append((i, clip.path.name, time, blur_score(frame)))

print(len(results), "scored frames")
print(results[:3])

# Per clip: how the scores are spread out
for i, clip in enumerate(clips):
    scores = [r[3] for r in results if r[0] == i]
    print(
        clip.path.name,
        "min", round(min(scores), 1),
        "p5", round(float(np.percentile(scores, 5)), 1),
        "median", round(float(np.median(scores)), 1),
        "max", round(max(scores), 1),
    )

# Pick three frames across ALL clips: the lowest, one near the 5th percentile, one at the median
all_scores = np.array([r[3] for r in results])
targets = {
    "lowest": all_scores.min(),
    "p5": np.percentile(all_scores, 5),
    "median": np.median(all_scores),
}
for label, target in targets.items():
    i, name, time, score = min(results, key=lambda r: abs(r[3] - target))
    # Pass 2: re-read that one clip until we reach the chosen time, and save that frame
    for t, frame in sample_frames(clips[i]):
        if t == time:
            path = OUT / f"blur_{label}_{score:.0f}.png"
            cv2.imwrite(str(path), frame)
            print(label, name, f"t={time:.0f}s", f"score={score:.1f}", "->", path)
            break
