# Exploration only (not part of the engine): look at brightness scores to choose min and max
# Run from the repo root:  uv run python -m scripts.explore_brightness
from pathlib import Path

import numpy as np

from egoqc.checks.brightness import brightness_score
from egoqc.checks.sampling import sample_frames
from egoqc.loaders.humanola import load_humanola

DATASET = Path("data/humanola_sample")
MAX_SAMPLES = 60  # per clip, to keep the run short

clips = [c for c in load_humanola(DATASET) if not c.error]  # skip clips that did not open

results = []  # (clip index, clip name, time in s, score)
for i, clip in enumerate(clips):
    count = 0  # resets for every clip
    for time, frame in sample_frames(clip):
        results.append((i, clip.path.name, time, brightness_score(frame)))
        count += 1
        if count == MAX_SAMPLES:
            break

print(len(results), "scored frames")

# Per clip: how the scores are spread out
for i, clip in enumerate(clips):
    scores = [r[3] for r in results if r[0] == i]
    print(
        clip.path.name,
        "min", round(min(scores), 1),
        "median", round(float(np.median(scores)), 1),
        "max", round(max(scores), 1),
    )

# Over all clips together
all_scores = [r[3] for r in results]
print(
    "ALL min", round(min(all_scores), 1),
    "median", round(float(np.median(all_scores)), 1),
    "max", round(max(all_scores), 1),
)

darkest = min(results, key=lambda r: r[3])
brightest = max(results, key=lambda r: r[3])
print("darkest ", darkest[1], f"t={darkest[2]:.0f}s", f"score={darkest[3]:.1f}")
print("brightest", brightest[1], f"t={brightest[2]:.0f}s", f"score={brightest[3]:.1f}")
