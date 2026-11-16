# Results

Only numbers measured on this repo's own runs. Reproduce with `python -m egoqc check <dataset>`.

## 1. Real data: Humanola egocentric sample

9 clips, 26.8 minutes, 1920x1200 at 30 fps, 1,617 sampled seconds.

**All 9 clips pass all seven checks.** This is a curated, published dataset, so a clean result
is expected; the point of this run is the score distributions it gives for setting thresholds.

| Score | Lowest | 5th percentile | Median | Highest | Threshold |
| --- | --- | --- | --- | --- | --- |
| Blur (Laplacian variance) | 150 | 529 | 994 | 5,221 | bad below 100 |
| Brightness (mean gray) | 65 | 77 | 96 | 151 | bad outside 40-215 |
| Frozen (difference to next frame) | 0.41 | 0.92 | 3.66 | 33.6 | bad below 0.1 |
| Frame gaps | 1/30 s on every frame of every clip | | | | drop above 1.5x |

### Hand visibility: MediaPipe vs the dataset's own tracking

| Clip | Task | Dataset tracking coverage | MediaPipe finds a hand |
| --- | --- | --- | --- |
| episode_000000 | ironing clothes | 100% | 75.4% |
| episode_000001 | folding laundry | 99.99% | 81.1% |
| episode_000002 | wiping | 99.99% | 68.9% |
| episode_000003 | washing dishes | 100% | 85.4% |
| episode_000004 | slicing carrots | 100% | 92.8% |
| episode_000005 | peeling and slicing cucumbers | 100% | 90.1% |
| episode_000006 | pouring cereal into a container | 100% | 90.7% |
| episode_000007 | washing dishes | 100% | 99.2% |
| episode_000008 | cutting an apple | 100% | 98.8% |

The dataset's tracker reports hand pose for essentially every frame; MediaPipe finds a hand in
69-99% of sampled frames (84.5% overall). The gap is largest for whole-arm tasks (wiping,
ironing) and smallest for close-up cutting. **Not yet known:** how much of the gap is MediaPipe
missing visible hands and how much is hands that are tracked but outside this camera's view.
Measuring that frame by frame against the ground-truth pose is the next step.

## 2. Synthetic defects: the demo set

`scripts/make_demo_set.py` takes the first 20 s of a real sample clip and makes 7 copies, each
with one known defect, so every check can be seen firing.

| Clip | Defect injected | Failed checks |
| --- | --- | --- |
| episode_000000 | none | none (passes) |
| episode_000001 | Gaussian blur | blur, hands |
| episode_000002 | brightness x 0.25 | exposure, blur, hands |
| episode_000003 | one frame repeated for 8 s | frozen (40% of samples, starting at 8 s) |
| episode_000004 | 90 of 600 frames removed | frame drops (15.0%, biggest gap 2.0 s), spec (25.5 fps) |
| episode_000005 | random bytes over 10% of the file | corrupt (50% undecodable, from 10 s) |
| episode_000006 | scaled to 1280x720 | spec |
| episode_000007 | camera view moved above the task | hands (80% of samples without a hand) |

Every defect is caught by the check designed for it. Extra failures are real effects, not
false alarms: blurred and dark footage also defeats hand detection, dark frames also score low on
blur, and dropping frames lowers the average frame rate below the spec.

## 3. Speed

| Run | Time |
| --- | --- |
| 9 sample clips, six checks each reading the video separately | 11 min 45 s |
| same, single decoding pass (identical results) | 2 min 13 s |
| one labelled-set clip (3840x3360) before the hands check | 156 s |

Times are on a Windows laptop CPU, no GPU.

## 4. Not measured yet

- Precision and recall of the pass/fail decision against hand-labelled clips.
- The threshold recheck on the 38-clip labelled set (3840x3360, head camera) was still running
  when this was written.
