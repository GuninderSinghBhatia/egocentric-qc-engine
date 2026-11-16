# The checks in detail

Every check returns the same `CheckResult`: `passed`, `bad_share` (0 to 1, NaN when nothing
could be measured), the longest bad stretch in seconds and where it starts, a readable `note`,
and the raw scores. All thresholds are in `config.yaml` under `checks:`.

Five of the seven checks score individual samples (about one frame per second). They share one
rule (`egoqc/checks/summary.py`): **a clip fails only if the share of bad samples is above
`max_bad_share`**, but the report always includes the longest bad stretch, so a short glitch
passes yet stays visible.

Numbers quoted below come from the 9-clip Humanola sample set (1,617 samples, 26.8 minutes,
1920x1200 at 30 fps) unless stated otherwise.

---

## Spec (`spec.py`)

**Question:** is the clip the resolution and frame rate the delivery spec asks for?

**How:** compares the measured width, height and fps (from the decoded video, not the metadata)
with `min_width`, `min_height`, `fps` and `fps_tolerance`. One verdict for the whole clip, so
`bad_share` is 0 or 1.

**Threshold:** at least 1920x1080, 30 ± 0.1 fps (so 29.97 passes). This is a placeholder until
a real buyer spec exists. Both Humanola sets meet it.

**Note:** OpenCV reports the *average* frame rate. A clip that lost frames therefore also shows
a lower fps and can fail spec too (the demo clip with 90 lost frames reads as 25.5 fps).

## Corrupt file (`corrupt.py`)

**Question:** can the whole file actually be decoded?

**How:** walks the file and decodes every `every_nth` (30th) frame, up to the frame count the
header promises. A promised position that is never reached, or whose pixels do not decode, is
bad.

**Why this design (found by experiment):**
- An mp4 truncated part-way loses its index (`moov` atom at the end of these files) and does not
  open at all; the loader already flags that as `could not open video`.
- Random bytes written into the middle of an mp4 do **not** make OpenCV report an error. The
  file opens, the header still promises every frame, `retrieve()` never fails, and reading
  simply stops at the damage. So the reliable signal is "frames promised but never reached".
  Without this check, blur and brightness would silently score only the first half of such a
  clip.

**Threshold:** `max_bad_share: 0.0`; any undecodable sampled position fails the clip.

**Cannot see:** damage the decoder conceals by guessing pixels.

## Frame drops (`frame_drops.py`)

**Question:** are frames missing from the recording?

**How:** reads the stored timestamp of **every** frame (cheap: `grab()` only). The expected gap
is the **median** gap between timestamps, not 1/fps, because OpenCV's fps is the average and
drops pull the average down (on a clip with 90 of 600 frames removed, 1/fps counted 76 lost
frames instead of 90; the median gap counts 90). A gap above 1.5x the expected gap is a drop;
frames lost = gap / expected gap, rounded, minus 1.

`bad_share` = frames lost / (frames read + frames lost). The note lists the number of drops,
frames lost, and the biggest gap with its time.

**Threshold:** `max_dropped_share: 0.01` (1%), stricter than the sampled checks because a lost
frame is gone for good. All 9 sample clips have perfectly even 1/30 s gaps, so the threshold is
untested on real damage; the demo set shows it catches injected drops.

**Cannot see:** drops a recorder filled by repeating frames (timestamps stay even).

## Frozen video (`frozen.py`)

**Question:** did the picture stop changing (stalled camera, repeated frames)?

**How:** for each sample, compares the frame with the very next frame: both are shrunk to 160 px
wide (sensor noise averages out, resolution stops mattering) and the score is the mean absolute
gray difference. 0 means identical.

**Threshold:** a sample is frozen below `difference_threshold: 0.1`; the clip fails above
`max_bad_share: 0.05`. Real neighbouring frames never scored below 0.41 (5th percentile 0.92,
median 3.66), and a repeated frame scores about 0, so there is a 4x margin.

**Cannot see:** a frozen stretch of a very noisy image can show small blips at the encoder's
keyframes (found while testing with pure-noise video; natural images scored 0.00-0.02).

## Blur (`blur.py`)

**Question:** are the frames sharp enough to use?

**How:** variance of the Laplacian of the gray frame, after resizing to 640 px wide so the score
does not depend on the camera's resolution. Low = soft.

**Threshold:** a sample is blurry below `score_threshold: 100`; the clip fails above
`max_bad_share: 0.10`. Real samples scored 150-5,221 (5th percentile 529, median 994). Blurring
real frames with 11-21 px kernels gave 20-190.

**Cannot see:** the difference between blurry and dark or blank; a black frame also scores low.
The exposure check names the cause.

## Exposure (`brightness.py`)

**Question:** is the frame too dark or blown out?

**How:** mean gray value, 0 (black) to 255 (white).

**Threshold:** bad below `minimum_brightness: 40` or above `maximum_brightness: 215`; the clip
fails above `max_bad_share: 0.10`. Real samples ranged 65-151 (median 96), well inside.

## Hands in view (`hands.py`)

**Question:** are the hands in the shot? Egocentric robot data is about manipulation; a clip
without hands teaches a robot nothing.

**How:** MediaPipe's hand landmarker (`models/hand_landmarker.task`, up to 2 hands, image mode)
on each sample resized to 640 px wide. A sample with no hand found is bad.

**Threshold:** `max_bad_share: 0.5`, so the clip fails if more than half its samples show no
hand. On the sample set, samples without a detected hand ranged from 1% to 31% per clip
(15.5% overall), so all pass.

**Cannot see:** the difference between "no hand in view" and "MediaPipe missed the hand". The
datasets ship their own hand tracking, which reports pose for ~100% of every episode while
MediaPipe finds a hand in 69-99% of samples; which part of the gap is misses and which is hands
outside this camera's view is the next thing to measure (see RESULTS.md). Blurred and dark
footage also defeats MediaPipe, so those clips fail hands as well.
