# Frame sampler: yields about one frame per second of video, shared by all QC checks
import cv2
from egoqc.clip import Clip



def sample_frames(clip: Clip, samples_per_second: int = 1):
    """Yield (time_in_seconds, frame) for about `samples_per_second` frames per second."""
    if clip.error:  # nothing to sample from a clip that did not open
        return

    gap = max(1, round(clip.fps / samples_per_second))  # frames between samples
    cap = cv2.VideoCapture(str(clip.path))
    counter = 0  # counts EVERY frame read, so time = counter / fps stays correct
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if counter % gap == 0:
                yield counter / clip.fps, frame
            counter += 1
    finally:
        cap.release()  # runs even if the caller stops early
