# Frame sampler: yields about one frame per second of video, shared by all QC checks
import cv2
from egoqc.clip import Clip


def sample_frames(clip: Clip, samples_per_second: int = 1):
    """Yield (time_in_seconds, frame) for about `samples_per_second` frames per second."""
    for time, frame, _ in _walk(clip, samples_per_second, with_next=False):
        yield time, frame


def sample_frame_pairs(clip: Clip, samples_per_second: int = 1):
    """Yield (time_in_seconds, frame, next_frame): each sample plus the frame right after it.

    Used by the frozen check: a frame identical to its neighbour means the video stood still.
    """
    yield from _walk(clip, samples_per_second, with_next=True)


def _walk(clip: Clip, samples_per_second: int, with_next: bool):
    if clip.error:  # nothing to sample from a clip that did not open
        return

    gap = max(1, round(clip.fps / samples_per_second))  # frames between samples
    cap = cv2.VideoCapture(str(clip.path))
    counter = 0  # counts EVERY frame grabbed, so time = counter / fps stays correct
    try:
        # grab() moves to the next frame cheaply; retrieve() decodes pixels only for samples
        while cap.grab():
            if counter % gap == 0:
                ok, frame = cap.retrieve()
                time = counter / clip.fps
                if with_next:
                    if not cap.grab():
                        break  # the sample was the last frame, so it has no neighbour
                    counter += 1
                    ok_next, next_frame = cap.retrieve()
                    if ok and ok_next:
                        yield time, frame, next_frame
                elif ok:
                    yield time, frame, None
            counter += 1
    finally:
        cap.release()  # runs even if the caller stops early
