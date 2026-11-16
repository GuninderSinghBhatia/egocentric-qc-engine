# Silences FFmpeg's decoder messages while a clip is scanned (damaged clips print hundreds
# of "Invalid NAL unit" lines). OPENCV_FFMPEG_LOGLEVEL=-8 works only if set BEFORE Python
# starts (on Windows, setting it from Python does not reach OpenCV), so instead stderr is
# pointed at nothing at the OS level for the duration. The corrupt check finds damage itself.
import os
import sys
from contextlib import contextmanager


@contextmanager
def quiet_stderr():
    sys.stderr.flush()
    saved = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(devnull)
        os.close(saved)
