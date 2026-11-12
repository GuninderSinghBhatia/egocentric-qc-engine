# exercise 1: read a video with OpenCV
from pathlib import Path

import cv2

path = Path('data/humanola_sample/videos/chunk-000/observation.images.main_stereo_left/episode_000003.mp4')

cap = cv2.VideoCapture(str(path))
if not cap.isOpened():
    raise SystemExit("Could not open video")

fps = cap.get(cv2.CAP_PROP_FPS)
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
duration = frames / fps
print(fps, frames, w, h, duration)

ok, frame = cap.read()
if not ok:
    raise SystemExit("Could not read first frame")
cv2.imwrite("outputs/ex1_frame0.png", frame)
cap.release()
