# exercise ex2
import cv2, mediapipe as mp
from pathlib import Path
from mediapipe.tasks.python import vision, BaseOptions
import numpy as np

path = Path('data/humanola_sample/videos/chunk-000/observation.images.main_stereo_left/episode_000003.mp4')

opts = vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="models/hand_landmarker.task"),
    num_hands=2)  # default mode is IMAGE, which suits individual frames
detector = vision.HandLandmarker.create_from_options(opts)

cap = cv2.VideoCapture(str(path))
if not cap.isOpened():
    raise SystemExit("Could not open video")

frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
n = 100                                        # frames you want
idxs = np.linspace(0, frames - 1, n).astype(int)  # evenly spaced

print(idxs[:5], len(idxs))
hits = total = saved = 0


for i in idxs :
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
    ok, frame = cap.read()
    if not ok:
        continue
    small = cv2.resize(frame,(640,400))
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_img)
    total += 1
    if result.hand_landmarks:
        hits += 1
        for hand in result.hand_landmarks:
            for p in hand:
                cv2.circle(small, (int(p.x * 640), int(p.y * 400)), 3, (0, 255, 0), -1)
        if saved < 5:
            cv2.imwrite(f"outputs/ex2_hands_{saved}.png", small)
            saved += 1

cap.release()
print(f"hands found in {hits}/{total} frames")