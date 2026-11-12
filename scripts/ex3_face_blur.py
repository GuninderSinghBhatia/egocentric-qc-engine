# exercise 3: detect a face and blur it
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions

IN = "outputs/ex3/face_input.jpg"
OUT = "outputs/ex3/face_blurred.png"
PAD = 0.2  # padding around the box, as a fraction of its size

detector = vision.FaceDetector.create_from_options(
    vision.FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path="models/blaze_face_short_range.tflite")))

img = cv2.imread(IN)
if img is None:
    raise SystemExit(f"Could not read {IN}")
h, w = img.shape[:2]

rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
print(f"faces found: {len(result.detections)}")

for det in result.detections:
    b = det.bounding_box
    px, py = int(b.width * PAD), int(b.height * PAD)
    x0, y0 = max(b.origin_x - px, 0), max(b.origin_y - py, 0)
    x1, y1 = min(b.origin_x + b.width + px, w), min(b.origin_y + b.height + py, h)
    img[y0:y1, x0:x1] = cv2.GaussianBlur(img[y0:y1, x0:x1], (0, 0), 25)

cv2.imwrite(OUT, img)
print("saved", OUT)
