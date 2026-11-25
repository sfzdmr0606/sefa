"""Detect faces (478-point mesh) and person/hair masks in the photo.

Usage: python3 detect.py aile.png build/models build/detect.npz
Writes face landmarks (pixel coords) and per-class masks for animate.
"""
import sys
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
from PIL import Image

SRC, MODELS, OUT = sys.argv[1:4]
img = np.array(Image.open(SRC).convert("RGB"))
H, W = img.shape[:2]
mpimg = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)

fl = vision.FaceLandmarker.create_from_options(
    vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=f"{MODELS}/face_landmarker.task"),
        num_faces=8,
        min_face_detection_confidence=0.3,
        min_face_presence_confidence=0.3,
    )
)
faces = []
for lm in fl.detect(mpimg).face_landmarks:
    pts = np.array([[p.x * W, p.y * H] for p in lm])
    faces.append(pts)
faces.sort(key=lambda p: p[:, 0].mean())
for f in faces:
    x0, y0 = f.min(0)
    x1, y1 = f.max(0)
    print(f"face x {x0:.0f}-{x1:.0f}  y {y0:.0f}-{y1:.0f}  size {x1 - x0:.0f}")

seg = vision.ImageSegmenter.create_from_options(
    vision.ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path=f"{MODELS}/selfie_multiclass_256x256.tflite"),
        output_confidence_masks=True,
        output_category_mask=True,
    )
)
r = seg.segment(mpimg)
conf = np.stack([m.numpy_view() for m in r.confidence_masks], 0)  # 0 bg,1 hair,2 body,3 face,4 clothes,5 other
cat = r.category_mask.numpy_view()
print("classes", conf.shape, [float((cat == i).mean()) for i in range(6)])
np.savez_compressed(OUT, faces=np.array(faces, dtype=object), conf=conf.astype(np.float16), cat=cat)
