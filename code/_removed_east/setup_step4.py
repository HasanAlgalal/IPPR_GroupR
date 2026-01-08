# D:\IPPR_GroupR\code\setup_step4.py
from __future__ import annotations

import io
import os
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(r"D:\IPPR_GroupR\code")
PIPELINE = ROOT / "pipeline"
PIPELINE.mkdir(parents=True, exist_ok=True)

# Official EAST download referenced by OpenCV sample (tar.gz contains the .pb)
# Source: OpenCV samples/dnn/text_detection.py :contentReference[oaicite:0]{index=0}
EAST_TAR_URL = "https://www.dropbox.com/s/r2ingd0l3zt8hxs/frozen_east_text_detection.tar.gz?dl=1"
EAST_PB_NAME = "frozen_east_text_detection.pb"
EAST_PB_PATH = PIPELINE / EAST_PB_NAME


TEXT_DETECT_PY = r'''# pipeline/text_detect.py
from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np

# Always resolve relative to this file so your project works on any PC
EAST_PATH = Path(__file__).resolve().parent / "frozen_east_text_detection.pb"


def _decode(scores: np.ndarray, geometry: np.ndarray, score_thresh: float):
    (num_rows, num_cols) = scores.shape[2:4]
    rects = []
    confidences = []

    for y in range(num_rows):
        scores_data = scores[0, 0, y]
        x0 = geometry[0, 0, y]
        x1 = geometry[0, 1, y]
        x2 = geometry[0, 2, y]
        x3 = geometry[0, 3, y]
        angles = geometry[0, 4, y]

        for x in range(num_cols):
            score = float(scores_data[x])
            if score < score_thresh:
                continue

            offset_x = x * 4.0
            offset_y = y * 4.0
            angle = float(angles[x])
            cos = np.cos(angle)
            sin = np.sin(angle)

            h = float(x0[x] + x2[x])
            w = float(x1[x] + x3[x])

            end_x = int(offset_x + (cos * x1[x]) + (sin * x2[x]))
            end_y = int(offset_y - (sin * x1[x]) + (cos * x2[x]))
            start_x = int(end_x - w)
            start_y = int(end_y - h)

            rects.append((start_x, start_y, end_x, end_y))
            confidences.append(score)

    return rects, confidences


def detect_text_regions(roi_bgr, score_thresh: float = 0.5, nms_thresh: float = 0.4, debug: bool = False):
    """
    Input: ROI image (BGR) from your plate detector.
    Output: list of cropped BGR images (text regions), sorted left-to-right, top-to-bottom.
    """
    if roi_bgr is None:
        return []

    if not EAST_PATH.exists():
        raise FileNotFoundError(f"Missing model: {EAST_PATH}")

    net = cv2.dnn.readNet(str(EAST_PATH))

    H, W = roi_bgr.shape[:2]
    newW = (W // 32) * 32
    newH = (H // 32) * 32
    if newW < 32: newW = 32
    if newH < 32: newH = 32

    rW = W / float(newW)
    rH = H / float(newH)

    resized = cv2.resize(roi_bgr, (newW, newH))
    blob = cv2.dnn.blobFromImage(resized, 1.0, (newW, newH),
                                 (123.68, 116.78, 103.94), swapRB=True, crop=False)

    net.setInput(blob)
    (scores, geometry) = net.forward([
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"
    ])

    rects, confidences = _decode(scores, geometry, score_thresh)
    if len(rects) == 0:
        return []

    boxes = cv2.dnn.NMSBoxes(rects, confidences, score_thresh, nms_thresh)
    crops = []

    if len(boxes) > 0:
        for i in boxes.flatten():
            (sx, sy, ex, ey) = rects[i]
            sx = int(max(0, sx * rW))
            sy = int(max(0, sy * rH))
            ex = int(min(W, ex * rW))
            ey = int(min(H, ey * rH))

            if ex - sx < 5 or ey - sy < 5:
                continue

            crop = roi_bgr[sy:ey, sx:ex].copy()
            crops.append((sx, sy, crop))

    # Sort: top-to-bottom, then left-to-right (good for 2-line plates)
    crops.sort(key=lambda t: (t[1], t[0]))
    out = [c[2] for c in crops]

    if debug:
        for idx, img in enumerate(out):
            cv2.imshow(f"CROP_{idx}", img)

    return out
'''


TEST_PIPELINE_PY = r'''# test_pipeline.py
from __future__ import annotations

import cv2

from pipeline.preprocess import preprocess
from pipeline.plate_detect import plate_detect_pipeline
from pipeline.text_detect import detect_text_regions
from pipeline.ocr_state import ocr_single, map_state


def ocr_from_crops(crops):
    parts = []
    for c in crops:
        t = ocr_single(c)
        if t:
            parts.append(t)
    # Join without spaces (Malaysian plates typically no spaces in OCR output)
    return "".join(parts)


def run(image_path: str):
    original, base, smoothed = preprocess(image_path)

    roi, edges, closed = plate_detect_pipeline(smoothed, original)
    if roi is None:
        print("NO PLATE DETECTED")
        return

    # EAST crops (turn on debug=True to see CROP_0, CROP_1...)
    crops = detect_text_regions(roi, debug=True)

    plate_text = ocr_from_crops(crops)
    state = map_state(plate_text)

    print("PLATE:", plate_text if plate_text else "Not detected")
    print("STATE:", state)

    cv2.imshow("Original", original)
    cv2.imshow("Edges", edges)
    cv2.imshow("Closed", closed)
    cv2.imshow("ROI_full_debug", roi)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    path = r"D:\IPPR_GroupR\images\motorcycle\sample1.jpg"
    run(path)
'''


def write_text(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def download_and_extract_pb():
    if EAST_PB_PATH.exists():
        return

    print("Downloading EAST model tar.gz ...")
    with urllib.request.urlopen(EAST_TAR_URL) as resp:
        data = resp.read()

    print("Extracting .pb ...")
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        member = None
        for m in tar.getmembers():
            if m.name.endswith(EAST_PB_NAME):
                member = m
                break
        if member is None:
            raise RuntimeError("EAST tar.gz did not contain frozen_east_text_detection.pb")

        f = tar.extractfile(member)
        if f is None:
            raise RuntimeError("Failed to extract model file from tar.gz")

        EAST_PB_PATH.write_bytes(f.read())

    print(f"Saved: {EAST_PB_PATH}")


def main():
    write_text(PIPELINE / "text_detect.py", TEXT_DETECT_PY)
    write_text(ROOT / "test_pipeline.py", TEST_PIPELINE_PY)
    download_and_extract_pb()

    print("\nDone.")
    print("Run:  cd D:\\IPPR_GroupR\\code")
    print("      py -3.13 test_pipeline.py")


if __name__ == "__main__":
    main()
