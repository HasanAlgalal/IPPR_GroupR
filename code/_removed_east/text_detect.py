# pipeline/text_detect.py
from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np

EAST_PATH = Path(__file__).resolve().parent / "frozen_east_text_detection.pb"
_NET = None


def _get_net():
    global _NET
    if _NET is None:
        if not EAST_PATH.exists():
            raise FileNotFoundError(f"Missing model: {EAST_PATH}")
        _NET = cv2.dnn.readNet(str(EAST_PATH))
    return _NET


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


def detect_text_regions(
    roi_bgr,
    score_thresh: float = 0.5,
    nms_thresh: float = 0.4,
    debug: bool = False,
):
    """
    Input: ROI image (BGR)
    Output: list of cropped BGR images (text regions), sorted top-to-bottom then left-to-right.
    """
    if roi_bgr is None:
        return []

    net = _get_net()

    H, W = roi_bgr.shape[:2]
    newW = max(32, (W // 32) * 32)
    newH = max(32, (H // 32) * 32)

    rW = W / float(newW)
    rH = H / float(newH)

    resized = cv2.resize(roi_bgr, (newW, newH))
    blob = cv2.dnn.blobFromImage(
        resized,
        1.0,
        (newW, newH),
        (123.68, 116.78, 103.94),
        swapRB=True,
        crop=False,
    )

    net.setInput(blob)
    (scores, geometry) = net.forward(
        ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
    )

    rects, confidences = _decode(scores, geometry, score_thresh)
    if not rects:
        return []

    boxes = cv2.dnn.NMSBoxes(rects, confidences, score_thresh, nms_thresh)

    crops = []
    pad_x = max(2, int(0.03 * W))
    pad_y = max(2, int(0.03 * H))

    if len(boxes) > 0:
        for i in boxes.flatten():
            (sx, sy, ex, ey) = rects[i]
            sx = int(sx * rW)
            sy = int(sy * rH)
            ex = int(ex * rW)
            ey = int(ey * rH)

            sx = max(0, sx - pad_x)
            sy = max(0, sy - pad_y)
            ex = min(W, ex + pad_x)
            ey = min(H, ey + pad_y)

            if ex - sx < 10 or ey - sy < 10:
                continue

            crop = roi_bgr[sy:ey, sx:ex].copy()
            crops.append((sx, sy, crop))

    crops.sort(key=lambda t: (t[1], t[0]))
    out = [c[2] for c in crops]

    if debug:
        for idx, img in enumerate(out):
            cv2.imshow(f"CROP_{idx}", img)

    return out
