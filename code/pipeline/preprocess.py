import cv2
import numpy as np


def resize_long_edge(bgr, long_edge: int = 1200):
    h, w = bgr.shape[:2]
    m = max(h, w)
    if m <= long_edge:
        return bgr
    scale = long_edge / float(m)
    return cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def clahe_gray(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def preprocess_for_plate_candidates(bgr):
    """
    Returns:
      gray_eq, edge, morph_edge, text_blob, candidate_map
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)
    gray_eq = clahe_gray(gray)

    # Edge map
    blur = cv2.GaussianBlur(gray_eq, (5, 5), 0)
    edge = cv2.Canny(blur, 70, 200)

    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5))
    morph_edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, k_close, iterations=2)
    morph_edge = cv2.morphologyEx(
        morph_edge,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )

    # Text-blob map
    k_tophat = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    tophat = cv2.morphologyEx(gray_eq, cv2.MORPH_TOPHAT, k_tophat)
    tophat = cv2.normalize(tophat, None, 0, 255, cv2.NORM_MINMAX)

    _, bw = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(
        bw,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )

    dil1 = cv2.dilate(bw, cv2.getStructuringElement(cv2.MORPH_RECT, (45, 7)), iterations=2)
    dil2 = cv2.dilate(bw, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25)), iterations=1)
    text_blob = cv2.bitwise_or(dil1, dil2)
    text_blob = cv2.morphologyEx(
        text_blob,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (17, 9)),
        iterations=2,
    )

    candidate_map = cv2.bitwise_or(morph_edge, text_blob)
    return gray_eq, edge, morph_edge, text_blob, candidate_map


def preprocess_plate_for_ocr(bgr_plate):
    """
    Light OCR prep (not aggressive). The heavy OCR selection is done in ocr_state.py.
    Returns BGR image.
    """
    h, w = bgr_plate.shape[:2]
    if h == 0 or w == 0:
        return bgr_plate

    # Remove bottom dealer strip (common on your motorcycle plates)
    cut = int(h * 0.88)
    plate = bgr_plate[:cut, :]

    # Add small border to help Tesseract
    plate = cv2.copyMakeBorder(plate, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # Normalize height
    ph, pw = plate.shape[:2]
    target_h = 220
    scale = target_h / float(ph)
    new_w = max(1, int(pw * scale))
    plate = cv2.resize(plate, (new_w, target_h), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 50, 50)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = clahe_gray(gray)

    # Otsu binarization (keeps shapes cleaner than heavy adaptive on some plates)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Keep characters bright
    if bw.mean() > 140:
        bw = 255 - bw

    # Light cleanup only
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)
    return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
