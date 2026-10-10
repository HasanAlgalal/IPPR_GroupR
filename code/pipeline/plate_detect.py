from __future__ import annotations

import os
import re
from dataclasses import dataclass

import cv2
import numpy as np

from pipeline.ocr_state import normalize_plate, get_prefix, get_state, get_vehicle_type


# Accepts: WSP5591, PFQ5217, JHQ2945, JHW6236, JTJ1220, DV1, DDJ303, etc.
PLATE_RE = re.compile(
    r"^(?:"
    r"(?:DV\d{1,4})|"
    r"(?:DDJ\d{1,4})|"
    r"(?:[A-Z]{1,3}\d{1,4}[A-Z]?)"
    r")$"
)

def vehicle_type_from_path(image_path: str) -> str:
    return get_vehicle_type(image_path)

def _imread_unicode(path: str):
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except (OSError, ValueError, cv2.error):
        return None

def _safe_resize(img, max_w=1600):
    h, w = img.shape[:2]
    if w <= max_w:
        return img, 1.0
    s = max_w / float(w)
    out = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_CUBIC)
    return out, s

def _prep_gray(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    return gray

def _find_plate_candidates(bgr):
    gray = _prep_gray(bgr)
    edges = cv2.Canny(gray, 60, 180)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = bgr.shape[:2]
    img_area = float(h * w)
    cands = []

    for c in cnts:
        x, y, ww, hh = cv2.boundingRect(c)
        if ww <= 0 or hh <= 0:
            continue

        ar = ww / float(hh)
        area = (ww * hh) / img_area

        # typical plate-ish constraints
        if ar < 1.6 or ar > 8.5:
            continue
        if area < 0.002 or area > 0.25:
            continue

        # score by contour area within bbox (edge density)
        box_area = ww * hh
        contour_area = cv2.contourArea(c)
        density = contour_area / float(box_area + 1e-6)

        score = density * 1.0 + min(area * 5.0, 1.0)
        cands.append((score, (x, y, ww, hh)))

    cands.sort(key=lambda t: t[0], reverse=True)
    return cands[:8]

def _crop_expand(bgr, bbox, pad=0.15):
    x, y, w, h = bbox
    H, W = bgr.shape[:2]
    px = int(w * pad)
    py = int(h * pad)
    x0 = max(0, x - px)
    y0 = max(0, y - py)
    x1 = min(W, x + w + px)
    y1 = min(H, y + h + py)
    return bgr[y0:y1, x0:x1].copy()

def _easyocr_reader():
    # cached singleton
    if not hasattr(_easyocr_reader, "_r"):
        import easyocr

        _easyocr_reader._r = easyocr.Reader(["en"], gpu=False)
    return _easyocr_reader._r

def _easyocr_read(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    reader = _easyocr_reader()
    # detail=1 -> returns bbox,text,conf
    return reader.readtext(rgb, detail=1, paragraph=False)

def _tokens_to_plate(tokens):
    """
    tokens: [(bbox, text, conf), ...]
    Try:
      - one-line join
      - two-line join (top letters + bottom digits)
      - best single token
    """
    cleaned = []
    for bbox, text, conf in tokens:
        t = normalize_plate(text)
        if not t:
            continue
        # discard obvious non-plate long words
        if len(t) >= 8 and not any(ch.isdigit() for ch in t):
            continue
        cleaned.append((bbox, t, float(conf)))

    if not cleaned:
        return "", 0.0, "no_tokens"

    # Sort by y center
    def y_center(b):
        pts = np.array(b, dtype=np.float32)
        return float(np.mean(pts[:, 1]))

    cleaned.sort(key=lambda x: y_center(x[0]))

    # Build 1-line candidate by x order
    def x_center(b):
        pts = np.array(b, dtype=np.float32)
        return float(np.mean(pts[:, 0]))

    one_line = "".join([t for _, t, _ in sorted(cleaned, key=lambda x: x_center(x[0]))])
    one_line = normalize_plate(one_line)

    # Two-line split by y gap
    ys = [y_center(b) for b, _, _ in cleaned]
    y_min, y_max = min(ys), max(ys)
    y_span = y_max - y_min

    two_line = ""
    if y_span > 18:  # heuristic: enough vertical separation
        y_mid = (y_min + y_max) / 2.0
        top = [x for x in cleaned if y_center(x[0]) <= y_mid]
        bot = [x for x in cleaned if y_center(x[0]) > y_mid]
        top_s = "".join([t for _, t, _ in sorted(top, key=lambda x: x_center(x[0]))])
        bot_s = "".join([t for _, t, _ in sorted(bot, key=lambda x: x_center(x[0]))])
        two_line = normalize_plate(top_s + bot_s)

    # Best single token
    best_single = max(cleaned, key=lambda x: x[2])
    best_single_text = best_single[1]

    cands = []
    if PLATE_RE.match(one_line):
        cands.append(("one_line", one_line, float(np.mean([c for _, _, c in cleaned]))))
    if two_line and PLATE_RE.match(two_line):
        cands.append(("two_line", two_line, float(np.mean([c for _, _, c in cleaned]))))
    if PLATE_RE.match(best_single_text):
        cands.append(("best_token", best_single_text, best_single[2]))

    if not cands:
        return "", 0.0, "no_plate_match"

    # choose highest confidence then longest
    cands.sort(key=lambda x: (x[2], len(x[1])), reverse=True)
    tag, plate, conf = cands[0]
    return plate, conf, tag

def _quality_score(conf: float, plate: str) -> float:
    # simple scoring similar to what you were seeing (100+)
    if not plate:
        return -1.0
    return round(conf * 100.0 + len(plate) * 5.0, 2)

@dataclass
class PlateInfo:
    image_name: str
    plate_text: str
    prefix: str
    state: str
    vehicle_type: str
    mode: str
    quality: float
    debug: str

def detect_plate_info(image_path: str) -> PlateInfo:
    bgr = _imread_unicode(image_path)
    name = os.path.basename(image_path)

    if bgr is None:
        return PlateInfo(
            image_name=name,
            plate_text="",
            prefix="Unknown",
            state="Unknown",
            vehicle_type=vehicle_type_from_path(image_path),
            mode="ERROR",
            quality=-1.0,
            debug="imread_failed",
        )

    vtype = vehicle_type_from_path(image_path)

    # 1) try contour candidates -> crop -> easyocr
    candidates = _find_plate_candidates(bgr)
    best_plate = ""
    best_conf = 0.0
    best_tag = ""
    best_dbg = ""

    for i, (score, bbox) in enumerate(candidates, start=1):
        crop = _crop_expand(bgr, bbox, pad=0.20)
        crop, _ = _safe_resize(crop, max_w=900)
        tokens = _easyocr_read(crop)
        plate, conf, tag = _tokens_to_plate(tokens)
        if plate and conf >= best_conf:
            best_plate, best_conf, best_tag = plate, conf, tag
            best_dbg = f"candidate={i}/{len(candidates)} det_score={score:.2f} tag={tag}"

    # 2) fallback: full image easyocr, then crop-refine around best match area (if any)
    if not best_plate:
        full, _ = _safe_resize(bgr, max_w=1600)
        tokens = _easyocr_read(full)
        plate, conf, tag = _tokens_to_plate(tokens)
        if plate:
            best_plate, best_conf, best_tag = plate, conf, tag
            best_dbg = f"fallback_full_easyocr tag={tag}"

    plate_norm = normalize_plate(best_plate)
    prefix = get_prefix(plate_norm)
    state = get_state(prefix)

    q = _quality_score(best_conf, plate_norm)

    if not plate_norm:
        return PlateInfo(
            image_name=name,
            plate_text="",
            prefix="Unknown",
            state="Unknown",
            vehicle_type=vtype,
            mode="EASYOCR",
            quality=-1.0,
            debug="no_candidates_and_no_plate_match",
        )

    return PlateInfo(
        image_name=name,
        plate_text=plate_norm,
        prefix=prefix,
        state=state,
        vehicle_type=vtype,
        mode="EASYOCR",
        quality=q,
        debug=f"{best_dbg} conf={best_conf:.2f}",
    )
