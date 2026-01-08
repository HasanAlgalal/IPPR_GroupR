import cv2
import numpy as np


def enhance_plate(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 9, 75, 75)
    return blur


def binarize(gray):
    th = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21,
        10
    )
    return th


def clean(th):
    return th


def extract_characters(binary):
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    chars = []

    H, W = binary.shape

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        if h < H * 0.18:
            continue

        if w < W * 0.03:
            continue

        aspect = w / h
        if aspect < 0.15 or aspect > 1.8:
            continue

        chars.append((x, y, w, h))

    if len(chars) == 0:
        return [], []

    chars_sorted = sorted(chars, key=lambda c: (c[1], c[0]))

    crops = [binary[y:y+h, x:x+w] for (x, y, w, h) in chars_sorted]

    return crops, chars_sorted


def segment_pipeline(roi):
    enhanced = enhance_plate(roi)
    binary = binarize(enhanced)
    cleaned = clean(binary)
    chars, boxes = extract_characters(cleaned)
    return enhanced, binary, cleaned, chars, boxes
