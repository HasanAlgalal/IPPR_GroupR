# D:\IPPR_GroupR\code\test_pipeline.py

import os
import glob
import cv2
import shutil
import pytesseract

from pipeline.preprocess import resize_long_edge, preprocess_plate_for_ocr
from pipeline.plate_detect import detect_plate
from pipeline.ocr_state import ocr_plate

ROOT = r"D:\IPPR_GroupR"
IMG_DIRS = [
    os.path.join(ROOT, "images", "motorcycle"),
    os.path.join(ROOT, "images", "van"),
]

EXTS = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp", "*.JPG", "*.JPEG", "*.PNG", "*.WEBP", "*.BMP")
DEFAULT_TESSERACT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def set_tesseract():
    t = shutil.which("tesseract") or DEFAULT_TESSERACT
    pytesseract.pytesseract.tesseract_cmd = t


def iter_images(folder):
    out = []
    for e in EXTS:
        out.extend(glob.glob(os.path.join(folder, e)))
    return sorted(out)


def main():
    set_tesseract()

    total = 0
    det_ok = 0
    ocr_ok = 0

    for d in IMG_DIRS:
        if not os.path.isdir(d):
            continue

        print(f"\n=== Folder: {d} ===")
        for p in iter_images(d):
            total += 1
            name = os.path.basename(p)

            bgr = cv2.imread(p)
            if bgr is None:
                print(f"[READ_FAIL] {name}")
                continue

            bgr = resize_long_edge(bgr, 1400)

            warped, bbox, dbg = detect_plate(bgr)
            if warped is None:
                print(f"[DETECT_FAIL] {name}")
                continue

            det_ok += 1

            ph, pw = warped.shape[:2]
            two_row = (ph / float(pw + 1e-6)) > 0.55

            plate_for_ocr = preprocess_plate_for_ocr(warped)
            info = ocr_plate(plate_for_ocr, tesseract_path=None, two_row=two_row)

            good = (info.prefix in ("W", "V", "P")) and any(ch.isdigit() for ch in (info.plate_text or ""))
            if good:
                ocr_ok += 1

            print(
                f"{name} | plate={info.plate_text} | prefix={info.prefix} | state={info.state} | "
                f"vehicle={info.vehicle_type} | quality={getattr(info,'quality',0):.2f}"
            )

    print(f"\nSUMMARY: total={total}  detected={det_ok}  ok_ocr={ocr_ok}")


if __name__ == "__main__":
    main()
