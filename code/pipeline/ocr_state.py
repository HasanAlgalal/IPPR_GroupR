# pipeline/ocr_state.py
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Tuple, Union

# Assignment mapping you requested:
# W -> Kuala Lumpur, P -> Penang, J -> Johor, A -> Ipoh (Perak)
STATE_BY_FIRST_LETTER = {
    "W": "Kuala Lumpur",
    "P": "Penang",
    "J": "Johor",
    "A": "Ipoh",
}

def normalize_plate(raw: str) -> str:
    """Uppercase + keep only A-Z0-9."""
    if raw is None:
        return ""
    s = raw.upper()
    s = re.sub(r"[^A-Z0-9]", "", s)

    # Tiny OCR fix: some OCR reads 'W' as 'VV'
    if s.startswith("VV"):
        s = "W" + s[2:]
    return s

def extract_state_prefix(plate_text: str) -> str:
    """
    Letters before first digit, then only FIRST letter as state prefix.
      ABF7429 -> A
      WB9656L -> W
      WSP5591 -> W
    """
    s = normalize_plate(plate_text)
    m = re.match(r"^[A-Z]+", s)
    if not m:
        return ""
    letters = m.group(0)
    return letters[:1] if letters else ""

def infer_state_from_prefix(prefix_letter: str) -> str:
    if not prefix_letter:
        return "Unknown"
    return STATE_BY_FIRST_LETTER.get(prefix_letter[0].upper(), "Unknown")

def infer_vehicle_type(image_path: Union[str, Path, None]) -> str:
    """Infer vehicle type from folder name in the selected image path."""
    if not image_path:
        return "Unknown"

    p = str(image_path).replace("/", "\\").lower()

    if "\\motorcycle\\" in p:
        return "Motorcycle"
    if "\\van\\" in p:
        return "Van"
    if "\\car\\" in p:
        return "Car"
    if "\\truck\\" in p:
        return "Truck"
    if "\\bus\\" in p:
        return "Bus"

    return "Unknown"

def infer_prefix_state_vehicle(
    plate_text: str,
    image_path: Optional[Union[str, Path]] = None
) -> Tuple[str, str, str]:
    prefix = extract_state_prefix(plate_text) or "Unknown"
    state = infer_state_from_prefix(prefix if prefix != "Unknown" else "")
    vehicle = infer_vehicle_type(image_path)
    return prefix, state, vehicle

# ----------------------------
# Backward-compatibility API
# (so your existing plate_detect.py imports keep working)
# ----------------------------

def get_prefix(plate_text: str) -> str:
    """Old name used by plate_detect.py"""
    return extract_state_prefix(plate_text) or "Unknown"

def get_state(prefix_or_plate: str) -> str:
    """
    Old name used by plate_detect.py.
    Accepts either a prefix like 'W' or a plate like 'WSP5591'.
    """
    if not prefix_or_plate:
        return "Unknown"
    s = normalize_plate(prefix_or_plate)
    # if caller passed a full plate, derive prefix first
    prefix = s[0] if s else ""
    # if caller passed only prefix, same result
    return infer_state_from_prefix(prefix)

def get_vehicle_type(image_path: Union[str, Path, None]) -> str:
    """Old name (if referenced elsewhere)."""
    return infer_vehicle_type(image_path)

def get_prefix_state_vehicle(plate_text: str, image_path: Optional[Union[str, Path]] = None):
    """Old helper name (if referenced elsewhere)."""
    return infer_prefix_state_vehicle(plate_text, image_path)
