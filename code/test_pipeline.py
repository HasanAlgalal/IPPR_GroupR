"""Batch demonstration runner for the active EasyOCR pipeline (not an accuracy benchmark)."""
from __future__ import annotations

import argparse
from pathlib import Path

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=Path, default=Path(__file__).resolve().parents[1] / 'images', help='Image file or directory to process recursively')
    parser.add_argument('--limit', type=int, help='Process at most this many images')
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be a positive integer')
    if not args.images.exists():
        parser.error(f'Image path does not exist: {args.images}')

    paths = [args.images] if args.images.is_file() else sorted(args.images.rglob('*'))
    paths = [path for path in paths if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    if args.limit is not None:
        paths = paths[:args.limit]
    if not paths:
        parser.error('No supported images found')

    # Delay expensive imports so --help works before installing OCR dependencies.
    from pipeline.plate_detect import detect_plate_info

    recognized = errors = 0
    for path in paths:
        try:
            info = detect_plate_info(str(path))
        except Exception as exc:
            errors += 1
            print(f'[ERROR] {path.name}: {exc}')
            continue
        recognized += bool(info.plate_text)
        errors += info.mode == 'ERROR'
        print(f'{path.name} | plate={info.plate_text or "NOT_RECOGNIZED"} | prefix={info.prefix} | state={info.state} | vehicle={info.vehicle_type} | score={info.quality} | {info.debug}')

    print(f'\nSUMMARY: images={len(paths)} recognized={recognized} errors={errors}')
    print('Recognition count is not an accuracy measurement; no ground-truth labels are compared.')
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
