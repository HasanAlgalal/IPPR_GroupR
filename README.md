# Malaysian License Plate Recognition

**Image Processing and Pattern Recognition group project — Asia Pacific University (APU).**

A Python desktop prototype that reads license plate text from vehicle images using OpenCV and EasyOCR, normalizes the result and looks up selected Malaysian plate prefixes. A Tkinter interface shows the detected text, prefix, mapped region, recognition mode and diagnostic score.

[Portfolio](https://hasan-algalal-portfolio.vercel.app/#work) · [Examples](docs/EXAMPLES.md) · [Technical notes](docs/ARCHITECTURE.md)

![Desktop recognition results](docs/screenshots/gui-results.png)

## Features

- Image selection through a Tkinter desktop interface.
- Grayscale filtering, Canny edges and contour-based plate candidate ranking.
- EasyOCR on candidate crops, with full-image OCR as a fallback.
- One-line and two-line token joining, normalization and plate-format filtering.
- Prefix lookup for the codes currently implemented in `code/pipeline/ocr_state.py`.
- Batch processing of the included sample images with a portable command-line runner.
- Additional preprocessing and character segmentation modules retained from the group work.

**Vehicle type is a folder label**, inferred from paths such as `images/motorcycle/` or `images/car/`; it is not a trained visual classifier. Prefix mapping is limited, and the quality score is a heuristic rather than an accuracy percentage.

## My contribution

This is a **group project**, not a solo implementation. Hasan Ahmed Yahya Algalal (TP075934) contributed across the following areas. The percentages are his estimates of his share of each module's work, not performance measurements or independently verified authorship statistics.

| Module | Hasan's estimated contribution |
| --- | --- |
| GUI | 50% |
| Plate detection | 100% |
| Preprocessing | 80% |
| Character segmentation | 50% |
| OCR and state mapping | 60% |

Other group members contributed to the remaining implementation and project work. The repository contains the available shared prototype; some report diagrams describe a broader pipeline than the active GUI currently uses.

## Quick start

Use **Python 3.11** and a desktop environment with Tkinter.

```bash
git clone https://github.com/HasanAlgalal/IPPR_GroupR.git
cd IPPR_GroupR
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

Then install and launch:

```bash
python -m pip install --upgrade pip
python -m pip install -r code/requirements.txt
python code/main_gui.py
```

On Linux, Tkinter may require your distribution's `python3-tk` package. EasyOCR downloads its pretrained models on the first recognition attempt, so internet access and writable model-cache storage are needed initially. This implementation uses CPU inference (`gpu=False`). First recognition can take longer while models load.

Select a sample under `images/`. The GUI displays results in its text panel. Images elsewhere can still be processed, but vehicle type will be `Unknown` unless their parent path contains a recognized category.

### Batch processing

```bash
python code/test_pipeline.py --limit 3
python code/test_pipeline.py --images images/motorcycle
python code/test_pipeline.py --images /path/to/your/images
```

The runner discovers images recursively and reports recognized plate strings. It is a demonstration runner, not a ground-truth accuracy benchmark. Tesseract is not required for the active EasyOCR pipeline.

## Checks

```bash
python -m unittest discover -s tests -v
```

GitHub Actions runs deterministic tests for normalization, prefix lookup, token assembly, image error handling and the candidate/fallback decision paths. OCR output is mocked in these tests, so they do not measure recognition accuracy. See [technical notes](docs/ARCHITECTURE.md) for the verification scope.

## Repository layout

```text
code/main_gui.py          Tkinter application entry point
code/test_pipeline.py     Portable batch demonstration runner
code/pipeline/            Detection, OCR mapping, preprocessing and segmentation
code/_removed_east/       Archived EAST experiment; not used by the active app
images/                  Existing vehicle sample images
report/                  Original draft report and historical run log
docs/                    Examples, screenshots and implementation notes
tests/                   Deterministic pipeline tests
```

The archived EAST setup script has machine-specific paths and writes files. It is retained as historical work and is not part of the installation instructions. The old report run log reflects an earlier pipeline, not a current benchmark.

## Limitations

Small, blurred, angled or low-contrast plates can be missed or misread. Contours can select non-plate text, and the full-image fallback can also return unrelated text that resembles a plate. The prefix table is incomplete and should not be treated as an authoritative vehicle registration lookup. Sample results are illustrations, not evidence of general accuracy.

The original sample images are retained for academic demonstration; their inclusion does not establish permission for unrelated commercial reuse.

## Contributor

**Hasan Ahmed Yahya Algalal (TP075934)** — Bachelor of Computer Science, APU, Malaysia.

[GitHub](https://github.com/HasanAlgalal) · [Portfolio](https://hasan-algalal-portfolio.vercel.app/)
