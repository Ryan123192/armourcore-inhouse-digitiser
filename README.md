# In-House CDS Digitiser

GUI + headless CLI for converting **scanned ArmourCore Calibrated Design
Sheets (CDS)** into clean vector SVGs.

This is the production in-house tool: scanned input only, scan-rectifier
only, light-pencil-sensitive cleaning, dilate-bridge vectorisation,
classifier-filtered REAL paths to SVG.

---

## Quick start (work PC, first time)

```cmd
git clone https://github.com/ArmourCore/armourcore-inhouse-digitiser.git
cd armourcore-inhouse-digitiser
install.bat
Launch_InHouse_Digitiser.bat
```

`install.bat` creates a local `.venv\`, installs the pinned
requirements, and prints a success line. After that you only ever
double-click `Launch_InHouse_Digitiser.bat`.

For updates: `git pull` then `install.bat` again (the venv is reused —
re-running is fast).

---

## Workflow

The GUI walks the operator through four steps:

| Step | What |
|---|---|
| 1 | Select Image or PDF — PDFs auto-converted at 300 dpi |
| 2 | Orient CDS — two big ↺ / ↻ buttons until the red markers sit in corners |
| 3 | Select CDS Size — Large (600 × 500), X-Large (900 × 500), or inline custom |
| 4 | Select Output Folder |
| - | VECTORISE — preview tiles populate as each stage finishes |

Preview tiles show: imported template, four corner-detection diagnostics
(ROI / bbox / minAreaRect / chosen inner point), rectified, cleaned,
vector recognition (bounding boxes), and final vectors on white. Output
folder gets a timestamped subfolder with the SVG plus all diagnostic
PNGs.

---

## Headless / batch use

```cmd
.venv\Scripts\python tools\vectorise_cli.py path\to\scan.png
.venv\Scripts\python tools\vectorise_cli.py scan.png --paper-w 900 --paper-h 500
```

---

## Drag-and-drop rectifier

`tools\standalone_rectifier\` is a fully portable folder — pick it up,
move it anywhere, drop a scan / PDF onto `Rectify.bat` and get a
perspective-corrected PNG back. Zero project imports. See its own
`README.txt`.

---

## Project layout

```
src/armourcore_cds/
  phase1/marker_rectify_scan.py   red-marker detect + perspective warp
  phase2/clean_scan.py            HSV colour-strip + 2 mm border kill
  phase2/pencil_enhance_v2.py     Sauvola adaptive binarise
  phase3/vectorise.py             SVG writer + path data classes
  phase3/vectorise_v3.py          Chaikin smoothing
  phase3/vectorise_pencil.py      dilate-bridge extractor (RETR_EXTERNAL)
  phase3/path_classifier.py       REAL / TEXT / SLIVER / NOISE filter

tools/
  in_house_digitiser.py           Tkinter GUI
  vectorise_cli.py                headless CLI
  assets/ArmourCore_Logo_Small.png
  standalone_rectifier/           portable drag-drop rectifier folder

Launch_InHouse_Digitiser.bat      double-click launcher
install.bat                       create .venv + pip install
requirements.txt                  pinned deps
```

---

## Requirements

- Windows 10/11
- Python 3.10 or newer on PATH
- About 200 MB free disk for `.venv` after install

`install.bat` will tell you if Python isn't found.

---

## Pinned versions

See `requirements.txt`. Bump deliberately — every work PC pulls the
same set so an `opencv-python` minor change can't surprise you.
