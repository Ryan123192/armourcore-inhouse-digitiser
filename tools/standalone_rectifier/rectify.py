"""Headless rectifier driver.

Designed to be called from ``Rectify.bat`` after a drag-drop OR from the
command line::

    python rectify.py path\\to\\scan.png
    python rectify.py path\\to\\scan.pdf --paper-w 900 --paper-h 500
    python rectify.py *.png --out C:\\some\\where

For each input we write ONE file: ``<stem>_rectified.png`` into the
output folder (default: a ``rectified\\`` sibling of the input).
The corner-debug overlay can be optionally emitted with ``--debug``.

Defaults
========
Paper size: Large GLOBAL CDS (600 x 500 mm).
Use ``--xlarge`` for 900 x 500 or ``--paper-w / --paper-h`` for custom.

Exit codes
==========
0   every input rectified OK
1   at least one input failed (per-file traceback printed)
2   argument / usage error
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import cv2

from _rectify_lib import rectify_scan


def _load_image(path: Path):
    """Return a BGR numpy array for either an image OR the first page of
    a PDF.  Lazy-import fitz so PDF support is optional."""
    if path.suffix.lower() == ".pdf":
        try:
            import fitz
        except ImportError as e:
            raise RuntimeError(
                "PDF input needs PyMuPDF.  Run:  pip install pymupdf"
            ) from e
        doc = fitz.open(str(path))
        pix = doc[0].get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))
        tmp = path.with_suffix(".__rect_tmp.png")
        try:
            pix.save(str(tmp))
            img = cv2.imread(str(tmp))
        finally:
            try:
                tmp.unlink()
            except OSError:
                pass
        return img
    return cv2.imread(str(path))


def rectify_one(in_path: Path, out_dir: Path,
                paper_w: float, paper_h: float,
                px_per_mm: float, debug: bool) -> Path:
    """Rectify a single file.  Returns the path of the written PNG."""
    img = _load_image(in_path)
    if img is None:
        raise RuntimeError(f"Could not read {in_path}")
    rr = rectify_scan(img, paper_w_mm=paper_w, paper_h_mm=paper_h,
                     px_per_mm=px_per_mm)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{in_path.stem}_rectified.png"
    cv2.imwrite(str(out_path), rr.warped)
    if debug:
        dbg_path = out_dir / f"{in_path.stem}_corners.png"
        cv2.imwrite(str(dbg_path), rr.debug_overlay)
    return out_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Headless CDS rectifier (drag-drop or CLI).")
    ap.add_argument("inputs", nargs="+", type=Path,
                   help="One or more image / PDF files (or globs).")
    ap.add_argument("--out", type=Path, default=None,
                   help="Output folder.  Default: a 'rectified' folder "
                        "next to the first input file.")
    ap.add_argument("--paper-w", type=float, default=600.0,
                   help="Calibrated DESIGN-AREA width in mm "
                        "(default: 600 = Large CDS)")
    ap.add_argument("--paper-h", type=float, default=500.0,
                   help="Calibrated design-area height in mm "
                        "(default: 500)")
    ap.add_argument("--xlarge", action="store_true",
                   help="Shortcut for --paper-w 900 --paper-h 500")
    ap.add_argument("--px-per-mm", type=float, default=6.0,
                   help="Output resolution (default 6 px/mm = "
                        "~3600px on the long side of a 600mm sheet)")
    ap.add_argument("--debug", action="store_true",
                   help="Also write <stem>_corners.png with the detected "
                        "markers drawn on the source.")
    args = ap.parse_args(argv)

    if args.xlarge:
        args.paper_w, args.paper_h = 900.0, 500.0

    # Expand any glob patterns the shell did not (bare cmd.exe doesn't).
    files: list[Path] = []
    for raw in args.inputs:
        if any(ch in str(raw) for ch in ("*", "?")):
            files.extend(Path(".").glob(str(raw)))
        elif raw.exists():
            files.append(raw)
        else:
            print(f"  ! skipped (not found): {raw}", file=sys.stderr)
    if not files:
        print("No input files.", file=sys.stderr)
        return 2

    out_dir = args.out or files[0].parent / "rectified"

    failures = 0
    for i, f in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {f.name} ... ", end="", flush=True)
        try:
            out = rectify_one(f, out_dir,
                             paper_w=args.paper_w, paper_h=args.paper_h,
                             px_per_mm=args.px_per_mm, debug=args.debug)
            print(f"OK -> {out.name}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAILED: {exc}")
            traceback.print_exc()
    print()
    print(f"Done.  {len(files) - failures}/{len(files)} succeeded.  "
          f"Output: {out_dir}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
