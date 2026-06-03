"""One-shot CLI for the In-House CDS tracing workflow (scan inputs).

Each run goes into a TIMESTAMPED folder under
``data/outputs/InhouseProduction/<YYYYMMDD-HHMMSS>_<stem>/`` so a
back-and-forth iteration produces an ordered history, not a soup.

Output per run:
    diagnostic.png      single 6-panel page
    vectors.svg         production SVG (REAL paths only)
    vectors_all.svg     includes anything the classifier flagged
    report.json         machine-readable verdict + counts
    rectified.png       phase 1 output
    cleaned.png         phase 2 cleaned image
    overlay.png         vectors on white
    vector_boxes.png    bounding boxes around detected vectors
    classified.png      path-coloured by classifier category
    _corner_TL/TR/BR/BL.png   per-corner debug crops

Usage:
    python tools/vectorise_cli.py path/to/scan.png
    python tools/vectorise_cli.py path/to/scan.png --paper-w 900 --paper-h 500
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

import cv2
import numpy as np

from armourcore_cds.phase1.marker_rectify_scan import (
    rectify_scan, PAPER_W_MM, PAPER_H_MM, DEFAULT_PX_PER_MM,
)
from armourcore_cds.phase2.clean_scan import (
    clean_scan_pencil, strip_outer_border,
)
from armourcore_cds.phase3.vectorise import write_svg, render_vector_overlay
from armourcore_cds.phase3.vectorise_pencil import extract_vector_paths_pencil
from armourcore_cds.phase3.path_classifier import (
    classify_paths, render_classified, category_counts,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _remove_small_components(mask: np.ndarray, min_area_px: int) -> np.ndarray:
    """Drop CC blobs smaller than min_area_px via an O(HW) LUT remap."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8)
    areas = stats[:, cv2.CC_STAT_AREA]
    lut = np.where(areas >= min_area_px, np.uint8(255), np.uint8(0))
    lut[0] = 0   # background always removed
    return lut[labels].astype(np.uint8)


def cleaned_to_trace_mask(cleaned_bgr, dark_thr=170, min_component=15,
                         drop_giant_frac=0.5):
    g = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2GRAY)
    m = (g < dark_thr).astype(np.uint8) * 255
    m = _remove_small_components(m, min_area_px=min_component)
    if drop_giant_frac > 0:
        H, W = m.shape
        image_area = H * W
        n, lbl, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        kill = np.zeros_like(m)
        for cid in range(1, n):
            bw = stats[cid, cv2.CC_STAT_WIDTH]
            bh = stats[cid, cv2.CC_STAT_HEIGHT]
            if bw * bh > image_area * drop_giant_frac:
                kill[lbl == cid] = 255
        m[kill > 0] = 0
    return m


def render_trace_mask_for_diag(trace_mask: np.ndarray) -> np.ndarray:
    """Trace mask visualised in a way that shows the gap-fill state."""
    H, W = trace_mask.shape
    out = np.full((H, W, 3), 255, dtype=np.uint8)
    out[trace_mask > 0] = (40, 40, 40)
    return out


def _fit(img, max_w, max_h):
    h, w = img.shape[:2]
    s = min(max_w / w, max_h / h, 1.0)
    return cv2.resize(img, (int(round(w * s)), int(round(h * s))),
                      interpolation=cv2.INTER_AREA)


def _stage_panel(img, title, tile_w, tile_h, subtitle=""):
    canvas = np.full((tile_h, tile_w, 3), 250, dtype=np.uint8)
    head = 64 if subtitle else 44
    cv2.rectangle(canvas, (0, 0), (tile_w, head), (35, 35, 35), -1)
    cv2.putText(canvas, title, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.78,
                (255, 255, 255), 2, cv2.LINE_AA)
    if subtitle:
        cv2.putText(canvas, subtitle, (14, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (180, 210, 255), 1, cv2.LINE_AA)
    body_h = tile_h - head - 8
    fit = _fit(img, tile_w - 10, body_h)
    fh, fw = fit.shape[:2]
    canvas[head + (body_h - fh) // 2:head + (body_h - fh) // 2 + fh,
           (tile_w - fw) // 2:(tile_w - fw) // 2 + fw] = fit
    return canvas


def build_diagnostic_page(stages: list[tuple[np.ndarray, str, str]],
                         header_text: str,
                         tile_w: int = 950,
                         tile_h: int = 720) -> np.ndarray:
    """Build a multi-panel diagnostic image.

    `stages` is a list of (image, title, subtitle).  Arranged 3-per-row.
    Header text is the big banner across the top.
    """
    panels = [_stage_panel(im, t, tile_w, tile_h, subtitle=s)
              for (im, t, s) in stages]
    cols = 3
    rows = (len(panels) + cols - 1) // cols
    row_imgs = []
    for r in range(rows):
        row = panels[r * cols:(r + 1) * cols]
        while len(row) < cols:
            row.append(np.full((tile_h, tile_w, 3), 250, dtype=np.uint8))
        row_imgs.append(np.hstack(row))
    body = np.vstack(row_imgs)
    header = np.full((96, body.shape[1], 3), 18, dtype=np.uint8)
    cv2.putText(header, header_text, (24, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                (255, 255, 255), 3, cv2.LINE_AA)
    return np.vstack([header, body])


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(image_path: Path, out_root: Path,
                forced_route: str | None = None,
                rectifier: str = "scan",
                ts_prefix: str | None = None,
                paper_w_mm: float | None = None,
                paper_h_mm: float | None = None,
                progress_callback=None):
    """``progress_callback(stage_name: str, file_path: Path)`` is called
    after each stage writes its PNG.  The GUI uses this to populate
    preview tiles incrementally instead of all at once at the end.
    Stages fired: ``corners``, ``corner_TL``, ``corner_TR``,
    ``corner_BL``, ``corner_BR``, ``rectified``, ``cleaned``,
    ``vector_boxes``, ``overlay``.  Failures inside the callback are
    swallowed - it must never break the pipeline."""
    # Paper size is the calibrated DESIGN area (inside the 4 markers).
    # We mutate the module globals so every downstream reference (rectify,
    # extraction mm-filters, SVG canvas) sees the selected size.  This is
    # the supported way for the GUI to pass the chosen GLOBAL CDS size -
    # the old approach (patching the phase1 module's attribute) silently
    # did nothing because this module imported the value by reference at
    # import time.
    global PAPER_W_MM, PAPER_H_MM
    if paper_w_mm is not None:
        PAPER_W_MM = paper_w_mm
    if paper_h_mm is not None:
        PAPER_H_MM = paper_h_mm

    def _notify(stage: str, path: Path):
        if progress_callback is None:
            return
        try:
            progress_callback(stage, path)
        except Exception:
            pass

    timings = {}
    overall_t0 = time.time()

    # Timestamped output folder
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    folder_name = f"{ts}_{image_path.stem}"
    if ts_prefix:
        folder_name = f"{ts_prefix}_{folder_name}"
    out_dir = out_root / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read {image_path}")

    # ---- Adaptive output resolution ----
    # The downstream algorithms were tuned at ~A3 / DEFAULT_PX_PER_MM=10
    # (~3500px long side).  On the big GLOBAL sheets (600-900mm) a fixed
    # px_per_mm=10 would produce a 6000-9000px canvas: slow in Phase 3
    # and big enough that whole traced shapes exceed the 35% bbox-drop.
    # So we scale px_per_mm DOWN to keep the long side near the tuned
    # target, but never UP (small sheets keep full fidelity).
    target_long_px = 3500
    long_mm = max(PAPER_W_MM, PAPER_H_MM)
    px_per_mm = min(DEFAULT_PX_PER_MM, target_long_px / long_mm)

    # ---- Phase 1: rectify ----
    t0 = time.time()
    rr = rectify_scan(img, paper_w_mm=PAPER_W_MM,
                     paper_h_mm=PAPER_H_MM, px_per_mm=px_per_mm)
    rect = rr.warped
    marker_debug = rr.debug_overlay
    timings["rectify"] = time.time() - t0
    cv2.imwrite(str(out_dir / "rectified.png"), rect)
    cv2.imwrite(str(out_dir / "_01_corners.png"), marker_debug)
    _notify("corners", out_dir / "_01_corners.png")

    # ------------------------------------------------------------------
    # Per-corner DIAGNOSTIC CROPS for the GUI preview
    # ------------------------------------------------------------------
    # Three overlays drawn on the cropped source so the operator can see
    # exactly how the corner was found and spot sub-mm drift:
    #
    #   * CYAN dashed rectangle ........ the ROI we searched in
    #   * YELLOW rectangle ............. axis-aligned bbox of the red blob
    #                                    the candidate filter chose
    #   * GREEN polygon ................ cv2.minAreaRect of the blob (rotated
    #                                    rect that snaps to the marker square
    #                                    body, ignoring tick-mark inflation)
    #   * RED filled dot + crosshair ... the chosen inner corner = the
    #                                    minAreaRect corner closest to page
    #                                    centre (= the rectify source point)
    try:
        pts = rr.marker_pixels_bgr
        cdebug = rr.corner_debug or {}
        src_clean = img  # un-annotated raw image
        Himg, Wimg = src_clean.shape[:2]
        roi = int(round(min(Himg, Wimg) * 0.07))
        for tag, (px, py) in zip(["TL", "TR", "BR", "BL"], pts):
            x0 = max(0, int(px) - roi); x1 = min(Wimg, int(px) + roi)
            y0 = max(0, int(py) - roi); y1 = min(Himg, int(py) + roi)
            crop = src_clean[y0:y1, x0:x1].copy()
            if not crop.size:
                continue

            # ROI border (cyan)
            cv2.rectangle(crop, (0, 0), (crop.shape[1] - 1,
                                         crop.shape[0] - 1),
                         (220, 220, 0), 3)

            info = cdebug.get(tag, {})
            if info:
                # axis-aligned bbox in YELLOW
                bx, by, bw, bh = info["bbox_xywh"]
                cv2.rectangle(crop,
                             (bx - x0, by - y0),
                             (bx + bw - x0, by + bh - y0),
                             (0, 220, 220), 2)
                # minAreaRect polygon in GREEN
                box = info.get("minrect_box")
                if box is not None:
                    poly = np.array(
                        [(int(round(p[0] - x0)),
                          int(round(p[1] - y0))) for p in box],
                        dtype=np.int32)
                    cv2.polylines(crop, [poly.reshape(-1, 1, 2)],
                                 True, (40, 200, 40), 2)
                # chosen inner corner in RED + crosshair
                ix, iy = info["inner_xy"]
                cx, cy = int(round(ix - x0)), int(round(iy - y0))
                cv2.line(crop, (cx - 14, cy), (cx + 14, cy),
                        (0, 0, 220), 2)
                cv2.line(crop, (cx, cy - 14), (cx, cy + 14),
                        (0, 0, 220), 2)
                cv2.circle(crop, (cx, cy), 5, (0, 0, 220), -1)

            # Tag label, top-left of crop
            cv2.putText(crop, tag, (8, 26),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                       (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(crop, tag, (8, 26),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                       (0, 0, 0), 1, cv2.LINE_AA)

            cv2.imwrite(str(out_dir / f"_corner_{tag}.png"), crop)
            _notify(f"corner_{tag}", out_dir / f"_corner_{tag}.png")
    except Exception:
        pass
    _notify("rectified", out_dir / "rectified.png")

    # Route is fixed to "scan" / "pencil" in the in-house workflow.  The
    # cleaning + extraction stages below are tuned for light-pencil and
    # also handle bold pen, so there is no operator choice to make.
    route = "scan"

    # ---- Phase 2: cleaning ----
    # Colour-strip template colours + Sauvola-binarise.  Handles BOTH
    # faint pencil AND bold pen so the operator never needs an ink mode.
    # Then blank a 2 mm frame so any printed BLACK page-border pixels
    # that bled into the warped canvas edge don't get traced as a
    # spurious rectangle hugging the sheet edge.
    t0 = time.time()
    cleaned = clean_scan_pencil(rect)
    cleaned = strip_outer_border(cleaned, border_px=int(round(2 * px_per_mm)))
    timings["phase2"] = time.time() - t0
    cv2.imwrite(str(out_dir / "cleaned.png"), cleaned)
    _notify("cleaned", out_dir / "cleaned.png")

    # ---- Phase 2.5: trace mask (= gap-fill input) ----
    t0 = time.time()
    trace = cleaned_to_trace_mask(cleaned)
    timings["trace_mask"] = time.time() - t0
    trace_vis = render_trace_mask_for_diag(trace)

    # ---- Phase 3: vectorise ----
    # Dilate-bridge sweep + RETR_EXTERNAL silhouette.
    # A single fixed dilate is a no-win: 5 px doesn't close every gap
    # (open ends produce long skinny "ribbon" contours instead of a
    # shape) but 13+ px starts merging adjacent shapes.  So we sweep
    # several values and pick the one that yields the most REAL paths
    # after classification.  Score by classifier REAL count, not raw
    # count - a bigger dilate that produces 30 noise blobs shouldn't
    # win over a smaller one with 9 cleanly-closed shapes.
    t0 = time.time()
    sweep = [(5, 3), (8, 5), (12, 7), (16, 9)]
    best_paths: list = []
    best_real = -1
    chosen_dilate = 5
    for dil, cls in sweep:
        try:
            cand, _ = extract_vector_paths_pencil(
                trace, design_width_mm=PAPER_W_MM,
                design_height_mm=PAPER_H_MM,
                dilate_px=dil, close_after_dilate_px=cls,
            )
        except Exception:
            cand = []
        cand_cls = classify_paths(cand, trace.shape, PAPER_W_MM, PAPER_H_MM)
        real_n = sum(1 for c in cand_cls if c.category == "REAL")
        if real_n > best_real:
            best_real = real_n
            best_paths = cand
            chosen_dilate = dil
    paths = best_paths
    timings["phase3"] = time.time() - t0

    H, W = trace.shape
    classified = classify_paths(paths, (H, W), PAPER_W_MM, PAPER_H_MM)
    counts = category_counts(classified)
    real_paths = [p for p, c in zip(paths, classified)
                 if c.category == "REAL"]

    base = np.full((H, W, 3), 255, dtype=np.uint8)
    overlay = render_vector_overlay(
        base, real_paths, mask_shape=(H, W),
        colour_bgr=(40, 180, 40), thickness=3)
    cv2.imwrite(str(out_dir / "overlay.png"), overlay)
    annotated = render_classified(paths, classified, (H, W))
    cv2.imwrite(str(out_dir / "classified.png"), annotated)

    # Vector-recognition preview: COLOURED BOXES around every path we
    # decided is a vector, drawn on a faded copy of the cleaned image
    # so the operator can see *what got picked up* in context.  Each REAL
    # path gets a green box, non-REAL gets its category colour so we can
    # see at a glance what's being filtered.
    from armourcore_cds.phase3.path_classifier import CATEGORY_BGR
    boxed = cv2.addWeighted(cleaned, 0.55,
                           np.full_like(cleaned, 255), 0.45, 0)
    for vp, cp in zip(paths, classified):
        x, y, bw, bh = vp.bbox_xywh
        colour = CATEGORY_BGR.get(cp.category, (40, 180, 40))
        cv2.rectangle(boxed, (x, y), (x + bw, y + bh), colour, 4)
        cv2.putText(boxed, cp.category, (x + 6, max(0, y - 8)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2, cv2.LINE_AA)
    cv2.imwrite(str(out_dir / "vector_boxes.png"), boxed)
    _notify("vector_boxes", out_dir / "vector_boxes.png")
    _notify("overlay", out_dir / "overlay.png")

    write_svg(real_paths, out_dir / "vectors.svg",
             mask_shape=(H, W), design_width_mm=PAPER_W_MM,
             design_height_mm=PAPER_H_MM)
    write_svg(paths, out_dir / "vectors_all.svg",
             mask_shape=(H, W), design_width_mm=PAPER_W_MM,
             design_height_mm=PAPER_H_MM)

    # ---- Verdict ----
    if counts["REAL"] == 0:
        verdict = "EMPTY"
    elif counts["TEXT"] == 0 and counts["NOISE"] == 0:
        verdict = "EXCELLENT"
    elif counts["TEXT"] + counts["NOISE"] <= 2:
        verdict = "GOOD"
    else:
        verdict = "POOR"

    total_time = time.time() - overall_t0
    timings["total"] = total_time

    # ---- Diagnostic page (5 stages) ----
    header = (f"{image_path.name}   |   route: {route}   |   "
              f"VERDICT: {verdict}   |   {total_time:.1f}s")
    stages = [
        (marker_debug,
         "1. corner recognition",
         f"rectify={timings['rectify']:.2f}s"),
        (rect,
         "2. rectified", ""),
        (cleaned,
         "3. grid + artefact removal",
         f"phase2={timings['phase2']:.2f}s"),
        (trace_vis,
         "4. trace mask (gap-fill input)",
         f"trace={timings['trace_mask']:.2f}s   ink_components={int((trace>0).sum())}"),
        (annotated,
         "5. classified vectors",
         f"phase3={timings['phase3']:.2f}s   "
         f"REAL={counts['REAL']} TEXT={counts['TEXT']} "
         f"SLIV={counts['SLIVER']} NOISE={counts['NOISE']}"),
        (overlay,
         "6. FINAL EXPORT (REAL only)",
         f"{len(real_paths)} paths to SVG"),
    ]
    diag = build_diagnostic_page(stages, header)
    cv2.imwrite(str(out_dir / "diagnostic.png"), diag)

    # ---- JSON report ----
    report = {
        "input": str(image_path),
        "timestamp": ts,
        "route": route,
        # rectifier kept for API back-compat (always "scan" in in-house build)
        "rectifier": rectifier,
        "verdict": verdict,
        "counts": counts,
        "total_paths": len(paths),
        "real_paths": len(real_paths),
        "pencil_chosen_dilate": chosen_dilate,
        "timings_sec": {k: round(v, 3) for k, v in timings.items()},
        "vector_svg": str(out_dir / "vectors.svg"),
    }
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n=== {image_path.name} ===")
    print(f"Folder:          {out_dir}")
    print(f"Rectifier:       {rectifier}")
    print(f"Route:           {route}")
    print(f"Verdict:         {verdict}")
    print(f"REAL paths:      {counts['REAL']}")
    print(f"TEXT rejected:   {counts['TEXT']}")
    print(f"SLIVER rejected: {counts['SLIVER']}")
    print(f"Time:            {total_time:.1f}s "
          f"(P1 {timings['rectify']:.1f}s + P2 {timings['phase2']:.1f}s + "
          f"P3 {timings['phase3']:.1f}s)")
    print(f"Diagnostic:      {out_dir / 'diagnostic.png'}")
    print(f"SVG:             {out_dir / 'vectors.svg'}")
    return out_dir, verdict


def main():
    ap = argparse.ArgumentParser(
        description="In-House CDS vectoriser (scanned sheets)")
    ap.add_argument("image", type=Path, help="Path to scan / PDF render")
    ap.add_argument("--out", type=Path,
                   default=Path("data/outputs/InhouseProduction"),
                   help="Output root (default: data/outputs/InhouseProduction/)")
    ap.add_argument("--tag", default=None,
                   help="Optional prefix for the output folder name")
    ap.add_argument("--paper-w", type=float, default=None,
                   help="Calibrated design-area width in mm "
                        "(e.g. 600 Large, 900 X-Large)")
    ap.add_argument("--paper-h", type=float, default=None,
                   help="Calibrated design-area height in mm (e.g. 500)")
    args = ap.parse_args()
    run_pipeline(args.image, args.out,
                ts_prefix=args.tag,
                paper_w_mm=args.paper_w, paper_h_mm=args.paper_h)


if __name__ == "__main__":
    main()
