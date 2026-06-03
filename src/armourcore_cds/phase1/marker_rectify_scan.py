"""Phase 1 sibling for SCANNED INPUTS.

Why a separate module?
======================
The production ``marker_rectify_fast_v4`` was designed for phone
photographs where the sheet sits on a desk with non-paper background
around it.  It does a lot of work to find the paper edge, handle
perspective distortion, deal with lighting, etc.

A scan is fundamentally different:
* Background is paper-white (the platen lid)
* No perspective distortion - it's a flatbed
* Red corner markers (#FF0033) are reliably ~near the 4 corners
* Small skew at most (paper sits a few degrees off-square on platen)

So the rectifier just needs to:
1. Find red pixels via colour threshold
2. Cluster them into 4 components - one near each image corner
3. Compute centroid of each marker cluster
4. Perspective-warp those 4 points to (PAPER_W_MM, PAPER_H_MM)

That's it.  No edge detection, no Hough, no Lab dance.

API mirrors v4 so the rest of the pipeline can swap in unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Calibrated GLOBAL CDS defaults.  These used to live in the
# phone-photo rectifier module; inlined here so this module has no
# cross-package import.  The GUI passes its own paper_w/h to override.
PAPER_W_MM = 600.0
PAPER_H_MM = 500.0
DEFAULT_PX_PER_MM = 10.0


@dataclass
class ScanRectifyResult:
    """Mirrors RectifyResult shape - has .warped so downstream code works."""
    warped: np.ndarray
    marker_pixels_bgr: tuple   # the 4 (x, y) centroids - useful for debug
    debug_overlay: np.ndarray  # original with markers + bounding box drawn
    corner_debug: dict | None = None
    """Per-corner debug info, keyed by TL/TR/BR/BL.  Each entry has:
        ``bbox_xywh``  - axis-aligned bbox of the red blob the candidate
                         filter chose (yellow rectangle in the diag image)
        ``minrect_box``- 4 corners of cv2.minAreaRect (green polygon)
        ``inner_xy``   - the chosen inner-corner point (red crosshair)
    Used by the CLI to render the per-corner diagnostic crops the
    operator sees in the GUI."""


def _red_marker_mask(image_bgr: np.ndarray) -> np.ndarray:
    """Detect red marker pixels.

    Marker colour is #FF0033 (R=255, G=0, B=51).  The CDS template ALSO
    prints warm orange #F1B01D (R=241, G=176, B=29) for the
    customer-facing size indicators, plus a few orange highlights in
    the ArmourCore logo above the TL marker.  On a scanner, the orange
    edges anti-alias toward red enough to slip past a loose hue band
    and a loose BGR-dominance test - which then drags the TL bbox
    upward into the logo and downward into the BL size indicator.

    The decisive discriminator is the GREEN CHANNEL:
        marker red  (#FF0033)  G = 0
        orange      (#F1B01D)  G = 176
    A single G<110 ceiling rejects orange with ~66 px of safety margin
    and never threatens the marker (which sits at G<40 even after
    scanner anti-aliasing).

    Combined rules:
        BGR:  R - max(G,B) > 80   AND  R > 120   AND  G < 110
        HSV:  (H<8 or H>170)      AND  S > 120   AND  V > 80
    Both must agree.  Tighter HSV than before (H<8 instead of <15,
    S>120 instead of >60) because the scanner gives us consistent
    lighting and there is no point chasing pale-pink pixels - the
    marker itself is bold and saturated.
    """
    # Why tighten ONLY the hue band, not the BGR / saturation rules?
    # Print quality varies marker-to-marker on real scans.  In one Large
    # CDS, the BL marker was visibly faded (pixels at R~224 G~223 B~227,
    # H median 169) while TL/TR/BR were bold (S=255).  A G<110 ceiling or
    # S>100 floor would have killed BL.  Orange #F1B01D sits at H=22 and
    # anti-aliased orange-on-white edges stay near H=22-28, so the only
    # way orange leaks in is through the H=5-15 transition band where
    # orange touches genuine red (boundary line, red text, etc.) and the
    # gradient pixels drift into the H<15 region.  Cutting the band to
    # H<5 surgically rejects that transition zone while leaving pure red
    # (H=0 or H>170) - which both bold and faded markers have - intact.
    B, G, R = cv2.split(image_bgr)
    red_bgr = (R.astype(np.int16) - np.maximum(G, B).astype(np.int16) > 30) & (R > 100)

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    red_hsv = ((H < 5) | (H > 170)) & (S > 60) & (V > 60)

    mask = (red_bgr & red_hsv).astype(np.uint8) * 255
    # Close holes in the X-inside-square marker design so each marker is
    # one connected SQUARE blob.  The kernel scales with image size: the
    # marker is a SQUARE OUTLINE with an X (not solid), so on the bigger
    # GLOBAL CDS sheets (Large/X-Large) the gaps between outline strokes
    # are wider in pixels and a fixed 15px close leaves the square split
    # into halves (which then fail the square-aspect test downstream).
    # ~1% of the short side reliably merges each marker into one square.
    H, W = mask.shape[:2]
    close_k = max(15, int(round(min(H, W) * 0.01)))
    if close_k % 2 == 0:
        close_k += 1
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k)))
    # Speckle clean
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    return mask


def _find_4_corner_markers(
    red_mask: np.ndarray,
    image_shape,
    debug_out: dict | None = None,
) -> list[tuple[float, float]]:
    """Return 4 marker INNER-CORNER points ordered TL, TR, BR, BL.

    For each marker (a square in a page corner), the INNER CORNER is the
    bbox corner facing the page centre.  So the design area sits exactly
    INSIDE the four markers, not overlapping them.

      TL marker -> its bbox BR corner (page-centre-facing)
      TR marker -> its bbox BL corner
      BR marker -> its bbox TL corner
      BL marker -> its bbox TR corner
    """
    H, W = image_shape[:2]
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(
        red_mask, connectivity=8)

    # The GLOBAL CDS prints a LOT of red that is NOT a corner marker:
    #   * the FF0000 calibrated-boundary line (thin, long -> high aspect)
    #   * red pencil annotations / arrows / "5mm" labels (irregular)
    #   * red text printed around the sheet outside the design area
    # The 4 real markers are SQUARES near the 4 page corners.  So instead
    # of "largest 8 blobs" (which the boundary line + scribbles dominate),
    # we keep only SQUARE-ish blobs of a plausible size, then for each
    # page corner pick the square CLOSEST to it.  Corner-proximity is the
    # decisive discriminator: a stray square annotation is never as close
    # to a physical page corner as the printed marker.
    cands = []
    for cid in range(1, n):
        area = stats[cid, cv2.CC_STAT_AREA]
        if area < 800:
            continue
        x = stats[cid, cv2.CC_STAT_LEFT]
        y = stats[cid, cv2.CC_STAT_TOP]
        bw = stats[cid, cv2.CC_STAT_WIDTH]
        bh = stats[cid, cv2.CC_STAT_HEIGHT]
        # Square-ish: reject the boundary line and elongated scribbles.
        aspect = max(bw, bh) / max(1, min(bw, bh))
        if aspect > 1.8:
            continue
        # Plausible marker size band relative to the page.  Reject huge
        # blobs (merged line corners) and tiny speckle.
        if bw > 0.18 * W or bh > 0.18 * H:
            continue
        if bw < 0.008 * W or bh < 0.008 * H:
            continue
        cands.append({
            "cx": x + bw / 2.0, "cy": y + bh / 2.0,
            "x": int(x), "y": int(y),
            "w": int(bw), "h": int(bh),
            "area": int(area),
            "cid": int(cid),
        })

    if len(cands) < 4:
        raise RuntimeError(
            f"Scan rectify: found only {len(cands)} square red marker "
            f"candidates (need 4).  Is the sheet scanned in colour mode "
            f"and are all four corner markers present?")

    # Assign each candidate to a page corner (TL, TR, BR, BL) by which
    # page corner its bbox-centre is closest to.  Guard against picking a
    # central blob when a marker is missing: the chosen square must lie
    # within ~35% of the image diagonal of its corner.
    diag = (W ** 2 + H ** 2) ** 0.5
    max_corner_dist = 0.35 * diag
    corners_ref = [
        ("TL", 0, 0),
        ("TR", W - 1, 0),
        ("BR", W - 1, H - 1),
        ("BL", 0, H - 1),
    ]
    picked = []
    used = set()
    for tag, ref_x, ref_y in corners_ref:
        best_idx = None
        best_d = float("inf")
        for i, c in enumerate(cands):
            if i in used:
                continue
            d = (c["cx"] - ref_x) ** 2 + (c["cy"] - ref_y) ** 2
            if d < best_d:
                best_d = d
                best_idx = i
        if best_idx is None or best_d ** 0.5 > max_corner_dist:
            raise RuntimeError(
                f"Scan rectify: no square marker found near {tag} corner")
        c = cands[best_idx]
        # The marker is a SQUARE registration target (a red square holding
        # a circle + crosshair) with ruler TICK MARKS along its two
        # page-centre-facing edges.  The previous axis-aligned approach
        # ("outer corner + side") broke down on slightly-skewed scans
        # because:
        #   1. ticks inflate the axis-aligned bbox toward the page centre
        #   2. small platen skew rotates the marker so neither bbox corner
        #      coincides with a real marker corner
        # Both effects vanish under a ROTATED minimum-area rectangle taken
        # over the marker's own pixels: ticks are thin lines and contribute
        # negligible area, so the rotated rect snaps to the dominant square
        # body.  We then pick the rect's corner closest to the page centre
        # as the inner corner.
        marker_pix = (lbl == c["cid"]).astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            marker_pix, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        inner_pt = None
        minrect_box: np.ndarray | None = None
        if contours:
            biggest = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(biggest)
            box = cv2.boxPoints(rect)
            minrect_box = box
            page_cx, page_cy = W / 2.0, H / 2.0
            best_d = -1.0
            for (bx, by) in box:
                # Inner corner = the one with the LONGEST vector from the
                # page corner toward page centre.  Equivalent to: closest
                # to page centre.
                d = (bx - page_cx) ** 2 + (by - page_cy) ** 2
                # Lower d = closer to page centre = inner.
                if best_d < 0 or d < best_d:
                    best_d = d
                    inner_pt = (float(bx), float(by))
        if inner_pt is None:
            # Fallback to the square-anchor heuristic.
            side = min(c["w"], c["h"])
            x0, y0, w0, h0 = c["x"], c["y"], c["w"], c["h"]
            if tag == "TL":
                inner_pt = (float(x0 + side), float(y0 + side))
            elif tag == "TR":
                inner_pt = (float(x0 + w0 - side), float(y0 + side))
            elif tag == "BR":
                inner_pt = (float(x0 + w0 - side), float(y0 + h0 - side))
            else:
                inner_pt = (float(x0 + side), float(y0 + h0 - side))
        picked.append(inner_pt)
        used.add(best_idx)
        if debug_out is not None:
            debug_out[tag] = {
                "bbox_xywh": (c["x"], c["y"], c["w"], c["h"]),
                "minrect_box": (None if minrect_box is None
                                else minrect_box.copy()),
                "inner_xy": inner_pt,
            }
    return picked


def rectify_scan(
    image_bgr: np.ndarray,
    paper_w_mm: float = PAPER_W_MM,
    paper_h_mm: float = PAPER_H_MM,
    px_per_mm: float = DEFAULT_PX_PER_MM,
) -> ScanRectifyResult:
    """Simple scan-based rectifier.

    Caller is responsible for any pre-rotation (use GUI controls or
    rotate before calling).  This function does NOT auto-rotate -
    auto-detection by image dimensions guessed the wrong direction
    in some cases.
    """
    H, W = image_bgr.shape[:2]
    red = _red_marker_mask(image_bgr)
    corner_debug: dict = {}
    pts = _find_4_corner_markers(red, image_bgr.shape, debug_out=corner_debug)

    # Build debug overlay: draw markers + connecting box
    debug = image_bgr.copy()
    for i, (x, y) in enumerate(pts):
        cv2.circle(debug, (int(x), int(y)), 30, (0, 255, 0), 5)
        cv2.putText(debug, ["TL", "TR", "BR", "BL"][i],
                   (int(x) + 35, int(y) + 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3,
                   cv2.LINE_AA)
    quad = np.array([(int(x), int(y)) for (x, y) in pts], dtype=np.int32)
    cv2.polylines(debug, [quad.reshape(-1, 1, 2)], True, (0, 255, 0), 4)

    # Compute homography to a clean (paper_w_mm * px) x (paper_h_mm * px) canvas
    out_w = int(round(paper_w_mm * px_per_mm))
    out_h = int(round(paper_h_mm * px_per_mm))
    dst = np.array([
        (0,         0),
        (out_w - 1, 0),
        (out_w - 1, out_h - 1),
        (0,         out_h - 1),
    ], dtype=np.float32)
    src = np.array(pts, dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image_bgr, M, (out_w, out_h),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(255, 255, 255))

    return ScanRectifyResult(
        warped=warped,
        marker_pixels_bgr=tuple(pts),
        debug_overlay=debug,
        corner_debug=corner_debug,
    )
