"""Self-contained scan rectifier.

Standalone copy of ``armourcore_cds.phase1.marker_rectify_scan`` with the
two cross-module constants inlined, so this folder is fully portable -
move it anywhere on the machine and Rectify.bat keeps working.

Sync notes
==========
If you make a correctness fix here, copy it back to
``src/armourcore_cds/phase1/marker_rectify_scan.py`` (and vice versa).
We deliberately keep these two files identical EXCEPT for the import
block at the top.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Inlined from armourcore_cds.phase1.marker_rectify_fast_v4 so this file
# has no project imports.  These are the calibrated GLOBAL CDS defaults;
# the caller can override at the rectify_scan() call.
PAPER_W_MM = 600.0
PAPER_H_MM = 500.0
DEFAULT_PX_PER_MM = 10.0


@dataclass
class ScanRectifyResult:
    warped: np.ndarray
    marker_pixels_bgr: tuple        # 4 (x, y) marker centroids
    debug_overlay: np.ndarray       # source img with markers drawn


def _red_marker_mask(image_bgr: np.ndarray) -> np.ndarray:
    """Pull out the four red corner-marker blobs.

    Tight thresholds because the CDS template prints brand-accent
    orange (#F1B01D) for size indicators + logo highlights right next
    to the TL marker.  Without the G<110 ceiling, anti-aliased orange
    pixels leak into the mask and morph-close fuses them to the marker
    blob - which then has a 2:1+ vertical aspect and pulls the rotated
    inner corner off by several mm.  See marker_rectify_scan.py for
    the full rationale.
    """
    # Tightened hue band (H<5 | H>170) surgically rejects the H=5-15
    # transition zone where orange-edge anti-aliasing drifts toward red.
    # BGR / S / V kept loose so faded marker prints survive.
    # See marker_rectify_scan.py for full rationale.
    B, G, R = cv2.split(image_bgr)
    red_bgr = (R.astype(np.int16) - np.maximum(G, B).astype(np.int16) > 30) & (R > 100)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    red_hsv = ((H < 5) | (H > 170)) & (S > 60) & (V > 60)
    mask = (red_bgr & red_hsv).astype(np.uint8) * 255
    H, W = mask.shape[:2]
    close_k = max(15, int(round(min(H, W) * 0.01)))
    if close_k % 2 == 0:
        close_k += 1
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k)))
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    return mask


def _find_4_corner_markers(red_mask: np.ndarray,
                          image_shape) -> list[tuple[float, float]]:
    """Return 4 inner-corner points ordered TL, TR, BR, BL."""
    H, W = image_shape[:2]
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(
        red_mask, connectivity=8)
    cands = []
    for cid in range(1, n):
        area = stats[cid, cv2.CC_STAT_AREA]
        if area < 800:
            continue
        x = stats[cid, cv2.CC_STAT_LEFT]
        y = stats[cid, cv2.CC_STAT_TOP]
        bw = stats[cid, cv2.CC_STAT_WIDTH]
        bh = stats[cid, cv2.CC_STAT_HEIGHT]
        aspect = max(bw, bh) / max(1, min(bw, bh))
        if aspect > 1.8:
            continue
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
        # Use the marker pixels' min-area rotated rect to find a
        # skew-robust inner corner.
        marker_pix = (lbl == c["cid"]).astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            marker_pix, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        inner_pt = None
        if contours:
            biggest = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(biggest)
            box = cv2.boxPoints(rect)
            page_cx, page_cy = W / 2.0, H / 2.0
            best_d = -1.0
            for (bx, by) in box:
                d = (bx - page_cx) ** 2 + (by - page_cy) ** 2
                if best_d < 0 or d < best_d:
                    best_d = d
                    inner_pt = (float(bx), float(by))
        if inner_pt is None:
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
    return picked


def rectify_scan(
    image_bgr: np.ndarray,
    paper_w_mm: float = PAPER_W_MM,
    paper_h_mm: float = PAPER_H_MM,
    px_per_mm: float = DEFAULT_PX_PER_MM,
) -> ScanRectifyResult:
    H, W = image_bgr.shape[:2]
    red = _red_marker_mask(image_bgr)
    pts = _find_4_corner_markers(red, image_bgr.shape)

    debug = image_bgr.copy()
    for i, (x, y) in enumerate(pts):
        cv2.circle(debug, (int(x), int(y)), 30, (0, 255, 0), 5)
        cv2.putText(debug, ["TL", "TR", "BR", "BL"][i],
                   (int(x) + 35, int(y) + 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3,
                   cv2.LINE_AA)
    quad = np.array([(int(x), int(y)) for (x, y) in pts], dtype=np.int32)
    cv2.polylines(debug, [quad.reshape(-1, 1, 2)], True, (0, 255, 0), 4)

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
    )
