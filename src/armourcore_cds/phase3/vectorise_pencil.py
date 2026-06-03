"""Phase 3 pencil-specific pipeline.

Why a dedicated module
======================
Pencil traces are SPARSE - the L11 cleaned mask gives lots of
disconnected dots/segments rather than a continuous stroked outline.
The contour-based approach in vectorise_v4 needs each shape to be a
single connected component after closing.

Approach
========
1. AGGRESSIVE dilation first: each pencil pixel grows by ~10-15px so
   neighbouring dots merge into one solid blob per shape.
2. MORPH_CLOSE on top to bridge remaining inter-blob gaps.
3. findContours RETR_EXTERNAL: now each shape is a connected blob,
   so the external contour is the shape outline.
4. Approximation back to original stroke - subtract the dilation amount
   from each contour point along its inward normal to reverse the
   dilation expansion.  (Approximate; just shrinks contour by N pixels.)
5. RDP simplify + smooth.

This is fundamentally different from vectorise_v4 because pencil ink
is treated as a CLOUD of pixels per shape, not a stroked outline.
"""
from __future__ import annotations

import cv2
import numpy as np

from armourcore_cds.phase3.vectorise import (
    VectorPath, _rdp_simplify, _catmull_rom_to_bezier,
)
from armourcore_cds.phase3.vectorise_v3 import chaikin_smooth


def _shrink_contour(contour: np.ndarray, shrink_px: float,
                   img_shape: tuple[int, int]) -> np.ndarray:
    """Move each contour point inward by ``shrink_px`` along the local
    inward normal.  Cheap approximation of inverse-dilation."""
    if len(contour) < 3:
        return contour
    pts = contour.reshape(-1, 2).astype(np.float64)
    n = len(pts)
    # Compute centroid (used as "inside" reference)
    M = cv2.moments(contour)
    if M["m00"] <= 0:
        return contour
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    shrunk = np.empty_like(pts)
    for i in range(n):
        # Direction from point toward centroid
        dx = cx - pts[i, 0]
        dy = cy - pts[i, 1]
        d = (dx * dx + dy * dy) ** 0.5
        if d < 1e-3:
            shrunk[i] = pts[i]
            continue
        shrunk[i, 0] = pts[i, 0] + shrink_px * dx / d
        shrunk[i, 1] = pts[i, 1] + shrink_px * dy / d
    H, W = img_shape
    shrunk[:, 0] = np.clip(shrunk[:, 0], 0, W - 1)
    shrunk[:, 1] = np.clip(shrunk[:, 1], 0, H - 1)
    return shrunk.astype(np.int32).reshape(-1, 1, 2)


def extract_vector_paths_pencil(
    trace_mask: np.ndarray,
    design_width_mm: float,
    design_height_mm: float,
    dilate_px: int = 13,
    close_after_dilate_px: int = 9,
    min_area_mm2: float = 25.0,    # 5x5mm minimum tracing
    min_compactness: float = 0.005,  # very low: long-thin tracings have tiny compactness
    rdp_epsilon: float = 3.5,
    smooth_iters: int = 1,
    max_image_area_frac: float = 0.35,
) -> tuple[list[VectorPath], dict]:
    """Pencil-specific extraction.  Returns (paths, stats_dict)."""
    H, W = trace_mask.shape[:2]
    px_per_mm_x = W / design_width_mm
    px_per_mm_y = H / design_height_mm
    min_area_px = min_area_mm2 * px_per_mm_x * px_per_mm_y

    binary = (trace_mask > 0).astype(np.uint8) * 255

    # Stage 1: heavy dilation - sparse dots merge per shape
    k_dil = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))
    dilated = cv2.dilate(binary, k_dil)

    # Stage 2: morphological close - bridge remaining inter-cloud gaps
    k_close = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (close_after_dilate_px, close_after_dilate_px))
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, k_close)

    # Stage 3: external contours
    raw_contours, _ = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    paths: list[VectorPath] = []
    image_area = H * W
    stats = {"raw_components": len(raw_contours),
             "dropped_small": 0, "dropped_huge": 0,
             "dropped_slither": 0, "kept": 0}

    for c in raw_contours:
        x, y, bw, bh = cv2.boundingRect(c)
        bbox_area = bw * bh
        if bbox_area > image_area * max_image_area_frac:
            stats["dropped_huge"] += 1
            continue
        contour_area = float(cv2.contourArea(c))
        area_mm2 = contour_area / (px_per_mm_x * px_per_mm_y)
        if area_mm2 < min_area_mm2:
            stats["dropped_small"] += 1
            continue
        perim = float(cv2.arcLength(c, closed=True))
        compactness = ((4 * np.pi * contour_area)
                      / (perim * perim)) if perim > 0 else 0
        if compactness < min_compactness:
            stats["dropped_slither"] += 1
            continue

        # Approximate inverse-dilation: shrink contour inward by dilate_px
        # plus close-half so the recovered outline matches the original
        # pencil-stroke centerline rather than the dilated boundary.
        shrink_amt = dilate_px + close_after_dilate_px / 2
        shrunk = _shrink_contour(c, shrink_amt, (H, W))

        simp = _rdp_simplify(shrunk, rdp_epsilon)
        if len(simp) < 3:
            stats["dropped_slither"] += 1
            continue
        smoothed = chaikin_smooth(simp, iterations=smooth_iters, closed=True)
        segs = _catmull_rom_to_bezier(smoothed, tension=1.0)

        xs, ys = smoothed[:, 0], smoothed[:, 1]
        x0, y0 = float(xs.min()), float(ys.min())
        bw_s, bh_s = float(xs.max() - x0), float(ys.max() - y0)
        # Use shoelace on the smoothed loop for area
        sh_area = 0.5 * float(np.abs(
            np.dot(xs, np.roll(ys, 1)) - np.dot(ys, np.roll(xs, 1))))
        sh_perim = float(np.sum(np.linalg.norm(
            np.diff(np.vstack([smoothed, smoothed[0:1]]), axis=0), axis=1)))

        paths.append(VectorPath(
            points=smoothed, bezier_segments=segs,
            area_px=sh_area, perimeter_px=sh_perim,
            bbox_xywh=(int(x0), int(y0), int(bw_s), int(bh_s)),
        ))
        stats["kept"] += 1

    paths.sort(key=lambda p: p.area_px, reverse=True)
    return paths, stats
