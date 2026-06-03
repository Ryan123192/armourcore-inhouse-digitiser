"""Self-diagnostic: classify each extracted path so I can debug without
asking the user every time.

Categories
==========
    REAL    - looks like a tool tracing (organic curves, area in tool
              range, not aligned to grid).
    GRID    - axis-aligned rectangle near a major grid intersection.
              These are residual major-grid artefacts that survived
              Phase 2 - tells me to strengthen grid_strip.
    TEXT    - small bbox with low aspect ratio.  Often "B", "L", etc.
              from "Boundary Line (Small Insert)" template text.
    SLIVER  - extreme aspect ratio.  Stroke fragment / grid dash.
    NOISE   - very small, no clear shape.

Output
======
Annotated PNG with each path drawn in its category colour + a JSON
report listing reasons.  Lets me see at a glance "img1 has 7 GRID
paths" -> I know to dial up grid_strip.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from armourcore_cds.phase3.vectorise import VectorPath


PathCategory = Literal["REAL", "TEXT", "SLIVER", "NOISE"]

CATEGORY_BGR = {
    "REAL":   (40, 180, 40),    # green
    "TEXT":   (30, 140, 220),   # orange
    "SLIVER": (200, 60, 200),   # magenta
    "NOISE":  (40, 40, 220),    # red
}


@dataclass
class ClassifiedPath:
    index: int
    category: PathCategory
    reasons: list[str] = field(default_factory=list)
    bbox_xywh: tuple[int, int, int, int] = (0, 0, 0, 0)
    area_mm2: float = 0.0
    bbox_w_mm: float = 0.0
    bbox_h_mm: float = 0.0
    aspect_ratio: float = 0.0
    compactness: float = 0.0
    bbox_to_grid_offset_px: float = 0.0
    rectangularity: float = 0.0


def _is_near_grid_position(x: float, y: float, w: float, h: float,
                          px_per_mm_x: float, px_per_mm_y: float,
                          tolerance_px: float = 8.0) -> bool:
    """True if the bbox corners are close to multiples of 50mm."""
    def near_50mm(coord_px, scale):
        coord_mm = coord_px / scale
        nearest = round(coord_mm / 50.0) * 50.0
        return abs(coord_mm - nearest) * scale < tolerance_px

    return (near_50mm(x, px_per_mm_x) or near_50mm(x + w, px_per_mm_x) or
            near_50mm(y, px_per_mm_y) or near_50mm(y + h, px_per_mm_y))


def _rectangularity(path: VectorPath) -> float:
    """Ratio of contour area to bbox area.  Pure rectangle = 1.0."""
    x, y, w, h = path.bbox_xywh
    if w * h == 0:
        return 0.0
    return float(path.area_px) / float(w * h)


def classify_paths(
    paths: list[VectorPath],
    mask_shape: tuple[int, int],
    design_width_mm: float,
    design_height_mm: float,
) -> list[ClassifiedPath]:
    H, W = mask_shape
    px_per_mm_x = W / design_width_mm
    px_per_mm_y = H / design_height_mm
    mm_per_px = min(1.0 / px_per_mm_x, 1.0 / px_per_mm_y)

    results: list[ClassifiedPath] = []
    for idx, p in enumerate(paths):
        x, y, w, h = p.bbox_xywh
        area_mm2 = p.area_px / (px_per_mm_x * px_per_mm_y)
        bbox_w_mm = w / px_per_mm_x
        bbox_h_mm = h / px_per_mm_y
        aspect = max(w, h) / max(min(w, h), 1.0)
        compactness = ((4 * np.pi * p.area_px) /
                      (p.perimeter_px ** 2)) if p.perimeter_px > 0 else 0
        rect = _rectangularity(p)
        near_grid = _is_near_grid_position(x, y, w, h,
                                          px_per_mm_x, px_per_mm_y)

        category: PathCategory = "REAL"
        reasons: list[str] = []

        # NB: GRID category retired - the GLOBAL CDS no longer prints a
        # major grid, so any rectangle that survives Phase 2 is either a
        # real rectangular tool body, the body of a tape measure / steel
        # ruler, or a fragment of the page border (the latter is killed by
        # the 2mm outer-border strip before Phase 3 ever runs).

        # TEXT: small + low aspect + small area
        if bbox_w_mm < 12 and bbox_h_mm < 12 and area_mm2 < 50:
            category = "TEXT"
            reasons.append(f"bbox {bbox_w_mm:.1f}x{bbox_h_mm:.1f} mm tiny")

        # SLIVER: only the most extreme cases.  Real tools include long
        # narrow shapes (screwdrivers, files, chisels) - don't kill those.
        # Catch only the obvious artefacts: very-skinny AND tiny.
        elif aspect > 12 and area_mm2 < 200:
            category = "SLIVER"
            reasons.append(
                f"aspect={aspect:.1f} > 12 AND area={area_mm2:.0f}mm² "
                f"< 200 (skinny-tiny artefact)")
        elif min(bbox_w_mm, bbox_h_mm) < 4.0:
            category = "SLIVER"
            reasons.append(
                f"short bbox dim {min(bbox_w_mm, bbox_h_mm):.1f}mm < 4 "
                f"(sub-pencil-stroke artefact)")
        # Compactness floor only triggers on tiny shapes (< 100 mm²)
        elif compactness < 0.05 and area_mm2 < 100:
            category = "SLIVER"
            reasons.append(
                f"compactness={compactness:.3f} < 0.05 AND area "
                f"< 100mm² (meandering dust)")

        # NOISE: very low area regardless of other
        elif area_mm2 < 30:
            category = "NOISE"
            reasons.append(f"area_mm2={area_mm2:.1f} < 30")

        else:
            reasons.append("organic shape, in size range")

        results.append(ClassifiedPath(
            index=idx, category=category, reasons=reasons,
            bbox_xywh=p.bbox_xywh,
            area_mm2=area_mm2,
            bbox_w_mm=bbox_w_mm, bbox_h_mm=bbox_h_mm,
            aspect_ratio=aspect, compactness=compactness,
            bbox_to_grid_offset_px=0.0,
            rectangularity=rect,
        ))
    return results


def render_classified(paths: list[VectorPath],
                     classified: list[ClassifiedPath],
                     mask_shape: tuple[int, int],
                     thickness: int = 3) -> np.ndarray:
    """Draw paths coloured by their classification + label each one."""
    from armourcore_cds.phase3.vectorise import _sample_path
    H, W = mask_shape
    canvas = np.full((H, W, 3), 255, dtype=np.uint8)

    # Legend strip
    legend_h = 36
    cv2.rectangle(canvas, (0, 0), (W, legend_h), (245, 245, 245), -1)
    cv2.line(canvas, (0, legend_h), (W, legend_h), (180, 180, 180), 1)
    cx = 10
    for cat, colour in CATEGORY_BGR.items():
        cv2.rectangle(canvas, (cx, 8), (cx + 24, 28), colour, -1)
        cv2.putText(canvas, cat, (cx + 30, 26),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1,
                   cv2.LINE_AA)
        cx += 130

    for vp, cp in zip(paths, classified):
        colour = CATEGORY_BGR[cp.category]
        pts = _sample_path(vp)
        if len(pts) < 2:
            continue
        pts_int = pts.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts_int], isClosed=True, color=colour,
                     thickness=thickness)
        x, y, w, h = vp.bbox_xywh
        cv2.putText(canvas, str(cp.index), (x + 4, y + 18),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
    return canvas


def category_counts(classified: list[ClassifiedPath]) -> dict[PathCategory, int]:
    counts = {"REAL": 0, "TEXT": 0, "SLIVER": 0, "NOISE": 0}
    for c in classified:
        counts[c.category] += 1
    return counts


def write_classification_report(classified: list[ClassifiedPath],
                               path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = category_counts(classified)
    payload = {
        "summary": counts,
        "paths": [asdict(c) for c in classified],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
