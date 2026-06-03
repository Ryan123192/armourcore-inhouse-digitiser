"""Phase 3 - vectorise_v3 (skeleton-based).

Why a third sibling?
====================
v1 (vectorise.extract_vector_paths) uses cv2.findContours on a closed mask.
Problems that surface in practice:

  * "Devil-horn" artefacts on radiused corners - findContours wraps around
    tiny stray pixels (text fragments, grid-dash residue) that MORPH_CLOSE
    merges into a tool outline, leaving a sharp spike on what should be a
    smooth curve.

  * Pencil double-outlines - a pencil stroke ~0.5mm thick has two outside
    boundaries; findContours traces both as separate parallel paths.

  * Customer "shift" doubles - when the customer's pencil shifts mid-trace
    they leave two parallel lines around the true outline.  findContours
    gives the union outline, not the intended single shape.

v3 approach: SKELETONISE first, vectorise the centerline.

  Mask -> close -> medial_axis -> connected skeleton components
       -> per-component polyline extraction:
            closed loop -> walk the cycle once
            open arc    -> walk endpoint to endpoint
       -> bridge nearby endpoints across small gaps (graph)
       -> RDP simplify -> Chaikin smooth (fewer, smoother nodes)
       -> VectorPath dataclass

The skeleton IS the centerline so:
  * No double-outline on pencil - one path per stroke
  * Customer-shift parallel doubles collapse to their median axis
  * No findContours traversal so no devil horns

Public API
==========
    extract_vector_paths_skeleton(trace_mask, *, route, ...)
        -> list[VectorPath]

Filter via existing vectorise_v2.filter_paths.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

import cv2
import numpy as np
from skimage.morphology import medial_axis

from armourcore_cds.phase3.vectorise import (
    VectorPath, _rdp_simplify, _catmull_rom_to_bezier,
)

# Recursion can be deep for long skeletons; raise the limit a touch.
sys.setrecursionlimit(max(sys.getrecursionlimit(), 5000))


# ---------------------------------------------------------------------------
# Pre-filter: remove tiny stray components near larger ones (devil-horn fix)
# ---------------------------------------------------------------------------

def remove_stray_satellites(mask: np.ndarray,
                           keep_min_area_px: int = 60,
                           satellite_max_area_px: int = 250,
                           proximity_px: int = 10) -> np.ndarray:
    """Drop small components that sit within *proximity_px* of a larger
    component.  These are typically text fragments / dashed-line bits
    that morphological closing would otherwise merge into the bigger
    tool outline as a "devil horn" spike.
    """
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask
    big = np.zeros_like(mask)
    for cid in range(1, n):
        if stats[cid, cv2.CC_STAT_AREA] >= keep_min_area_px:
            big[lbl == cid] = 255
    if not np.any(big):
        return mask

    k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (proximity_px * 2 + 1, proximity_px * 2 + 1))
    big_neighbourhood = cv2.dilate(big, k)

    out = mask.copy()
    for cid in range(1, n):
        area = stats[cid, cv2.CC_STAT_AREA]
        if area >= keep_min_area_px:
            continue
        if area > satellite_max_area_px:
            continue
        comp = lbl == cid
        # If component lies within the neighbourhood of a bigger blob, drop.
        if (comp & (big_neighbourhood > 0) & (big == 0)).any():
            out[comp] = 0
    return out


# ---------------------------------------------------------------------------
# Skeleton polyline extraction (handles both closed loops and open arcs)
# ---------------------------------------------------------------------------

_NEIGH_8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _skel_neighbour_count(skel_u8: np.ndarray) -> np.ndarray:
    k = np.ones((3, 3), np.uint8)
    k[1, 1] = 0
    return cv2.filter2D(skel_u8, cv2.CV_8U, k)


def _walk_polyline(skel: np.ndarray, start_yx: tuple[int, int],
                  visited: np.ndarray) -> list[tuple[int, int]]:
    """Walk a polyline from *start_yx* until dead-end or back to start."""
    H, W = skel.shape
    path: list[tuple[int, int]] = [start_yx]
    visited[start_yx] = True
    cy, cx = start_yx
    while True:
        next_pix = None
        for dy, dx in _NEIGH_8:
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < H and 0 <= nx < W and skel[ny, nx] and not visited[ny, nx]:
                next_pix = (ny, nx)
                break
        if next_pix is None:
            break
        path.append(next_pix)
        visited[next_pix] = True
        cy, cx = next_pix
    return path


def extract_polylines(skel_mask: np.ndarray,
                     min_length_px: int = 80) -> list[np.ndarray]:
    """Convert a skeleton mask into ordered polylines.

    Returns a list of (N, 2) float64 arrays in (x, y) order.
    Closed loops are returned as a cycle ending where it started.
    """
    skel = (skel_mask > 0).astype(np.uint8)
    H, W = skel.shape
    nb = _skel_neighbour_count(skel)
    visited = np.zeros_like(skel, dtype=bool)

    polylines: list[np.ndarray] = []

    # First pass: walk from every endpoint (nb == 1) - open arcs.
    ep_ys, ep_xs = np.where(skel & (nb == 1))
    for ey, ex in zip(ep_ys, ep_xs):
        if visited[ey, ex]:
            continue
        path = _walk_polyline(skel, (int(ey), int(ex)), visited)
        if len(path) < min_length_px:
            continue
        pts = np.array([(x, y) for (y, x) in path], dtype=np.float64)
        polylines.append(pts)

    # Second pass: any unvisited skel pixel must belong to a CLOSED LOOP
    # (no endpoints).  Walk it as a cycle.
    rem_y, rem_x = np.where(skel & ~visited)
    for sy, sx in zip(rem_y, rem_x):
        if visited[sy, sx]:
            continue
        path = _walk_polyline(skel, (int(sy), int(sx)), visited)
        if len(path) < min_length_px:
            continue
        # Close the loop by appending the start
        path.append(path[0])
        pts = np.array([(x, y) for (y, x) in path], dtype=np.float64)
        polylines.append(pts)

    return polylines


# ---------------------------------------------------------------------------
# Endpoint bridging (gap-fill in centerline space)
# ---------------------------------------------------------------------------

def bridge_endpoints(polylines: list[np.ndarray],
                    max_gap_px: int = 40) -> list[np.ndarray]:
    """Greedy join of nearby endpoints (open arcs only).  Closed loops
    (start == end) are passed through unchanged."""
    arcs: list[np.ndarray] = []
    closed: list[np.ndarray] = []
    for p in polylines:
        if len(p) >= 4 and np.allclose(p[0], p[-1]):
            closed.append(p)
        else:
            arcs.append(p)

    changed = True
    while changed and len(arcs) > 1:
        changed = False
        endpoints = []
        for i, a in enumerate(arcs):
            endpoints.append((i, "start", a[0]))
            endpoints.append((i, "end",   a[-1]))

        best = None
        best_d = max_gap_px + 1.0
        for i in range(len(endpoints)):
            for j in range(i + 1, len(endpoints)):
                ai, ae, ap = endpoints[i]
                bi, be, bp = endpoints[j]
                if ai == bi:
                    continue
                d = float(np.hypot(ap[0] - bp[0], ap[1] - bp[1]))
                if d < best_d:
                    best_d = d
                    best = (ai, ae, bi, be)

        if best is None:
            break

        ai, ae, bi, be = best
        a = arcs[ai]
        b = arcs[bi]
        if ae == "end" and be == "start":
            merged = np.vstack([a, b])
        elif ae == "end" and be == "end":
            merged = np.vstack([a, b[::-1]])
        elif ae == "start" and be == "start":
            merged = np.vstack([a[::-1], b])
        else:
            merged = np.vstack([b, a])

        arcs = [arcs[k] for k in range(len(arcs)) if k not in (ai, bi)]
        arcs.append(merged)
        changed = True

    return closed + arcs


# ---------------------------------------------------------------------------
# Chaikin smoothing (cheap, no overshoot, fewer-node-friendly)
# ---------------------------------------------------------------------------

def chaikin_smooth(pts: np.ndarray, iterations: int = 2,
                  closed: bool = False) -> np.ndarray:
    """Corner-cutting smoothing.  Each iteration roughly doubles point
    count but rounds sharp corners (= devil horns) without overshoot."""
    p = pts.copy()
    for _ in range(iterations):
        if len(p) < 3:
            return p
        if closed:
            shifted = np.vstack([p[1:], p[:1]])
        else:
            shifted = p[1:]
            head = p[:-1]
            p = np.vstack([
                head + 0.25 * (shifted - head),
                head + 0.75 * (shifted - head),
            ]).reshape(-1, 2, order='F').reshape(-1, 2)
            # Preserve endpoints
            p = np.vstack([pts[0:1], p, pts[-1:]])
            continue
        new_p = np.empty((len(p) * 2, 2), dtype=np.float64)
        new_p[0::2] = 0.75 * p + 0.25 * shifted
        new_p[1::2] = 0.25 * p + 0.75 * shifted
        p = new_p
    return p


# ---------------------------------------------------------------------------
# Polyline -> VectorPath
# ---------------------------------------------------------------------------

def _polyline_to_path(pts: np.ndarray, rdp_epsilon: float,
                     smooth_iters: int, tension: float) -> VectorPath | None:
    if len(pts) < 4:
        return None

    closed = bool(np.allclose(pts[0], pts[-1]))
    # RDP simplify against the input
    if closed:
        contour = pts[:-1].astype(np.int32).reshape(-1, 1, 2)
    else:
        contour = pts.astype(np.int32).reshape(-1, 1, 2)
    simplified = _rdp_simplify(contour, rdp_epsilon)
    if len(simplified) < 3:
        return None

    smoothed = chaikin_smooth(simplified, iterations=smooth_iters, closed=closed)

    if closed:
        segs = _catmull_rom_to_bezier(smoothed, tension=tension)
        xs, ys = smoothed[:, 0], smoothed[:, 1]
        x0, y0 = float(xs.min()), float(ys.min())
        bw, bh = float(xs.max() - x0), float(ys.max() - y0)
        # Approximate area via shoelace on the smoothed loop
        area = 0.5 * float(np.abs(np.dot(xs, np.roll(ys, 1))
                                  - np.dot(ys, np.roll(xs, 1))))
        perim = float(np.sum(np.linalg.norm(np.diff(
            np.vstack([smoothed, smoothed[0:1]]), axis=0), axis=1)))
    else:
        # Open arc - represent as polyline of bezier "line" segments
        # (degenerate cubics where control points sit on the chord).
        segs = []
        for i in range(len(smoothed) - 1):
            p1 = smoothed[i]
            p2 = smoothed[i + 1]
            cp1 = p1 + (p2 - p1) / 3.0
            cp2 = p1 + 2.0 * (p2 - p1) / 3.0
            segs.append((p1.copy(), cp1, cp2, p2.copy()))
        xs, ys = smoothed[:, 0], smoothed[:, 1]
        x0, y0 = float(xs.min()), float(ys.min())
        bw, bh = float(xs.max() - x0), float(ys.max() - y0)
        # Use bbox area for filtering thresholds
        area = float(bw * bh)
        perim = float(np.sum(np.linalg.norm(np.diff(smoothed, axis=0), axis=1)))

    return VectorPath(
        points=smoothed,
        bezier_segments=segs,
        area_px=area,
        perimeter_px=perim,
        bbox_xywh=(int(x0), int(y0), int(bw), int(bh)),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

ROUTE_SKEL_PRESETS: dict[str, dict] = {
    "pen": {
        "pre_close_px":       9,
        "stray_proximity_px": 12,
        "skel_prune_spurs":   25,    # drop skeleton branches < 25px
        "skel_min_length":    100,   # drop polylines < ~25mm long
        "bridge_max_gap":     40,
        "rdp_epsilon":        4.5,   # fewer nodes
        "smooth_iters":       1,     # one Chaikin pass is enough
    },
    "pencil": {
        "pre_close_px":       15,
        "stray_proximity_px": 18,
        "skel_prune_spurs":   20,
        "skel_min_length":    80,
        "bridge_max_gap":     60,
        "rdp_epsilon":        3.5,
        "smooth_iters":       1,
    },
}


def _prune_skeleton_spurs(skel_u8: np.ndarray, max_spur_px: int) -> np.ndarray:
    """Iteratively remove skeleton endpoints up to max_spur_px steps.
    Cleans short branches off junctions (sources of stray paths)."""
    out = skel_u8.copy()
    for _ in range(max_spur_px):
        nb = _skel_neighbour_count(out)
        eps = (out > 0) & (nb == 1)
        if not eps.any():
            break
        out[eps] = 0
    return out


def extract_vector_paths_skeleton(
    trace_mask: np.ndarray,
    route: str = "pen",
    **overrides,
) -> tuple[list[VectorPath], dict]:
    """Skeleton-based vector extraction.  Returns (paths, params_used)."""
    cfg = ROUTE_SKEL_PRESETS.get(route, ROUTE_SKEL_PRESETS["pen"]).copy()
    cfg.update(overrides)

    binary = (trace_mask > 0).astype(np.uint8) * 255

    # 1. Strip stray satellites (devil-horn sources)
    binary = remove_stray_satellites(
        binary, proximity_px=cfg["stray_proximity_px"],
    )

    # 2. Close to bridge sub-px breaks
    k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (cfg["pre_close_px"], cfg["pre_close_px"]))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)

    # 3. Medial axis (skeleton) + spur pruning
    skel = medial_axis(closed > 0).astype(np.uint8) * 255
    if cfg.get("skel_prune_spurs", 0) > 0:
        skel = _prune_skeleton_spurs(skel, cfg["skel_prune_spurs"])

    # 4. Walk skeleton into polylines
    polylines = extract_polylines(skel, min_length_px=cfg["skel_min_length"])

    # 5. Bridge nearby endpoints (gap-fill in centerline space)
    polylines = bridge_endpoints(polylines, max_gap_px=cfg["bridge_max_gap"])

    # 6. Build VectorPaths (RDP + Chaikin smoothing)
    paths: list[VectorPath] = []
    for pl in polylines:
        vp = _polyline_to_path(
            pl, rdp_epsilon=cfg["rdp_epsilon"],
            smooth_iters=cfg["smooth_iters"], tension=1.0)
        if vp is not None:
            paths.append(vp)

    paths.sort(key=lambda p: p.area_px, reverse=True)
    return paths, cfg
