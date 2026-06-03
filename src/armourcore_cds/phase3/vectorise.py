"""Phase 3: raster-to-vector conversion with smooth cubic Bezier curves.

Strategy
--------
0. Orange-gap repair (``repair_orange_gaps``) -- the Phase 2 orange-removal
   pass leaves gaps wherever grid lines crossed the customer's tool traces.
   Since the orange mask records exactly which pixels were removed, we dilate
   existing trace ink *only into those pixels*, bridging every gap without
   risk of merging unrelated traces (the repair is geometrically confined to
   the grid-line footprint).
1. ``cv2.findContours`` with ``RETR_EXTERNAL`` / ``CHAIN_APPROX_NONE`` --
   dense pixel chains give the most accurate curve-fitting data.
2. RDP simplification (``cv2.approxPolyDP``) -- reduces node count while
   preserving shape.  Epsilon is expressed in trace-mask pixels.
3. Catmull-Rom -> cubic Bezier -- C1-continuous smooth curves; each chord
   uses the surrounding points as tangent guides.
4. SVG export -- one ``<path>`` per contour, coordinates in millimetres.
5. Overlay rendering -- Bezier curves sampled and drawn on the cleaned raster.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence
import xml.etree.ElementTree as ET

import time

import cv2
import numpy as np
from skimage.morphology import medial_axis as _skimage_medial_axis


# ---------------------------------------------------------------------------
# Centerline-graph completion
# ---------------------------------------------------------------------------
#
# Strategy: instead of trying to fill bitmap pixels in the orange-mask
# footprint, we extract the *centerline* (medial axis) of every trace stub,
# treat broken-line endpoints as nodes in a graph, and match facing endpoint
# pairs to bridge gaps with smooth Bézier completions.  Working in the
# centerline domain is far more robust than pixel-level dilation:
#
# * The medial axis is mathematically defined; it does not introduce
#   morphological artefacts the way iterative erosion does.
# * Skeleton pruning eliminates spurious branches from rough trace edges,
#   which previously created phantom endpoints (= random green artefacts).
# * Endpoint direction comes from walking back along the skeleton — a
#   precise tangent, not a PCA estimate over a noisy cluster.
# * The local trace thickness is read directly from the medial axis
#   distance map, so completion curves taper to match the trace width.
# * Matching is closure-aware: pairs are scored by direction continuity
#   and distance, then greedily resolved best-first.
# ---------------------------------------------------------------------------

def _skel_neighbour_count(skel_u8: np.ndarray) -> np.ndarray:
    """Count of skeleton 8-neighbours for every pixel (uint8 array)."""
    kernel = np.ones((3, 3), np.uint8)
    kernel[1, 1] = 0
    return cv2.filter2D(skel_u8, cv2.CV_8U, kernel)


def _walk_skeleton(
    skel_bool: np.ndarray,
    start_x: int,
    start_y: int,
    n_steps: int,
    forbid_junctions: bool = False,
    neighbor_count: np.ndarray | None = None,
) -> list[tuple[int, int]]:
    """Walk *n_steps* along the skeleton from (start_x, start_y).

    Returns the path as a list of (x, y) tuples (including the start).
    Stops early at a dead-end, or — when *forbid_junctions* is True — when
    a junction (degree ≥ 3) is reached.
    """
    H, W = skel_bool.shape
    path: list[tuple[int, int]] = [(start_x, start_y)]
    visited: set[tuple[int, int]] = {(start_x, start_y)}
    cx, cy = start_x, start_y
    for _ in range(n_steps):
        if (
            forbid_junctions
            and neighbor_count is not None
            and len(path) > 1
            and neighbor_count[cy, cx] >= 3
        ):
            break
        found = False
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = cx + dx, cy + dy
                if (
                    (nx, ny) not in visited
                    and 0 <= nx < W
                    and 0 <= ny < H
                    and skel_bool[ny, nx]
                ):
                    path.append((nx, ny))
                    visited.add((nx, ny))
                    cx, cy = nx, ny
                    found = True
                    break
            if found:
                break
        if not found:
            break
    return path


def _prune_skeleton(
    skel: np.ndarray,
    min_branch_len: int = 10,
    max_iter: int = 6,
) -> np.ndarray:
    """Iteratively remove short branches from a skeleton.

    A branch is "short" when an endpoint reaches a junction within
    *min_branch_len* pixels.  Such branches are typically morphological
    noise from rough trace edges and create phantom endpoints.

    Returns a uint8 (0/255) mask of the pruned skeleton.
    """
    skel_u8 = (skel > 0).astype(np.uint8)
    H, W = skel_u8.shape

    for _ in range(max_iter):
        nb = _skel_neighbour_count(skel_u8)
        endpoint_mask = (skel_u8 > 0) & (nb == 1)
        ep_ys, ep_xs = np.where(endpoint_mask)
        if len(ep_xs) == 0:
            break

        skel_bool = skel_u8 > 0
        to_clear: list[tuple[int, int]] = []

        for ex, ey in zip(ep_xs.tolist(), ep_ys.tolist()):
            # Walk up to (min_branch_len + 1) pixels — stop at a junction
            path = [(ex, ey)]
            visited = {(ex, ey)}
            cx, cy = ex, ey
            hit_junction = False
            for step in range(min_branch_len + 1):
                if step > 0 and nb[cy, cx] >= 3:
                    hit_junction = True
                    break
                found = False
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = cx + dx, cy + dy
                        if (
                            (nx, ny) not in visited
                            and 0 <= nx < W
                            and 0 <= ny < H
                            and skel_bool[ny, nx]
                        ):
                            path.append((nx, ny))
                            visited.add((nx, ny))
                            cx, cy = nx, ny
                            found = True
                            break
                    if found:
                        break
                if not found:
                    break

            if hit_junction:
                # Drop everything except the junction pixel itself
                to_clear.extend(path[:-1])

        if not to_clear:
            break

        for px, py in to_clear:
            skel_u8[py, px] = 0

    return skel_u8 * 255


def _line_obstacle_score(
    trace_binary: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    n_samples: int = 30,
) -> float:
    """Fraction of the inner 70 % of A→B straight line that hits trace.

    High score = the straight chord crosses unrelated trace pixels (bad pair).
    The first/last 15 % is excluded since both endpoints are by definition
    on the trace.
    """
    pts = np.linspace(A, B, n_samples)
    inner = pts[int(0.15 * n_samples): int(0.85 * n_samples)]
    H, W = trace_binary.shape[:2]
    hits = 0
    for px, py in inner:
        ix, iy = int(px), int(py)
        if 0 <= ix < W and 0 <= iy < H and trace_binary[iy, ix] > 0:
            hits += 1
    return hits / max(1, len(inner))


def _compute_tool_groups(
    trace_binary: np.ndarray,
    dilate_radius: int = 25,
) -> np.ndarray:
    """Group all trace stubs that probably belong to the same tool.

    Dilates the trace mask by *dilate_radius* and labels connected components
    of the result.  Stubs that are within ``2 × dilate_radius`` of each other
    end up in the same component (= same tool).  Different tools, separated
    by clean whitespace, stay in separate components.

    Returns an int32 label image (0 = background, 1..N = tool group IDs).

    Pick *dilate_radius* slightly smaller than the typical tool-to-tool
    spacing in the image so neighbouring tools don't merge accidentally.
    """
    r = max(1, int(dilate_radius))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    dilated = cv2.dilate(trace_binary, kernel)
    _, labels = cv2.connectedComponents(
        (dilated > 0).astype(np.uint8), connectivity=8
    )
    return labels


def _find_ray_target(
    trace_binary: np.ndarray,
    pos: np.ndarray,
    direction: np.ndarray,
    max_search: int,
    skip_pixels: int = 2,
) -> tuple[np.ndarray | None, float | None]:
    """Cast a ray from *pos* along *direction*; return the first trace pixel
    encountered AFTER walking through empty (background) space.

    This is the geometric heart of the matching algorithm.  An endpoint that
    sits at the start of a real gap will find another trace stub on the
    other side of the gap.  A phantom endpoint embedded in trace will fail
    immediately (no empty space → no target).

    Returns
    -------
    (target_xy, distance) — sub-pixel target and the integer step count it
    took to reach it.  ``(None, None)`` if no trace was hit after empty
    space within *max_search* steps, or the ray walked off the image.
    """
    H, W = trace_binary.shape[:2]
    in_empty = False
    for o in range(skip_pixels, max_search + 1):
        px_f = pos[0] + o * direction[0]
        py_f = pos[1] + o * direction[1]
        ix = int(round(px_f))
        iy = int(round(py_f))
        if not (0 <= ix < W and 0 <= iy < H):
            return None, None
        if trace_binary[iy, ix] > 0:
            if in_empty:
                return np.array([px_f, py_f]), float(o)
            # else: still on the originating stub — keep walking
        else:
            in_empty = True
    return None, None


def _filter_small_components(
    binary: np.ndarray,
    min_area_px: int = 30,
) -> np.ndarray:
    """Remove connected components with area < *min_area_px* pixels.

    Tiny components in the trace mask come from scanning noise / dust /
    label specks.  After medial-axis they create isolated short skeleton
    fragments with phantom endpoints that pollute the matching.
    """
    if min_area_px <= 0:
        return binary
    n_lbl, lbl, stats, _ = cv2.connectedComponentsWithStats(
        (binary > 0).astype(np.uint8), connectivity=8
    )
    if n_lbl <= 1:
        return binary
    # Label 0 is the BACKGROUND — must stay False so the output stays binary.
    # Foreground components 1..N-1 are kept only if large enough.
    keep = np.zeros(n_lbl, dtype=bool)
    for c in range(1, n_lbl):
        if stats[c, cv2.CC_STAT_AREA] >= min_area_px:
            keep[c] = True
    return (np.where(keep[lbl], 255, 0)).astype(np.uint8)


def _render_tapered_bezier(
    canvas: np.ndarray,
    desc_a: dict,
    desc_b: dict,
) -> None:
    """Draw a cubic Hermite Bézier from desc_a to desc_b on *canvas*.

    Thickness tapers linearly from desc_a["thick"] to desc_b["thick"].
    Tangents at both endpoints are taken from the descriptors so the curve
    leaves each endpoint smoothly along the trace direction.
    """
    A  = desc_a["pos"];  B  = desc_b["pos"]
    dA = desc_a["dir"];  dB = desc_b["dir"]
    tA = desc_a["thick"]; tB = desc_b["thick"]

    dist = float(np.linalg.norm(B - A))
    scale = dist / 3.0
    P0 = A
    P1 = A + scale * dA
    P2 = B - scale * dB
    P3 = B

    n_samp = max(20, int(dist))
    ts = np.linspace(0.0, 1.0, n_samp)
    mt = 1.0 - ts
    curve = (
        mt[:, None] ** 3 * P0
        + 3 * mt[:, None] ** 2 * ts[:, None] * P1
        + 3 * mt[:, None] * ts[:, None] ** 2 * P2
        + ts[:, None] ** 3 * P3
    )

    H, W = canvas.shape[:2]
    for k in range(len(curve) - 1):
        t = (k + 0.5) / max(1, len(curve) - 1)
        thickness = max(2, int(round(tA * (1.0 - t) + tB * t)))
        p1 = (int(round(curve[k][0])),     int(round(curve[k][1])))
        p2 = (int(round(curve[k + 1][0])), int(round(curve[k + 1][1])))
        cv2.line(canvas, p1, p2, 255, thickness, cv2.LINE_AA)




# =====================================================================
# Polyline extraction from a pruned skeleton
# =====================================================================
#
# Each connected skeleton component (= one broken trace stub) is reduced
# to a single polyline: the diameter of the component (longest path
# between two extremities).  This collapses Y-junctions and noise branches
# into a single principal centerline per stub.
#
# Components with no degree-1 endpoints (closed-loop skeletons = already
# complete tool outlines) are skipped.
# =====================================================================

def _bfs_farthest_in_component(
    skel_u8: np.ndarray,
    comp_mask: np.ndarray,
    start_x: int,
    start_y: int,
) -> tuple[int, int]:
    """BFS over skeleton pixels of one component; return the farthest pixel."""
    from collections import deque
    H, W = skel_u8.shape
    visited = np.zeros((H, W), dtype=bool)
    visited[start_y, start_x] = True
    queue = deque([(start_x, start_y, 0)])
    farthest_x, farthest_y, farthest_d = start_x, start_y, 0
    while queue:
        cx, cy, d = queue.popleft()
        if d > farthest_d:
            farthest_x, farthest_y, farthest_d = cx, cy, d
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = cx + dx, cy + dy
                if (
                    0 <= nx < W and 0 <= ny < H
                    and skel_u8[ny, nx] > 0
                    and comp_mask[ny, nx]
                    and not visited[ny, nx]
                ):
                    visited[ny, nx] = True
                    queue.append((nx, ny, d + 1))
    return farthest_x, farthest_y


def _bfs_path_in_component(
    skel_u8: np.ndarray,
    comp_mask: np.ndarray,
    sx: int,
    sy: int,
    ex: int,
    ey: int,
) -> list[tuple[int, int]] | None:
    """BFS shortest path between two skeleton pixels of the same component."""
    from collections import deque
    H, W = skel_u8.shape
    visited = np.zeros((H, W), dtype=bool)
    visited[sy, sx] = True
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    queue = deque([(sx, sy)])
    found = False
    while queue:
        cx, cy = queue.popleft()
        if cx == ex and cy == ey:
            found = True
            break
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = cx + dx, cy + dy
                if (
                    0 <= nx < W and 0 <= ny < H
                    and skel_u8[ny, nx] > 0
                    and comp_mask[ny, nx]
                    and not visited[ny, nx]
                ):
                    visited[ny, nx] = True
                    parent[(nx, ny)] = (cx, cy)
                    queue.append((nx, ny))
    if not found:
        return None
    # Walk back via parent pointers
    path: list[tuple[int, int]] = []
    cur: tuple[int, int] | None = (ex, ey)
    while cur is not None:
        path.append(cur)
        cur = parent.get(cur)  # type: ignore[arg-type]
    path.reverse()
    return path


def _extract_skeleton_polylines(
    skel_pruned: np.ndarray,
    distance_map: np.ndarray,
    min_polyline_length: int = 15,
) -> list[dict]:
    """Extract one polyline per skeleton connected component.

    For each component:
      * If it has no endpoints (closed loop), skip — already complete.
      * Otherwise, find the geodesic diameter (longest path) and trace it.

    Returns a list of polyline dicts:
      ``{'points': (N, 2) float64, 'thickness_a': float, 'thickness_b': float,
         'skel_component': int}``

    Polylines shorter than *min_polyline_length* skeleton pixels are dropped
    as noise.
    """
    skel_u8 = (skel_pruned > 0).astype(np.uint8)
    H, W = skel_u8.shape
    n_comp, comp_labels = cv2.connectedComponents(skel_u8, connectivity=8)

    # Neighbour count for endpoint detection
    kernel = np.ones((3, 3), np.uint8)
    kernel[1, 1] = 0
    nb = cv2.filter2D(skel_u8, cv2.CV_8U, kernel)

    polylines: list[dict] = []

    for c in range(1, n_comp):
        comp_mask = (comp_labels == c)
        comp_size = int(comp_mask.sum())
        if comp_size < min_polyline_length:
            continue

        # Endpoints in this component
        ep_y, ep_x = np.where(comp_mask & (nb == 1))
        if len(ep_x) == 0:
            # No endpoints → closed loop → already-complete tool, skip
            continue

        # Diameter: BFS twice
        s_x, s_y = int(ep_x[0]), int(ep_y[0])
        f_x, f_y = _bfs_farthest_in_component(skel_u8, comp_mask, s_x, s_y)
        e_x, e_y = _bfs_farthest_in_component(skel_u8, comp_mask, f_x, f_y)

        # Trace the diameter path
        path = _bfs_path_in_component(skel_u8, comp_mask, f_x, f_y, e_x, e_y)
        if path is None or len(path) < min_polyline_length:
            continue

        points = np.array(path, dtype=np.float64)  # (N, 2) as (x, y)
        ta = float(np.clip(distance_map[int(points[0, 1]),  int(points[0, 0])]  * 2.0, 2.0, 35.0))
        tb = float(np.clip(distance_map[int(points[-1, 1]), int(points[-1, 0])] * 2.0, 2.0, 35.0))

        polylines.append({
            "points":         points,
            "thickness_a":    ta,
            "thickness_b":    tb,
            "skel_component": int(c),
        })

    return polylines


def _polyline_tangent(
    points: np.ndarray,
    end: str = "a",
    lookback: int = 20,
) -> np.ndarray | None:
    """Outward tangent at one end of a polyline.

    end='a' → tangent at points[0] pointing away from the polyline body.
    end='b' → tangent at points[-1] pointing away.
    """
    n = len(points)
    if n < 2:
        return None
    if end == "a":
        far_idx = min(lookback, n - 1)
        d = points[0] - points[far_idx]
    else:
        far_idx = max(0, n - 1 - lookback)
        d = points[-1] - points[far_idx]
    norm = float(np.linalg.norm(d))
    if norm < 1e-9:
        return None
    return d / norm


# =====================================================================
# Alignment graph: link polylines whose endpoint rays land on each other
# =====================================================================
#
# This handles long internal gaps that the dilation tool-group cannot
# bridge (e.g., a screwdriver shaft cut by a wide grid line — the two
# stubs may be too far apart to merge under R=15 dilation, but they DO
# point at each other).
#
# Two polylines become adjacent iff one's endpoint ray (cast through
# empty space) lands within target_tol of the OTHER polyline's endpoint.
# =====================================================================

def _build_alignment_graph(
    polylines: list[dict],
    trace_binary: np.ndarray,
    max_search: int = 400,
    target_tol_px: int = 30,
    lookback_px: int = 20,
) -> list[set[int]]:
    """Adjacency list: two polylines are linked when one's endpoint ray
    lands near the other's endpoint.

    Returns a list of length len(polylines), each element a set of
    polyline indices that are directly linked to the corresponding one.
    """
    n = len(polylines)
    if n < 2:
        return [set() for _ in range(n)]

    # Flatten endpoints into a KDTree for fast lookup near ray targets
    endpoints_xy: list[np.ndarray] = []
    endpoint_owner: list[int] = []
    for i, pl in enumerate(polylines):
        endpoints_xy.append(pl["points"][0])
        endpoint_owner.append(i)
        endpoints_xy.append(pl["points"][-1])
        endpoint_owner.append(i)
    endpoints_arr = np.array(endpoints_xy)

    from scipy.spatial import cKDTree
    tree = cKDTree(endpoints_arr)

    # Step 1: collect every directed "i's ray landed near j's endpoint" hit.
    directed: set[tuple[int, int]] = set()
    for i, pl in enumerate(polylines):
        for end in ("a", "b"):
            tangent = _polyline_tangent(pl["points"], end, lookback_px)
            if tangent is None:
                continue
            origin = pl["points"][0] if end == "a" else pl["points"][-1]

            target, _ = _find_ray_target(
                trace_binary, origin, tangent, max_search=max_search,
            )
            if target is None:
                continue

            nearby_ep = tree.query_ball_point(target, r=float(target_tol_px))
            for ep_idx in nearby_ep:
                j = endpoint_owner[ep_idx]
                if j != i:
                    directed.add((i, j))

    # Step 2: only mutual hits become adjacency edges.  Bidirectional
    # agreement is the structural fix that keeps unrelated tools apart —
    # one tool's endpoint ray accidentally reaching another tool's trace
    # is fine, but it's *very* rare for both directions to agree if they
    # don't actually share an outline.
    adjacency: list[set[int]] = [set() for _ in range(n)]
    for (i, j) in directed:
        if (j, i) in directed:
            adjacency[i].add(j)
            adjacency[j].add(i)

    return adjacency


# =====================================================================
# Tool-group resolution: union of (tight dilation) ∪ (alignment edges)
# =====================================================================
#
# Two-criterion union-find — polylines belong to the same tool if EITHER:
#   * They are spatially close (tight dilation: stubs within 2·R px of
#     each other), OR
#   * Their endpoint rays bidirectionally agree across a gap (alignment).
#
# The dilation radius is critical: TOO LARGE merges parallel adjacent
# tools (closely-spaced screwdrivers get falsely linked), TOO SMALL fails
# to merge legitimate within-tool stubs that are close but not perfectly
# aligned.  Default R=10 (reach = 20 px) is calibrated for the typical
# tool spacing on the test scans.
# =====================================================================

def _final_groups(
    polylines: list[dict],
    alignment_adj: list[set[int]],
    tool_groups: np.ndarray,
) -> list[list[int]]:
    """Combine alignment edges and dilation tool-group memberships via
    union-find.  Two polylines end up in the same final group iff either
    mechanism links them.
    """
    n = len(polylines)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    # 1) Alignment edges
    for i, neighbours in enumerate(alignment_adj):
        for j in neighbours:
            union(i, j)

    # 2) Polylines sharing a (tight) dilation tool group
    H, W = tool_groups.shape
    by_dil_group: dict[int, list[int]] = {}
    for i, pl in enumerate(polylines):
        mid = pl["points"].mean(axis=0)
        ex, ey = int(mid[0]), int(mid[1])
        if not (0 <= ex < W and 0 <= ey < H):
            continue
        g = int(tool_groups[ey, ex])
        if g == 0:
            continue
        by_dil_group.setdefault(g, []).append(i)

    for ids in by_dil_group.values():
        for i in ids[1:]:
            union(ids[0], i)

    # Collect groups
    final: dict[int, list[int]] = {}
    for i in range(n):
        final.setdefault(find(i), []).append(i)

    return list(final.values())


# =====================================================================
# Cyclic loop completion — bridges stubs by angle-sort around centroid
# =====================================================================

def _bridges_for_group(
    polylines: list[dict],
    indices: list[int],
    lookback_px: int = 20,
) -> list[dict]:
    """Construct the bridges for one tool's polylines via cyclic ordering.

    K = number of polylines in this group:
      * K=1 — connect the polyline's two endpoints (closes a single break).
      * K=2 — the two stubs need TWO bridges.  Pick the endpoint pairing
              with minimum total bridge length (avoids the "X" case).
      * K≥3 — sort polylines by angle around tool centroid and connect each
              consecutive pair at their closest endpoint pair.
    """
    K = len(indices)
    if K == 0:
        return []

    pls = [polylines[i] for i in indices]

    def make_bridge(p1: dict, end1: str, p2: dict, end2: str) -> dict | None:
        t1 = _polyline_tangent(p1["points"], end1, lookback_px)
        t2 = _polyline_tangent(p2["points"], end2, lookback_px)
        if t1 is None or t2 is None:
            return None
        return {
            "p_a":   p1["points"][0]  if end1 == "a" else p1["points"][-1],
            "p_b":   p2["points"][0]  if end2 == "a" else p2["points"][-1],
            "dir_a": t1,
            "dir_b": t2,
            "thick_a": p1[f"thickness_{end1}"],
            "thick_b": p2[f"thickness_{end2}"],
        }

    # ---- K = 1 -------------------------------------------------------
    if K == 1:
        b = make_bridge(pls[0], "a", pls[0], "b")
        return [b] if b else []

    # ---- K = 2 -------------------------------------------------------
    if K == 2:
        p1, p2 = pls
        a1, b1 = p1["points"][0], p1["points"][-1]
        a2, b2 = p2["points"][0], p2["points"][-1]
        d_aa = float(np.linalg.norm(a1 - a2))
        d_ab = float(np.linalg.norm(a1 - b2))
        d_ba = float(np.linalg.norm(b1 - a2))
        d_bb = float(np.linalg.norm(b1 - b2))
        # Two valid pairings; pick the lower-total-length one
        if d_aa + d_bb <= d_ab + d_ba:
            ends = [("a", "a"), ("b", "b")]
        else:
            ends = [("a", "b"), ("b", "a")]
        out: list[dict] = []
        for e1, e2 in ends:
            br = make_bridge(p1, e1, p2, e2)
            if br is not None:
                out.append(br)
        return out

    # ---- K ≥ 3: cyclic ordering by angle around tool centroid -------
    midpoints = np.array([pl["points"].mean(axis=0) for pl in pls])
    tool_centroid = midpoints.mean(axis=0)
    angles = np.arctan2(
        midpoints[:, 1] - tool_centroid[1],
        midpoints[:, 0] - tool_centroid[0],
    )
    order = np.argsort(angles)
    ordered = [pls[i] for i in order]

    bridges: list[dict] = []
    for i in range(K):
        cur = ordered[i]
        nxt = ordered[(i + 1) % K]
        ca, cb = cur["points"][0], cur["points"][-1]
        na, nb = nxt["points"][0], nxt["points"][-1]
        candidates = [
            (float(np.linalg.norm(ca - na)), "a", "a"),
            (float(np.linalg.norm(ca - nb)), "a", "b"),
            (float(np.linalg.norm(cb - na)), "b", "a"),
            (float(np.linalg.norm(cb - nb)), "b", "b"),
        ]
        candidates.sort(key=lambda t: t[0])
        _, e_cur, e_nxt = candidates[0]
        br = make_bridge(cur, e_cur, nxt, e_nxt)
        if br is not None:
            bridges.append(br)

    return bridges


def _render_bridge(canvas: np.ndarray, bridge: dict) -> None:
    """Draw a tapered cubic Hermite Bézier from p_a to p_b on *canvas*."""
    desc_a = {"pos": bridge["p_a"], "dir": bridge["dir_a"], "thick": bridge["thick_a"]}
    desc_b = {"pos": bridge["p_b"], "dir": bridge["dir_b"], "thick": bridge["thick_b"]}
    _render_tapered_bezier(canvas, desc_a, desc_b)


# =====================================================================
# Main entry — repair_orange_gaps (vector-domain cyclic completion)
# =====================================================================

def repair_orange_gaps(
    cleaned_bgr: np.ndarray,
    orange_mask: np.ndarray | None = None,
    dark_threshold: int = 170,
    min_branch_len: int = 10,
    lookback_px: int = 30,
    max_gap_px: int = 400,
    tool_group_radius: int = 10,
    smooth_kernel_px: int = 0,
    min_skel_component_px: int = 8,
    min_polyline_length: int = 20,
    alignment_target_tol_px: int = 20,
    alignment_max_search: int = 400,
    noise_min_area_px: int = 0,
    max_processing_dim: int | None = 3000,
    # ---- back-compat: previously used by older matching pipelines ----
    facing_threshold: float | None = None,
    obstacle_threshold: float | None = None,
    target_offset_tol_px: int | None = None,
    fallback_max_dist_px: int | None = None,
    fallback_facing_min: float | None = None,
    bridge_px: int | None = None,
    extra_passes: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vector-domain cyclic loop completion.

    Pipeline
    --------
    1. Threshold *cleaned_bgr* → trace binary.
    2. Compute medial axis (skimage) + distance map.
    3. Prune short branches; drop tiny isolated skeleton fragments.
    4. **Extract polylines**: one principal centerline per skeleton
       component (the geodesic diameter).  Closed-loop components (already
       complete outlines) are skipped.
    5. **Build alignment graph**: link polylines whose endpoint rays
       (cast along the tangent) land near another polyline's endpoint.
       This catches long internal gaps where dilation alone cannot merge.
    6. **Compute dilation tool groups**: connected components of the
       trace mask dilated by *tool_group_radius*.
    7. **Union-find** alignment + dilation: a polyline group emerges iff
       polylines are linked by EITHER mechanism.
    8. **Cyclic loop completion** per group: sort polylines by angle
       around the group centroid, bridge each consecutive pair at their
       closest endpoint pair.  K=1 → close the single break.  K=2 → two
       bridges in min-cost pairing.  K≥3 → cyclic neighbours.
    9. Render bridges as tapered cubic Hermite Béziers; OR with the
       original trace; return.

    Returns
    -------
    ``(repaired_mask, completion_mask, skeleton_mask)`` — uint8 binary
    arrays at the processing resolution.
    """
    # -----------------------------------------------------------------
    # Optional downscale
    # -----------------------------------------------------------------
    h0, w0 = cleaned_bgr.shape[:2]
    if max_processing_dim and max(h0, w0) > max_processing_dim:
        scale = max_processing_dim / max(h0, w0)
        new_w = max(1, int(round(w0 * scale)))
        new_h = max(1, int(round(h0 * scale)))
        cleaned_bgr = cv2.resize(
            cleaned_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA
        )

    # -----------------------------------------------------------------
    # 1. Threshold + (optional) noise filter + (optional) smoothing
    # -----------------------------------------------------------------
    gray = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2GRAY)
    trace_binary = ((gray < dark_threshold).astype(np.uint8)) * 255
    H, W = trace_binary.shape[:2]

    if not np.any(trace_binary > 0):
        empty = np.zeros((H, W), dtype=np.uint8)
        return trace_binary, empty, empty

    if noise_min_area_px > 0:
        trace_binary = _filter_small_components(
            trace_binary, min_area_px=noise_min_area_px
        )

    if smooth_kernel_px and smooth_kernel_px >= 3:
        sk = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (smooth_kernel_px, smooth_kernel_px)
        )
        trace_binary = cv2.morphologyEx(trace_binary, cv2.MORPH_CLOSE, sk)
        trace_binary = cv2.morphologyEx(trace_binary, cv2.MORPH_OPEN, sk)

    # -----------------------------------------------------------------
    # 2. Medial axis + distance map
    # -----------------------------------------------------------------
    t0 = time.time()
    skel_bool, distance_map = _skimage_medial_axis(
        trace_binary > 0, return_distance=True
    )
    skel_u8 = (skel_bool.astype(np.uint8)) * 255
    t_medial = time.time() - t0

    # -----------------------------------------------------------------
    # 3. Prune short branches + drop tiny components
    # -----------------------------------------------------------------
    skel_pruned = _prune_skeleton(skel_u8, min_branch_len=min_branch_len)

    if min_skel_component_px > 0:
        n_sk, sk_lbl, sk_stats, _ = cv2.connectedComponentsWithStats(
            (skel_pruned > 0).astype(np.uint8), connectivity=8
        )
        if n_sk > 1:
            keep = np.zeros(n_sk, dtype=bool)
            for c in range(1, n_sk):
                if sk_stats[c, cv2.CC_STAT_AREA] >= min_skel_component_px:
                    keep[c] = True
            skel_pruned = (np.where(keep[sk_lbl], 255, 0)).astype(np.uint8)

    # -----------------------------------------------------------------
    # 4. Extract polylines (one per skeleton component)
    # -----------------------------------------------------------------
    polylines = _extract_skeleton_polylines(
        skel_pruned, distance_map, min_polyline_length=min_polyline_length,
    )

    if not polylines:
        empty = np.zeros((H, W), dtype=np.uint8)
        return trace_binary, empty, skel_pruned

    # -----------------------------------------------------------------
    # 5. Alignment graph (long-gap-aware)
    # -----------------------------------------------------------------
    alignment_adj = _build_alignment_graph(
        polylines, trace_binary,
        max_search=alignment_max_search,
        target_tol_px=alignment_target_tol_px,
        lookback_px=lookback_px,
    )

    # -----------------------------------------------------------------
    # 6. Dilation-based tool groups
    # -----------------------------------------------------------------
    tool_groups = _compute_tool_groups(
        trace_binary, dilate_radius=tool_group_radius
    )

    # -----------------------------------------------------------------
    # 7. Final groups: union of tight-dilation and alignment edges
    # -----------------------------------------------------------------
    final_groups = _final_groups(
        polylines, alignment_adj, tool_groups,
    )

    # -----------------------------------------------------------------
    # 8. Cyclic loop completion per group
    # -----------------------------------------------------------------
    all_bridges: list[dict] = []
    for group_indices in final_groups:
        all_bridges.extend(
            _bridges_for_group(polylines, group_indices, lookback_px=lookback_px)
        )

    # -----------------------------------------------------------------
    # 9. Render bridges → completion mask → repaired mask
    # -----------------------------------------------------------------
    completion = np.zeros((H, W), dtype=np.uint8)
    for bridge in all_bridges:
        _render_bridge(completion, bridge)

    completion = cv2.bitwise_and(completion, cv2.bitwise_not(trace_binary))
    repaired = cv2.bitwise_or(trace_binary, completion)

    n_added = int(np.count_nonzero(completion))
    n_pl = len(polylines)
    n_groups = len(final_groups)
    n_bridges = len(all_bridges)
    sizes_distrib = [len(g) for g in final_groups]
    n_singletons = sum(1 for s in sizes_distrib if s == 1)

    print(
        f"  centerline repair: medial_axis {t_medial:.1f}s | "
        f"{n_pl} polylines | {n_groups} loop groups "
        f"({n_singletons} singletons) | "
        f"{n_bridges} bridges | {n_added:,} px completion"
    )

    return repaired, completion, skel_pruned


@dataclass
class VectorPath:
    """A single closed vector shape extracted from the raster."""

    points: np.ndarray          # (N, 2) float64 — simplified nodes, mask-px
    bezier_segments: list       # [(p1, cp1, cp2, p2), …], each (2,) float64
    area_px: float
    perimeter_px: float
    bbox_xywh: tuple[int, int, int, int]  # in mask pixels


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _rdp_simplify(contour: np.ndarray, epsilon: float) -> np.ndarray:
    """Ramer–Douglas–Peucker via cv2.approxPolyDP.

    Parameters
    ----------
    contour : (N, 1, 2) int32 as returned by findContours
    epsilon : RDP tolerance in pixels

    Returns
    -------
    (M, 2) float64 simplified points
    """
    approx = cv2.approxPolyDP(contour, epsilon, closed=True)
    return approx.reshape(-1, 2).astype(np.float64)


def _catmull_rom_to_bezier(pts: np.ndarray, tension: float = 1.0) -> list:
    """Convert a closed polyline into cubic Bézier segments.

    Uses the Catmull-Rom parameterisation: control points are derived from
    neighbouring nodes so the curve passes *through* every simplified vertex
    with C1 continuity.

    Parameters
    ----------
    pts : (N, 2) float64
    tension : scaling of tangent length; 1.0 = standard Catmull-Rom

    Returns
    -------
    List of (p1, cp1, cp2, p2) tuples, each element a (2,) float64 array.
    The list is length N (one segment per node), wrapping around to close the
    shape.
    """
    n = len(pts)
    if n < 2:
        return []

    segments: list[tuple] = []
    for i in range(n):
        p0 = pts[(i - 1) % n]
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        p3 = pts[(i + 2) % n]
        # Catmull-Rom tangent at p1 points from p0 → p2
        # Tangent at p2 points from p1 → p3
        cp1 = p1 + tension * (p2 - p0) / 6.0
        cp2 = p2 - tension * (p3 - p1) / 6.0
        segments.append((p1.copy(), cp1, cp2, p2.copy()))

    return segments


def _sample_bezier_segment(
    p1: np.ndarray,
    cp1: np.ndarray,
    cp2: np.ndarray,
    p2: np.ndarray,
    n_pts: int = 20,
) -> np.ndarray:
    """Sample a cubic Bézier at *n_pts* uniform parameter values.

    Returns (n_pts, 2) float64.
    """
    ts = np.linspace(0.0, 1.0, n_pts, endpoint=False)
    mt = 1.0 - ts
    pts = (
        mt[:, None] ** 3 * p1
        + 3 * mt[:, None] ** 2 * ts[:, None] * cp1
        + 3 * mt[:, None] * ts[:, None] ** 2 * cp2
        + ts[:, None] ** 3 * p2
    )
    return pts


def _sample_path(path: VectorPath, n_pts_per_seg: int = 20) -> np.ndarray:
    """Return a dense polyline approximation of the Bézier path.

    Returns (M, 2) float64 in mask-pixel space.
    """
    parts = []
    for seg in path.bezier_segments:
        parts.append(_sample_bezier_segment(*seg, n_pts=n_pts_per_seg))
    return np.vstack(parts) if parts else np.empty((0, 2), dtype=np.float64)


# ---------------------------------------------------------------------------
# Public extraction API
# ---------------------------------------------------------------------------

def _circularity(contour: np.ndarray) -> float:
    """Isoperimetric ratio: 1.0 = perfect circle, <<1 = ribbon/elongated."""
    area = cv2.contourArea(contour)
    perim = cv2.arcLength(contour, closed=True)
    if perim <= 0:
        return 0.0
    return float(4.0 * np.pi * area / (perim * perim))


def extract_vector_paths(
    trace_mask: np.ndarray,
    min_area_px: float = 200.0,
    rdp_epsilon: float = 3.0,
    tension: float = 1.0,
    gap_close_px: int = 17,
    max_gap_close_px: int = 51,
    circularity_min: float = 0.05,
) -> list[VectorPath]:
    """Extract smooth closed vector paths from a binary trace mask.

    Uses a two-pass gap-closing strategy to handle traces broken by grid
    crossings:

    1. **Global close** (``gap_close_px``) — connects most small fragments.
    2. **Circularity filter** — a proper closed loop has a much larger
       area-to-perimeter ratio than a ribbon.  Any contour whose
       ``4*pi*area/perimeter^2`` falls below *circularity_min* is still a
       ribbon (an open line whose boundary has been traced on both sides).
    3. **Local retry** — for each ribbon, progressively larger closing
       kernels are applied *only within that contour's bounding box*, so
       the large kernel never touches neighbouring traces.  This bridges
       gaps from wider grid lines without merging distinct tool shapes.

    Parameters
    ----------
    trace_mask : uint8 binary mask (non-zero = trace foreground)
    min_area_px : discard contours smaller than this area (mask pixels).
        Raise to suppress noise when using larger closing kernels.
    rdp_epsilon : RDP simplification tolerance in mask pixels.
        Larger values -> fewer nodes.  Good starting range: 2-6.
    tension : Catmull-Rom tension factor (1.0 = standard smooth).
    gap_close_px : global MORPH_CLOSE kernel size (mask pixels).
        At 1500px/350mm scale 1px ~ 0.23 mm, so the default k=17 bridges
        gaps up to ~2 mm (typical minor-grid-line width).
    max_gap_close_px : maximum kernel tried per-ribbon in the local retry.
        Default 51 (~6 mm) covers major grid crossings.  Set equal to
        gap_close_px to disable the local retry.
    circularity_min : contours below this isoperimetric ratio are treated
        as ribbons and sent to the local retry.  0.05 works well; lower
        values are more permissive.

    Returns
    -------
    List of :class:`VectorPath`, sorted largest -> smallest by area.
    """
    binary = (trace_mask > 0).astype(np.uint8) * 255
    H, W = binary.shape[:2]

    # ------------------------------------------------------------------
    # Pass 1 — global close to connect most fragments
    # ------------------------------------------------------------------
    if gap_close_px > 0:
        gk = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (gap_close_px, gap_close_px)
        )
        globally_closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, gk)
    else:
        globally_closed = binary.copy()

    raw_contours, _ = cv2.findContours(
        globally_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    # ------------------------------------------------------------------
    # Pass 2 — circularity filter + local retry for ribbons
    # ------------------------------------------------------------------
    accepted: list[np.ndarray] = []   # contour arrays in full-image coords
    n_ribbons_resolved = 0
    n_ribbons_dropped = 0

    for contour in raw_contours:
        area = float(cv2.contourArea(contour))
        if area < min_area_px:
            continue

        if _circularity(contour) >= circularity_min:
            accepted.append(contour)
            continue

        # --- ribbon: retry with progressively larger local close --------
        x, y, bw, bh = cv2.boundingRect(contour)
        resolved = False

        k_retry = gap_close_px + 8
        while k_retry <= max_gap_close_px:
            pad = k_retry // 2 + 2
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(W, x + bw + pad), min(H, y + bh + pad)

            sub = binary[y1:y2, x1:x2].copy()
            rk = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (k_retry, k_retry)
            )
            sub_closed = cv2.morphologyEx(sub, cv2.MORPH_CLOSE, rk)
            sub_contours, _ = cv2.findContours(
                sub_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )

            for sc in sub_contours:
                sc_area = float(cv2.contourArea(sc))
                if sc_area < min_area_px:
                    continue
                if _circularity(sc) >= circularity_min:
                    # Offset back to full-image coordinate space
                    accepted.append(sc + np.array([[[x1, y1]]], dtype=np.int32))
                    resolved = True

            if resolved:
                n_ribbons_resolved += 1
                break
            k_retry += 8

        if not resolved:
            n_ribbons_dropped += 1

    if n_ribbons_resolved or n_ribbons_dropped:
        print(
            f"  gap-close: {n_ribbons_resolved} ribbon(s) resolved via local retry, "
            f"{n_ribbons_dropped} dropped (gaps too large or traces too close)"
        )

    # ------------------------------------------------------------------
    # Build VectorPath objects from accepted contours
    # ------------------------------------------------------------------
    paths: list[VectorPath] = []
    for contour in accepted:
        area = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, closed=True))
        x, y, w, h = cv2.boundingRect(contour)

        pts = _rdp_simplify(contour, rdp_epsilon)
        if len(pts) < 2:
            continue

        segs = _catmull_rom_to_bezier(pts, tension=tension)

        paths.append(VectorPath(
            points=pts,
            bezier_segments=segs,
            area_px=area,
            perimeter_px=perimeter,
            bbox_xywh=(int(x), int(y), int(w), int(h)),
        ))

    paths.sort(key=lambda p: p.area_px, reverse=True)
    return paths


# ---------------------------------------------------------------------------
# SVG export
# ---------------------------------------------------------------------------

def _path_to_svg_d(
    path: VectorPath,
    scale_x: float,
    scale_y: float,
) -> str:
    """Render a VectorPath as an SVG ``d`` attribute string.

    Coordinates are scaled by *scale_x* / *scale_y* (mask pixels → mm).
    """
    segs = path.bezier_segments
    if not segs:
        return ""
    sx, sy = scale_x, scale_y
    start = segs[0][0]
    parts = [f"M {start[0]*sx:.4f},{start[1]*sy:.4f}"]
    for p1, cp1, cp2, p2 in segs:
        parts.append(
            f"C {cp1[0]*sx:.4f},{cp1[1]*sy:.4f} "
            f"{cp2[0]*sx:.4f},{cp2[1]*sy:.4f} "
            f"{p2[0]*sx:.4f},{p2[1]*sy:.4f}"
        )
    parts.append("Z")
    return " ".join(parts)


def write_svg(
    paths: list[VectorPath],
    output_path: Path,
    mask_shape: tuple[int, int],   # (H, W) in mask pixels
    design_width_mm: float,
    design_height_mm: float,
    stroke_colour: str = "#E63000",
    stroke_width_mm: float = 0.3,
    fill: str = "none",
) -> Path:
    """Write an SVG file with all vector paths.

    Parameters
    ----------
    paths : extracted vector paths (mask-pixel coordinate space)
    output_path : destination ``.svg`` file
    mask_shape : (H, W) of the trace mask (for pixel→mm scaling)
    design_width_mm / design_height_mm : physical dimensions of the design area
    stroke_colour : hex colour for path strokes
    stroke_width_mm : stroke width in mm
    fill : SVG fill attribute (default "none" = outlines only)

    Returns
    -------
    Resolved output path.
    """
    mask_h, mask_w = mask_shape
    scale_x = design_width_mm / mask_w
    scale_y = design_height_mm / mask_h

    svg = ET.Element(
        "svg",
        xmlns="http://www.w3.org/2000/svg",
        viewBox=f"0 0 {design_width_mm} {design_height_mm}",
        width=f"{design_width_mm}mm",
        height=f"{design_height_mm}mm",
    )
    # Metadata comment
    svg.append(ET.Comment(
        f" ArmourCore CDS vectorisation — {len(paths)} paths "
        f"| mask {mask_w}×{mask_h}px → {design_width_mm}×{design_height_mm}mm "
    ))

    g = ET.SubElement(
        svg, "g",
        id="traces",
        stroke=stroke_colour,
        **{"stroke-width": str(stroke_width_mm)},
        fill=fill,
        **{"stroke-linecap": "round", "stroke-linejoin": "round"},
    )

    for i, path in enumerate(paths):
        d = _path_to_svg_d(path, scale_x, scale_y)
        if not d:
            continue
        ET.SubElement(g, "path", d=d, id=f"trace_{i:04d}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tree = ET.ElementTree(svg)
    ET.indent(tree, space="  ")
    tree.write(str(output_path), encoding="unicode", xml_declaration=True)

    return output_path


# ---------------------------------------------------------------------------
# Overlay rendering (PNG)
# ---------------------------------------------------------------------------

def render_vector_overlay(
    base_bgr: np.ndarray,
    paths: list[VectorPath],
    mask_shape: tuple[int, int],
    colour_bgr: tuple[int, int, int] = (0, 48, 230),  # vivid red-orange
    thickness: int = 2,
    n_pts_per_seg: int = 20,
) -> np.ndarray:
    """Draw Bézier curves onto *base_bgr* for visual inspection.

    Paths live in mask-pixel space; they are scaled to *base_bgr* pixel space
    using the ratio of the two image sizes.

    Parameters
    ----------
    base_bgr : background image (any resolution)
    paths : vector paths in mask-pixel coordinates
    mask_shape : (H, W) of the trace mask used to extract the paths
    colour_bgr : BGR stroke colour
    thickness : stroke width in overlay pixels
    n_pts_per_seg : Bézier sampling density (more = smoother visual)
    """
    overlay = base_bgr.copy()
    h_img, w_img = overlay.shape[:2]
    mask_h, mask_w = mask_shape

    sx = w_img / mask_w
    sy = h_img / mask_h

    for path in paths:
        sampled = _sample_path(path, n_pts_per_seg)  # (M, 2) mask-px
        if len(sampled) < 2:
            continue
        # Scale to image space
        pts_img = sampled.copy()
        pts_img[:, 0] *= sx
        pts_img[:, 1] *= sy
        pts_int = pts_img.round().astype(np.int32)
        # Draw as closed polyline
        cv2.polylines(overlay, [pts_int], isClosed=True, color=colour_bgr,
                      thickness=thickness, lineType=cv2.LINE_AA)

    return overlay
