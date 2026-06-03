"""Simplified Phase 2 cleaner for scanned controlled-input sheets.

Why a new module?
=================
The production pipeline (v14 adaptive + grid_strip + line_strip +
dark_border + aggressive variants) was built to survive PHONE PHOTOS
with coloured lighting cast, dark camera vignettes, and lens
distortion.  None of that applies to a scanner.

For scan inputs we have:
* Pure-white paper background
* Known template colours (cyan border, peach/orange grid, red markers,
  optionally orange dots)
* Tool ink that is BLACK/DARK-GRAY and ACHROMATIC

So we can clean by colour alone in a single HSV pass:
  * cyan band -> erase to white
  * orange/peach band -> erase to white
  * red band -> erase to white
  * what remains: the achromatic tool ink + (sometimes) faint pencil

This module replaces ALL of v14_adaptive + grid_strip_all + line_strip
+ grid_strip_aggressive + strip_dark_border for the scan workflow.
No Hough, no CLAHE, no per-route logic.  ~0.3s instead of ~5s.

Two passes
==========
  ``clean_scan_pen`` - direct colour strip, leaves any dark pixel intact
  ``clean_scan_pencil`` - colour strip THEN Sauvola adaptive binarise
                          (the only enhancement needed for faint pencil)
"""
from __future__ import annotations

import cv2
import numpy as np

from armourcore_cds.phase2.pencil_enhance_v2 import sauvola_adaptive


def _template_colour_mask(image_bgr: np.ndarray,
                         strip_blue_notes: bool = False) -> np.ndarray:
    """Union mask of TEMPLATE-PRINTED colours only.

    Catches:
      * cyan border (~#00CAEC) - tight hue band, high sat required
      * peach grid (~#FFEBCC, low-sat warm)
      * orange grid (~#FFC466) / orange dots (~#FF9C00) - chromatic only
      * red markers (~#FF0033)

    Does NOT touch black/dark pen ink.  Saturation thresholds tuned so
    scanner anti-aliasing edges of black strokes (which carry slight
    hue) stay out of the mask.

    ``strip_blue_notes`` flag is kept for API back-compat but is
    OFF by default and the underlying mask call is gated separately
    in the CLI (pen_dense route only).
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    # CYAN BORDER: very tight band + high saturation requirement so
    # scanner-edge-tinted black pen never lands in here.
    cyan = ((H >= 82) & (H <= 96) & (S >= 100) & (V >= 100))

    # WARM family (peach, orange, dots) - chromatic only, high value.
    warm = ((H >= 0) & (H <= 30) & (S >= 50) & (V >= 130))

    # RED markers - strong sat + high value
    red = (((H <= 10) | (H >= 170)) & (S >= 120) & (V >= 80))

    mask = cyan | warm | red

    if strip_blue_notes:
        mask = mask | _blue_notes_mask(hsv)

    return mask.astype(np.uint8) * 255


def _blue_notes_mask(hsv: np.ndarray) -> np.ndarray:
    """Detect blue-pen scribbles for removal.

    Band catches royal blue (~110), navy (~115), light blue (~100),
    purple-blue (~130).  Cyan border (~88) excluded by lower bound.

    ``S >= 60`` ensures the pixel is GENUINELY chromatic.  Black tool
    ink scanned with slight hue noise typically sits at S < 30, so
    this floor protects black ink from being misclassified as blue.
    """
    H, S, V = cv2.split(hsv)
    return ((H >= 99) & (H <= 145) & (S >= 60) & (V >= 40))


def strip_blue_notes(image_bgr: np.ndarray,
                    protect_dark_below: int = 20) -> np.ndarray:
    """Dedicated Phase-2 step: erase any blue-pen scribbles.

    Place AFTER the rectifier and BEFORE the rest of Phase 2 cleaning
    so subsequent grid/text steps don't have to deal with blue noise.

    ``protect_dark_below``: gray below this is too dark to be blue pen
    (probably very dark navy that's nearly black - or genuine black
    tool ink with a hue noise blip).  Keep it.  Default 20 = only
    truly-black pixels protected.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blue = _blue_notes_mask(hsv)
    kill = blue & (gray > protect_dark_below)
    out = image_bgr.copy()
    out[kill] = (255, 255, 255)
    return out


def clean_scan_pen(image_bgr: np.ndarray,
                  protect_dark_below: int = 140,
                  core_gray_max: int = 95,
                  core_sat_max: int = 60,
                  chroma_sat_min: int = 40,
                  chroma_val_min: int = 90,
                  warm_sat_min: int = 8,
                  red_sat_min: int = 18,
                  sat_boost: float = 1.8,
                  protect_grow_px: int = 5) -> np.ndarray:
    """Erase everything CHROMATIC, leaving achromatic tool ink intact.

    Why this approach (replaces the old "remove only specific template
    hues" logic)
    ============================================================
    On the GLOBAL controlled scan the tool tracing is ALWAYS achromatic
    (black/dark-gray pen, saturation < ~30).  Everything else that is
    coloured is either TEMPLATE PRINT (gold dots, faded spacing dots,
    red boundary line, black-or-coloured legacy text) or a HAND-WRITTEN
    NOTE (dimension labels, arrows - drawn in red now, orange later).
    None of those belong in the vector output.

    So instead of chasing individual template hues we simply keep
    achromatic dark ink and erase any chromatic pixel.  This is robust
    to new template colours AND to whatever colour the staff use for
    notes - it never needs retuning.

    Pen-quality protection (the regression we fixed before)
    -------------------------------------------------------
    Scanner anti-aliasing tints the EDGES of black strokes a faint
    blue/cyan, so a naive "remove all chromatic" would nibble the
    stroke edges and break continuity.  We guard against that by
    building a PROTECTION ZONE = the genuine black-pen core
    (``gray < core_gray_max`` AND ``sat < core_sat_max``) dilated by
    ``protect_grow_px``.  Any chromatic pixel within that zone (i.e. a
    tinted edge of a real stroke) is kept; chromatic pixels far from
    any black core (red notes, gold dots, boundary line) are erased.

    ``protect_dark_below`` is retained for API back-compat: any pixel
    darker than it that is ALSO achromatic is unconditionally kept.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    H, S_raw, V = cv2.split(hsv)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # SATURATION PRE-BOOST: faint template print (30%-opacity gold spacing
    # dots, light orange dots, anti-aliased red text edges) sits at very low
    # raw saturation (S~15-35) and was sneaking past the chromatic mask.
    # Multiplying S by ~1.8x widens the gap between paper noise (S~3-8) and
    # any printed/handwritten colour (S~25+ once boosted to >45), so a single
    # global saturation floor cleanly separates them.  The achromatic black-
    # pen core (S~6) stays achromatic after boost (~11) and is protected
    # anyway by the dark-pen guard below.
    if sat_boost and sat_boost != 1.0:
        S = np.clip(S_raw.astype(np.int32) * sat_boost, 0, 255).astype(np.uint8)
    else:
        S = S_raw

    # Genuine black-pen core: dark AND achromatic (excludes dark-but-red
    # annotation strokes, which carry high saturation).  Use RAW saturation
    # here so we don't accidentally promote ink edges out of "core".
    core = (gray < core_gray_max) & (S_raw < core_sat_max)
    core_u8 = core.astype(np.uint8) * 255
    if protect_grow_px > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (protect_grow_px * 2 + 1, protect_grow_px * 2 + 1))
        core_u8 = cv2.dilate(core_u8, k)
    protect = (core_u8 > 0) | ((gray <= protect_dark_below) & (S_raw < core_sat_max))

    # Chromatic removal is HUE-BAND AWARE so the faint template colours
    # (light gold dots, 30%-opacity spacing dots) and the red boundary
    # line + its red text get caught WITHOUT touching the achromatic tool
    # ink.  Measured on a real GLOBAL scan:
    #   * black tool ink ............ S median ~6   (very achromatic)
    #   * gold / orange dots ........ S median ~34  (faint, was missed by
    #                                 the old global sat floor of 55)
    #   * red boundary line + text .. S median ~93  (vivid)
    # So we drop the saturation floor for the WARM (orange/gold) and RED
    # bands well below the dots' level but still far above ink's ~6, and
    # keep a higher floor for any other stray colour.
    warm = ((H <= 35) & (S >= warm_sat_min) & (V >= chroma_val_min))
    red = (((H <= 12) | (H >= 165)) & (S >= red_sat_min) & (V >= 60))
    other = (S >= chroma_sat_min) & (V >= chroma_val_min)
    chromatic = warm | red | other

    kill = chromatic & (~protect)
    out = image_bgr.copy()
    out[kill] = (255, 255, 255)
    return out


def strip_outer_border(image_bgr: np.ndarray, border_px: int) -> np.ndarray:
    """Whiten a frame ``border_px`` wide around the image edge.

    After rectification a few pixels of the printed BLACK page border can
    bleed in around the very edge of the warped canvas.  Those dark pixels
    are achromatic, so the colour cleaner keeps them, and Phase 3 then
    traces them as a spurious rectangle hugging the sheet edge.  Since the
    calibrated design area never legitimately touches the extreme edge,
    we simply blank a thin frame.  ``border_px`` is computed by the caller
    from px_per_mm (e.g. 2mm worth).
    """
    if border_px <= 0:
        return image_bgr
    out = image_bgr.copy()
    b = border_px
    out[:b, :] = (255, 255, 255)
    out[-b:, :] = (255, 255, 255)
    out[:, :b] = (255, 255, 255)
    out[:, -b:] = (255, 255, 255)
    return out


def clean_scan_pencil(image_bgr: np.ndarray,
                     sauvola_window: int = 25,
                     sauvola_k: float = 0.15,
                     post_dilate_px: int = 1,
                     contrast_boost: bool = False) -> np.ndarray:
    """Erase template colours then Sauvola-binarise to darken pencil.

    ``contrast_boost`` (default True): before Sauvola, apply a CLAHE
    luminance boost so faint pencil strokes become more separable
    from paper.  Helps V3-style scans where pencil came out very light.

    ``post_dilate_px`` (default 1): after Sauvola, dilate ink pixels by
    1px to bridge sub-stroke skip-marks so faint outlines become
    continuous closed loops for Phase 3.
    """
    # 1) erase template colours
    stripped = clean_scan_pen(image_bgr, protect_dark_below=110)

    # 2) contrast boost via CLAHE on the L channel - makes faint pencil
    # stand out more before binarisation
    if contrast_boost:
        lab = cv2.cvtColor(stripped, cv2.COLOR_BGR2LAB)
        L, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(12, 12))
        L = clahe.apply(L)
        stripped = cv2.cvtColor(cv2.merge([L, a, b]), cv2.COLOR_LAB2BGR)

    # 3) Sauvola adaptive binarise
    binary_bgr = sauvola_adaptive(stripped, window_size=sauvola_window,
                                 k=sauvola_k)

    # 4) Small dilation: bridge skip-marks between adjacent dark pixels
    if post_dilate_px > 0:
        gray = cv2.cvtColor(binary_bgr, cv2.COLOR_BGR2GRAY)
        ink = (gray < 170).astype(np.uint8) * 255
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (post_dilate_px * 2 + 1, post_dilate_px * 2 + 1))
        ink = cv2.dilate(ink, k)
        out = np.full(binary_bgr.shape, 255, dtype=np.uint8)
        out[ink > 0] = (40, 40, 40)
        return out
    return binary_bgr
