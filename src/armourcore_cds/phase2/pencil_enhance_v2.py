"""Phase 2 pencil-enhancement methods (v2 - expanded toolkit).

Multiple approaches so the lab can pick whichever works best for the
BLUE_PENCIL_FLAT_01 case.

Methods (all take + return BGR for pipeline compatibility):
  * black_hat_tophat
  * sauvola_adaptive
  * frangi_ridges            (best for line/stroke structures)
  * multi_scale_tophat
  * niblack_adaptive
  * gamma_then_clahe
  * combined_frangi_sauvola  (frangi-enhanced then sauvola-threshold)
"""
from __future__ import annotations

import cv2
import numpy as np

try:
    from skimage.filters import frangi as _sk_frangi, sauvola as _sk_sauvola
    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False


def _to_bgr_dark_on_white(gray: np.ndarray) -> np.ndarray:
    """Convert a single-channel image where HIGH = ink to a BGR image
    where LOW (dark) = ink so downstream gray<170 threshold works."""
    if gray.dtype != np.uint8:
        # rescale to 0-255
        g = gray.astype(np.float32)
        g = (g - g.min()) / max(g.max() - g.min(), 1e-6)
        gray = (g * 255).astype(np.uint8)
    inverted = 255 - gray  # high signal -> dark
    return cv2.cvtColor(inverted, cv2.COLOR_GRAY2BGR)


def black_hat_tophat(image_bgr: np.ndarray, kernel_px: int = 25) -> np.ndarray:
    """Black-hat morphological transform - highlights dark structures
    smaller than the kernel.  Great for pencil-on-paper since strokes
    are thin compared to paper background."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                  (kernel_px, kernel_px))
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k)
    # bh has HIGH values where pencil strokes are
    return _to_bgr_dark_on_white(bh)


def sauvola_adaptive(image_bgr: np.ndarray,
                    window_size: int = 25,
                    k: float = 0.15) -> np.ndarray:
    """Sauvola adaptive threshold.  Returns BGR (dark ink, white paper)."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ksize = (window_size, window_size)
    mean = cv2.boxFilter(gray, ddepth=cv2.CV_32F, ksize=ksize)
    sq_mean = cv2.boxFilter(gray * gray, ddepth=cv2.CV_32F, ksize=ksize)
    var = np.maximum(sq_mean - mean * mean, 0.0)
    std = np.sqrt(var)
    T = mean * (1 + k * (std / 128.0 - 1))
    binary = (gray < T).astype(np.uint8) * 255   # ink = 255
    out = np.full(image_bgr.shape, 255, dtype=np.uint8)
    out[binary > 0] = (40, 40, 40)
    return out


def niblack_adaptive(image_bgr: np.ndarray,
                    window_size: int = 25,
                    k: float = -0.2) -> np.ndarray:
    """Niblack adaptive threshold.  More sensitive than Sauvola."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ksize = (window_size, window_size)
    mean = cv2.boxFilter(gray, ddepth=cv2.CV_32F, ksize=ksize)
    sq_mean = cv2.boxFilter(gray * gray, ddepth=cv2.CV_32F, ksize=ksize)
    var = np.maximum(sq_mean - mean * mean, 0.0)
    std = np.sqrt(var)
    T = mean + k * std
    binary = (gray < T).astype(np.uint8) * 255
    out = np.full(image_bgr.shape, 255, dtype=np.uint8)
    out[binary > 0] = (40, 40, 40)
    return out


def frangi_ridges(image_bgr: np.ndarray,
                 scale_range: tuple = (2, 6),
                 scale_step: float = 2.0,
                 alpha: float = 0.5,
                 beta: float = 0.5) -> np.ndarray:
    """Frangi multi-scale ridge filter - excellent for stroke/vessel
    detection.  Pencil strokes have ridge-like cross-section so Frangi
    responds strongly to them regardless of absolute darkness.
    """
    if not _HAS_SKIMAGE:
        raise RuntimeError("scikit-image not available")
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # Frangi expects ridges to be DARK on light background by default
    # (black_ridges=True is the default; we want dark pencil on white paper)
    response = _sk_frangi(
        gray,
        sigmas=np.arange(scale_range[0], scale_range[1] + 1, scale_step),
        alpha=alpha, beta=beta, black_ridges=True,
    )
    # response is float [0,1] - higher = stronger ridge
    return _to_bgr_dark_on_white(response)


def multi_scale_tophat(image_bgr: np.ndarray) -> np.ndarray:
    """Black-hat at multiple scales, max-combined.  Catches strokes of
    different thicknesses."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    out = np.zeros_like(gray, dtype=np.uint8)
    for ks in (11, 21, 35):
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
        bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k)
        out = np.maximum(out, bh)
    return _to_bgr_dark_on_white(out)


def gamma_then_clahe(image_bgr: np.ndarray, gamma: float = 1.6) -> np.ndarray:
    """Gamma boost + CLAHE.  Gamma > 1 stretches dark range."""
    img = image_bgr.astype(np.float32) / 255.0
    img = np.power(img, gamma)
    img8 = (img * 255).astype(np.uint8)
    lab = cv2.cvtColor(img8, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(12, 12))
    L = clahe.apply(L)
    return cv2.cvtColor(cv2.merge([L, a, b]), cv2.COLOR_LAB2BGR)


def combined_frangi_sauvola(image_bgr: np.ndarray) -> np.ndarray:
    """Frangi enhancement -> Sauvola binarisation.  Best of both worlds:
    Frangi pulls pencil strokes out of paper noise, Sauvola makes a
    crisp binary with locally-adaptive threshold."""
    enhanced = frangi_ridges(image_bgr)
    return sauvola_adaptive(enhanced, window_size=31, k=0.1)
