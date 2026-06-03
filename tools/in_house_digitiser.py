"""In-House CDS Digitiser - GUI for the controlled CDS tracing workflow.

Purpose-built for the GLOBAL Calibrated Design Sheet scan workflow:
printed colour template + red corner markers + hand-traced tool outlines.

Workflow (mirrors the on-screen STEP labels):
    STEP 1  Select Image or PDF   (PDF auto-converted to PNG at 300dpi)
    STEP 2  Orient CDS            (CW / CCW until markers sit in corners)
    STEP 3  Select CDS Size       (Large / X-Large / Custom inline)
    STEP 4  Select Output Folder
    VECTORISE  ->  preview tiles populate; SVG written to the folder

The pipeline always runs in its most sensitive (lightest-pencil) mode,
so there is no pen/pencil choice to make.

Preview layout (static, no scroll bar)
    +---------------------------+----------+----------+
    |                           | corner   | corner   |
    |  Imported CDS Template    |   TL     |   TR     |
    |       (big)               |----------+----------|
    |                           | corner   | corner   |
    |                           |   BL     |   BR     |
    +---------+---------+---------+--------+----------+
    | Rectified | Cleaned | Vector boxes | Vectors    |
    +-----------+---------+--------------+------------+
Grey placeholders sit in every slot until the run populates them.
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path
from tkinter import (Tk, Toplevel, Frame, Button, Label, Entry, Canvas,
                     StringVar, filedialog, messagebox)
from tkinter import font as tkfont
from tkinter import ttk

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

import cv2
import numpy as np
from PIL import Image, ImageTk

from vectorise_cli import run_pipeline
import armourcore_cds.phase1.marker_rectify_scan as _p1

# ---------------------------------------------------------------------------
# Brand theme
# ---------------------------------------------------------------------------
BG = "#3F3F3F"          # window / panel background
BG_CARD = "#4A4A4A"     # slightly lighter card background
BG_PLACE = "#5A5A5A"    # grey placeholder rectangles
FG = "#FFFFFF"          # text + borders
ACCENT = "#F1B01D"      # ArmourCore gold accent
ACCENT_DK = "#C8920F"   # pressed accent
MUTED = "#B9B9B9"       # secondary text
DISABLED = "#6A6A6A"

# Look in a couple of likely places so the operator can drop a logo at
# tools/assets/ OR at the repo root.
_LOGO_CANDIDATES = [
    REPO / "tools" / "assets" / "ArmourCore_Logo_Small.png",
    REPO / "tools" / "assets" / "armourcore_logo.png",
    REPO / "tools" / "assets" / "armourcore_badge.png",
    REPO / "assets" / "armourcore_logo.png",
    REPO / "assets" / "armourcore_badge.png",
    REPO / "armourcore_logo.png",
    REPO / "armourcore_badge.png",
]

# Preset GLOBAL CDS calibrated design areas (W x H mm, inside the markers).
PRESET_LARGE = (600.0, 500.0)
PRESET_XLARGE = (900.0, 500.0)

# Preview tile sizes (image is scaled to fit inside; tile footprint is
# fixed so the layout never reflows).  Numbers chosen so the whole right
# column fits in a 1280-wide window beside the 360-wide controls.
BIG_W, BIG_H = 480, 330         # imported template
SMALL_W, SMALL_H = 160, 115     # corner ROIs (2x2 grid)
STAGE_W, STAGE_H = 320, 220     # rectified / cleaned (row 2)
RESULT_W, RESULT_H = 320, 220   # vector recognition / vectors (row 3, zoom-cropped)


class InHouseDigitiser:
    def __init__(self, root: Tk):
        self.root = root
        self.root.title("In-House CDS Digitiser")
        self.root.geometry("1360x980")
        self.root.configure(bg=BG)
        self.root.minsize(1240, 880)

        # ---- fonts: match the ArmourCore website (Montserrat / Open Sans),
        # falling back to the closest installed system face ----
        fams = set(tkfont.families())

        def pick(prefs):
            for f in prefs:
                if f in fams:
                    return f
            return prefs[-1]

        head_face = pick(["Montserrat", "Montserrat SemiBold",
                          "Segoe UI Semibold", "Segoe UI", "Arial"])
        base_face = pick(["Open Sans", "Segoe UI", "Arial"])
        self.f_logo = tkfont.Font(family=head_face, size=20, weight="bold")
        self.f_h1 = tkfont.Font(family=head_face, size=15, weight="bold")
        self.f_step = tkfont.Font(family=head_face, size=11, weight="bold")
        self.f_btn = tkfont.Font(family=head_face, size=10, weight="bold")
        self.f_huge = tkfont.Font(family=head_face, size=28, weight="bold")
        self.f_base = tkfont.Font(family=base_face, size=10)
        self.f_small = tkfont.Font(family=base_face, size=9)

        # ---- state ----
        self.input_path: Path | None = None
        self.original_image_bgr: np.ndarray | None = None
        self.oriented_image_bgr: np.ndarray | None = None  # the thing we run
        self.last_out_dir: Path | None = None
        self.paper_w, self.paper_h = PRESET_LARGE
        self.paper_label = "Large CDS (600 x 500)"
        self.output_root: Path = REPO / "data/outputs/InhouseProduction"
        self._thumbs: list[ImageTk.PhotoImage] = []  # keep refs from GC

        # ttk progressbar styling
        style = ttk.Style()
        try:
            style.theme_use("default")
        except Exception:
            pass
        style.configure("AC.Horizontal.TProgressbar",
                        troughcolor=BG_CARD, background=ACCENT,
                        bordercolor=BG_CARD, lightcolor=ACCENT,
                        darkcolor=ACCENT)

        # references to preview-tile Label widgets so we can fill them
        # in later without rebuilding the layout
        self.tile_imported: Label | None = None
        self.tile_corners: dict[str, Label] = {}
        self.tile_stage: dict[str, Label] = {}

        self._build_layout()
        self._refresh_size_buttons()
        self._set_run_state(ready=False)

    # ===================================================================
    # Layout
    # ===================================================================
    def _build_layout(self):
        # ---- header bar with badge ----
        header = Frame(self.root, bg=BG)
        header.pack(side="top", fill="x", padx=18, pady=(14, 6))
        self._place_logo(header)
        Label(header, text="In-House CDS Digitiser", bg=BG, fg=FG,
              font=self.f_logo).pack(side="left", padx=(14, 0))
        Label(header, text="GLOBAL calibrated-design-sheet workflow",
              bg=BG, fg=ACCENT, font=self.f_small).pack(side="left",
                                                        padx=(14, 0), pady=(8, 0))

        body = Frame(self.root, bg=BG)
        body.pack(side="top", fill="both", expand=True)

        # ---- left controls (narrower so preview can extend further left) ----
        left = Frame(body, bg=BG, width=360)
        left.pack(side="left", fill="y", padx=(14, 4), pady=8)
        left.pack_propagate(False)
        self._build_controls(left)

        # ---- right STATIC preview grid ----
        right = Frame(body, bg=BG)
        right.pack(side="left", fill="both", expand=True, padx=(4, 14), pady=8)
        Label(right, text="PREVIEW", bg=BG, fg=FG,
              font=self.f_h1).pack(anchor="w", pady=(0, 6))
        self._build_preview_grid(right)

    def _place_logo(self, parent):
        """Top-left badge.  Falls back to a gold ARMOURCORE wordmark if no
        image file has been dropped into tools/assets/."""
        for cand in _LOGO_CANDIDATES:
            if cand.exists():
                try:
                    im = Image.open(cand).convert("RGBA")
                    h = 52
                    w = int(im.width * (h / im.height))
                    im = im.resize((w, h), Image.LANCZOS)
                    self._logo_img = ImageTk.PhotoImage(im)
                    Label(parent, image=self._logo_img,
                          bg=BG).pack(side="left")
                    return
                except Exception:
                    continue
        # fallback wordmark
        Label(parent, text="ARMOURCORE", bg=BG, fg=ACCENT,
              font=self.f_logo).pack(side="left")

    # ---- themed widget helpers ----
    def _step_label(self, parent, text):
        Label(parent, text=text, bg=BG, fg=ACCENT,
              font=self.f_step).pack(anchor="w", pady=(14, 4))

    def _btn(self, parent, text, command, primary=False, font=None, **kw):
        bg = ACCENT if primary else BG_CARD
        fg = "#222222" if primary else FG
        b = Button(parent, text=text, command=command,
                   bg=bg, fg=fg, activebackground=ACCENT_DK,
                   activeforeground="#222222",
                   font=font or self.f_btn,
                   relief="flat", bd=0, highlightthickness=1,
                   highlightbackground=FG, highlightcolor=FG,
                   cursor="hand2", padx=10, pady=6, **kw)
        return b

    def _build_controls(self, left):
        # STEP 1
        self._step_label(left, "STEP 1  |  Select Image or PDF")
        self._btn(left, "Open Image or PDF…", self.pick_input,
                  width=26).pack(anchor="w")
        self.input_label = Label(left, text="No file selected", bg=BG,
                                 fg=MUTED, font=self.f_small,
                                 wraplength=380, justify="left")
        self.input_label.pack(anchor="w", pady=(4, 0))

        # STEP 2 - just CW / CCW, BIG, symbol only
        self._step_label(left, "STEP 2  |  Orient CDS")
        Label(left, text="Rotate until the red markers sit in the "
                         "four corners.", bg=BG, fg=MUTED,
              font=self.f_small, wraplength=380,
              justify="left").pack(anchor="w")
        orient_row = Frame(left, bg=BG)
        orient_row.pack(anchor="w", pady=(6, 0))
        self._btn(orient_row, "↺", lambda: self.transform("ccw"),
                  font=self.f_huge, width=3).pack(side="left", padx=(0, 12))
        self._btn(orient_row, "↻", lambda: self.transform("cw"),
                  font=self.f_huge, width=3).pack(side="left")

        # STEP 3 - size buttons + INLINE custom boxes (no popup)
        self._step_label(left, "STEP 3  |  Select CDS Size")
        size_row = Frame(left, bg=BG); size_row.pack(anchor="w")
        self.btn_large = self._btn(
            size_row, "Large CDS\n(600 x 500)",
            lambda: self.set_size(*PRESET_LARGE, "Large CDS (600 x 500)"),
            width=14)
        self.btn_large.pack(side="left", padx=(0, 6))
        self.btn_xlarge = self._btn(
            size_row, "X-Large CDS\n(900 x 500)",
            lambda: self.set_size(*PRESET_XLARGE, "X-Large CDS (900 x 500)"),
            width=14)
        self.btn_xlarge.pack(side="left", padx=6)

        # Inline Custom W/H entry boxes - typing in either applies the
        # custom size immediately.  No popup window.
        custom_row = Frame(left, bg=BG); custom_row.pack(anchor="w", pady=(8, 0))
        Label(custom_row, text="Custom (mm):", bg=BG, fg=FG,
              font=self.f_small).pack(side="left")
        self.custom_w = StringVar()
        self.custom_h = StringVar()
        self._cw_entry = Entry(custom_row, textvariable=self.custom_w, width=6,
                               bg=BG_CARD, fg=FG, insertbackground=FG,
                               relief="flat", highlightthickness=1,
                               highlightbackground=FG, font=self.f_base)
        self._cw_entry.pack(side="left", padx=(6, 2))
        Label(custom_row, text="W ×", bg=BG, fg=MUTED,
              font=self.f_small).pack(side="left")
        self._ch_entry = Entry(custom_row, textvariable=self.custom_h, width=6,
                               bg=BG_CARD, fg=FG, insertbackground=FG,
                               relief="flat", highlightthickness=1,
                               highlightbackground=FG, font=self.f_base)
        self._ch_entry.pack(side="left", padx=(2, 2))
        Label(custom_row, text="H", bg=BG, fg=MUTED,
              font=self.f_small).pack(side="left")
        # Apply custom size whenever the user finishes editing a field.
        for w in (self._cw_entry, self._ch_entry):
            w.bind("<FocusOut>", lambda e: self._apply_custom_if_valid())
            w.bind("<Return>", lambda e: self._apply_custom_if_valid())

        self.size_label = Label(left, text="", bg=BG, fg=FG, font=self.f_small)
        self.size_label.pack(anchor="w", pady=(6, 0))

        # STEP 4
        self._step_label(left, "STEP 4  |  Select Output Folder")
        self._btn(left, "Choose folder…", self.pick_output_folder,
                  width=26).pack(anchor="w")
        self.output_label = Label(left, text=str(self.output_root), bg=BG,
                                  fg=MUTED, font=self.f_small,
                                  wraplength=380, justify="left")
        self.output_label.pack(anchor="w", pady=(4, 0))

        # ---- VECTORISE ----
        self.run_btn = self._btn(left, "VECTORISE", self.run_pipeline_threaded,
                                 primary=True, width=26)
        self.run_btn.config(font=self.f_h1, pady=10)
        self.run_btn.pack(anchor="w", pady=(18, 4))

        self.run_note = Label(left, text="", bg=BG, fg=ACCENT,
                              font=self.f_small, wraplength=380,
                              justify="left")
        self.run_note.pack(anchor="w")

        self.repeat_btn = self._btn(left, "↻ Re-vectorise (same CDS)",
                                    self.run_pipeline_threaded, width=26)
        # shown only after a successful run
        self.progress = ttk.Progressbar(
            left, mode="indeterminate", length=300,
            style="AC.Horizontal.TProgressbar")

        self.status = Label(left, text="Ready – open a CDS to start.",
                            bg=BG, fg=MUTED, font=self.f_small,
                            wraplength=380, justify="left")
        self.status.pack(anchor="w", pady=(10, 4))

        self.open_folder_btn = self._btn(left, "Open output folder",
                                         self.open_folder, width=26)

    # ===================================================================
    # Static preview grid
    # ===================================================================
    def _build_preview_grid(self, parent):
        wrap = Frame(parent, bg=BG, highlightthickness=1,
                     highlightbackground=FG)
        wrap.pack(fill="both", expand=True)

        # ROW 1: big imported tile (left) + 2x2 corner ROI grid (right)
        row1 = Frame(wrap, bg=BG); row1.pack(fill="x", padx=8, pady=8)
        self.tile_imported = self._make_tile(row1, "Imported CDS Template",
                                             BIG_W, BIG_H, accent=True)
        self.tile_imported.master.pack(side="left", padx=(0, 10))

        # corner 2x2 grid (all pack, no .grid() — avoids mixing managers)
        corner_col = Frame(row1, bg=BG); corner_col.pack(side="left", anchor="n")
        Label(corner_col, text="Corner Detection", bg=BG, fg=ACCENT,
              font=self.f_step).pack(anchor="w", pady=(0, 4))
        for row_tags in (("TL", "TR"), ("BL", "BR")):
            row_f = Frame(corner_col, bg=BG)
            row_f.pack(anchor="w")
            for tag in row_tags:
                tile = self._make_tile(row_f, tag, SMALL_W, SMALL_H)
                tile.master.pack(side="left", padx=3, pady=3)
                self.tile_corners[tag] = tile

        # ROW 2: rectified | cleaned (intermediate stages)
        row2 = Frame(wrap, bg=BG); row2.pack(fill="x", padx=8, pady=(2, 6))
        for name, title in [
            ("rectified", "Rectified"),
            ("cleaned", "Cleaned"),
        ]:
            tile = self._make_tile(row2, title, STAGE_W, STAGE_H)
            tile.master.pack(side="left", padx=4)
            self.tile_stage[name] = tile

        # ROW 3: vector recognition | vectors (RESULTS - bigger, zoom-cropped)
        row3 = Frame(wrap, bg=BG); row3.pack(fill="x", padx=8, pady=(2, 10))
        for name, title in [
            ("vector_boxes", "Vector Recognition"),
            ("overlay", "Vectors"),
        ]:
            tile = self._make_tile(row3, title, RESULT_W, RESULT_H, accent=True)
            tile.master.pack(side="left", padx=4)
            self.tile_stage[name] = tile

    def _make_tile(self, parent, title, w, h, accent=False):
        """Build a labelled tile with a fixed-size Canvas placeholder.
        Returns the Canvas so we can draw images later via
        canvas.create_image(...).  Canvas is more reliable than Label for
        pixel-accurate placement and explicit sizing."""
        cell = Frame(parent, bg=BG_CARD, highlightthickness=1,
                     highlightbackground=FG)
        Label(cell, text=title, bg=BG_CARD,
              fg=ACCENT if accent else FG,
              font=self.f_step).pack(fill="x", padx=6, pady=(4, 2))
        cv_widget = Canvas(cell, width=w, height=h, bg=BG_PLACE,
                           highlightthickness=0, bd=0)
        cv_widget.pack(padx=6, pady=(0, 6))
        # tag the canvas so _set_tile knows its target footprint
        cv_widget._tile_size = (w, h)
        # paint a default "(placeholder)" caption at centre
        cv_widget.create_text(w // 2, h // 2, text="(placeholder)",
                              fill=MUTED, font=self.f_small,
                              tags=("placeholder",))
        return cv_widget

    def _set_tile(self, tile: Canvas, img_bgr: np.ndarray | None):
        """Replace tile contents with `img_bgr` scaled-to-fit, or revert
        to the grey placeholder when img_bgr is None."""
        w, h = getattr(tile, "_tile_size", (STAGE_W, STAGE_H))
        tile.delete("all")
        if img_bgr is None:
            tile.configure(bg=BG_PLACE)
            tile.create_text(w // 2, h // 2, text="(placeholder)",
                             fill=MUTED, font=self.f_small)
            return
        ih, iw = img_bgr.shape[:2]
        s = min(w / iw, h / ih)
        nw, nh = max(1, int(iw * s)), max(1, int(ih * s))
        small = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self._thumbs.append(photo)   # keep ref from GC
        tile.configure(bg=BG_CARD)
        tile.create_image(w // 2, h // 2, image=photo, anchor="center")

    def _clear_all_tiles(self):
        """Reset to placeholder state, keeping the imported tile if loaded."""
        self._thumbs.clear()
        for t in self.tile_corners.values():
            self._set_tile(t, None)
        for t in self.tile_stage.values():
            self._set_tile(t, None)

    def _show_imported_only(self):
        self._clear_all_tiles()
        if self.tile_imported is not None:
            self._set_tile(self.tile_imported, self.oriented_image_bgr)

    def _show_all_stages(self, out_dir: Path):
        """Final pass after the run: re-populate every tile from disk in
        case streaming notifications missed any (defensive)."""
        self._thumbs.clear()
        self._set_tile(self.tile_imported, self.oriented_image_bgr)
        for tag in ("TL", "TR", "BR", "BL"):
            self._stream_tile(f"corner_{tag}",
                              out_dir / f"_corner_{tag}.png")
        for stage, file in [
            ("rectified", "rectified.png"),
            ("cleaned", "cleaned.png"),
            ("vector_boxes", "vector_boxes.png"),
            ("overlay", "overlay.png"),
        ]:
            self._stream_tile(stage, out_dir / file)

    # ---- streaming preview from the pipeline thread ----
    def _stream_tile(self, stage: str, path: Path):
        """Load `path` and drop it into the matching tile.  For result
        tiles (vector_boxes, overlay) crop to the non-white content bbox
        so the user can actually see what was detected."""
        if not Path(path).exists():
            return
        img = cv2.imread(str(path))
        if img is None:
            return
        if stage in ("vector_boxes", "overlay"):
            img = self._crop_to_content(img)
        if stage.startswith("corner_"):
            tag = stage.split("_", 1)[1]
            if tag in self.tile_corners:
                self._set_tile(self.tile_corners[tag], img)
        elif stage in self.tile_stage:
            self._set_tile(self.tile_stage[stage], img)

    def _crop_to_content(self, img_bgr: np.ndarray,
                         margin_frac: float = 0.04) -> np.ndarray:
        """Crop `img_bgr` to the bbox of its non-near-white pixels +
        a small margin.  Lets us zoom in on the actual vectors instead
        of showing 90% empty white canvas."""
        try:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            content = gray < 230
            if not content.any():
                return img_bgr
            ys, xs = np.where(content)
            y0, y1 = ys.min(), ys.max()
            x0, x1 = xs.min(), xs.max()
            H, W = gray.shape
            my = int((y1 - y0 + 1) * margin_frac) + 6
            mx = int((x1 - x0 + 1) * margin_frac) + 6
            y0 = max(0, y0 - my); y1 = min(H - 1, y1 + my)
            x0 = max(0, x0 - mx); x1 = min(W - 1, x1 + mx)
            return img_bgr[y0:y1 + 1, x0:x1 + 1]
        except Exception:
            return img_bgr

    def _on_pipeline_progress(self, stage: str, path: Path):
        """Called from the pipeline WORKER thread.  We hop back onto the
        Tk main thread via after(0, ...) before touching widgets."""
        self.root.after(0, lambda: self._stream_tile(stage, path))

    # ===================================================================
    # Step actions
    # ===================================================================
    def pick_input(self):
        fp = filedialog.askopenfilename(
            title="Pick a scanned CDS sheet or PDF",
            filetypes=[("All supported",
                        "*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.pdf"),
                       ("Image files",
                        "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"),
                       ("PDF files", "*.pdf"),
                       ("All files", "*.*")],
        )
        if not fp:
            return
        path = Path(fp)
        if path.suffix.lower() == ".pdf":
            # PDF conversion can take 1-3s and looked like the UI had
            # frozen.  Show a tiny modal "please wait" while it happens.
            popup = self._show_busy_popup(
                "Converting PDF",
                f"Converting {path.name}\nto a 300 dpi PNG…")
            self.root.update()
            try:
                import fitz
                doc = fitz.open(str(path))
                pix = doc[0].get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))
                png_path = REPO / "data/outputs/InhouseProduction/_gui_tmp.png"
                png_path.parent.mkdir(parents=True, exist_ok=True)
                pix.save(str(png_path))
                img = cv2.imread(str(png_path))
            except Exception as exc:
                popup.destroy()
                messagebox.showerror("PDF conversion failed", str(exc))
                return
            popup.destroy()
        else:
            img = cv2.imread(str(path))
            if img is None:
                messagebox.showerror("Cannot read file",
                                     f"Could not load {path}")
                return
        self.input_path = path
        self.original_image_bgr = img
        self.oriented_image_bgr = img.copy()
        self.last_out_dir = None
        self.input_label.config(text=path.name)
        self._show_imported_only()
        self._set_run_state(ready=True)
        self.status.config(text="Loaded – orient if needed, then Vectorise.")

    def _show_busy_popup(self, title: str, message: str) -> Toplevel:
        """Modal 'please wait' popup that visibly hangs while a slow
        synchronous operation runs.  Caller is responsible for calling
        .destroy() once the work is done (or after an exception)."""
        top = Toplevel(self.root)
        top.title(title)
        top.configure(bg=BG)
        top.transient(self.root)
        top.resizable(False, False)
        # centre on parent
        self.root.update_idletasks()
        px = self.root.winfo_rootx() + self.root.winfo_width() // 2
        py = self.root.winfo_rooty() + self.root.winfo_height() // 2
        top.geometry(f"+{px - 160}+{py - 60}")
        Label(top, text=title, bg=BG, fg=ACCENT,
              font=self.f_h1).pack(padx=24, pady=(16, 4))
        Label(top, text=message, bg=BG, fg=FG,
              font=self.f_base, justify="center").pack(padx=24, pady=(0, 12))
        bar = ttk.Progressbar(top, mode="indeterminate", length=240,
                              style="AC.Horizontal.TProgressbar")
        bar.pack(padx=24, pady=(0, 16))
        bar.start(10)
        # prevent the user clicking anything else while it runs
        top.grab_set()
        top.update()
        return top

    def transform(self, kind: str):
        if self.oriented_image_bgr is None:
            return
        m = self.oriented_image_bgr
        if kind == "cw":
            m = cv2.rotate(m, cv2.ROTATE_90_CLOCKWISE)
        elif kind == "ccw":
            m = cv2.rotate(m, cv2.ROTATE_90_COUNTERCLOCKWISE)
        self.oriented_image_bgr = m
        # any orientation change invalidates a previous run
        self.last_out_dir = None
        self._show_imported_only()
        self._set_run_state(ready=True)

    def set_size(self, w, h, label):
        self.paper_w, self.paper_h = w, h
        self.paper_label = label
        # If user picked a preset, clear the custom entry fields.
        self.custom_w.set("")
        self.custom_h.set("")
        self._refresh_size_buttons()

    def _apply_custom_if_valid(self):
        """If BOTH width and height entry boxes parse to a positive
        integer, switch the active size to that custom value.  Empty or
        invalid input is silently ignored so the user can still tab
        between fields mid-edit."""
        try:
            w = int(float(self.custom_w.get().strip()))
            h = int(float(self.custom_h.get().strip()))
        except (ValueError, TypeError):
            return
        if w < 10 or h < 10 or w > 5000 or h > 5000:
            return
        self.paper_w = float(w)
        self.paper_h = float(h)
        self.paper_label = f"Custom ({w} x {h})"
        self._refresh_size_buttons(keep_custom_text=True)

    def _refresh_size_buttons(self, keep_custom_text: bool = False):
        for b, lbl in [
            (self.btn_large, "Large CDS (600 x 500)"),
            (self.btn_xlarge, "X-Large CDS (900 x 500)"),
        ]:
            active = (self.paper_label == lbl)
            b.config(bg=ACCENT if active else BG_CARD,
                     fg="#222222" if active else FG)
        # Highlight the custom Entry boxes when custom is the active mode.
        custom_active = self.paper_label.startswith("Custom")
        entry_bg = ACCENT if custom_active else BG_CARD
        entry_fg = "#222222" if custom_active else FG
        for w in (self._cw_entry, self._ch_entry):
            w.config(bg=entry_bg, fg=entry_fg, insertbackground=entry_fg)
        self.size_label.config(
            text=f"Selected: {self.paper_label}   "
                 f"→ {self.paper_w:g} x {self.paper_h:g} mm")

    def pick_output_folder(self):
        d = filedialog.askdirectory(title="Choose output folder",
                                    initialdir=str(self.output_root))
        if d:
            self.output_root = Path(d)
            self.output_label.config(text=str(self.output_root))

    # ===================================================================
    # Run state management
    # ===================================================================
    def _set_run_state(self, ready: bool):
        if ready:
            self.run_btn.config(state="normal", bg=ACCENT, fg="#222222",
                                text="VECTORISE")
            self.run_note.config(text="")
            self.repeat_btn.pack_forget()
            self.open_folder_btn.pack_forget()
        else:
            self.run_btn.config(state="disabled", bg=DISABLED, fg=MUTED)

    def _set_done_state(self):
        self.run_btn.config(state="disabled", bg=DISABLED, fg=MUTED)
        self.run_note.config(
            text="Please select a new CDS, or re-vectorise below.")
        self.repeat_btn.pack(anchor="w", pady=(2, 4))
        self.open_folder_btn.pack(anchor="w", pady=(4, 0))

    def run_pipeline_threaded(self):
        if self.oriented_image_bgr is None:
            messagebox.showwarning("No file", "Open a CDS first.")
            return
        # last chance to commit any pending custom-size edit
        self._apply_custom_if_valid()
        self.run_btn.config(state="disabled", bg=DISABLED, fg=MUTED)
        self.repeat_btn.config(state="disabled")
        self.progress.pack(anchor="w", pady=6)
        self.progress.start(10)
        self.status.config(text="Running pipeline…")
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        try:
            tmp_dir = REPO / "data/outputs/InhouseProduction/_gui_tmp"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            tmp_png = tmp_dir / f"{self.input_path.stem}.png"
            cv2.imwrite(str(tmp_png), self.oriented_image_bgr)
            _p1.PAPER_W_MM = self.paper_w
            _p1.PAPER_H_MM = self.paper_h
            out_dir, verdict = run_pipeline(
                tmp_png, self.output_root,
                forced_route=None, rectifier="scan", ts_prefix="gui",
                paper_w_mm=self.paper_w, paper_h_mm=self.paper_h,
                progress_callback=self._on_pipeline_progress,
            )
            self.last_out_dir = out_dir
            self.root.after(0, lambda: self._done_ok(out_dir, verdict))
        except Exception as exc:
            err = str(exc)
            self.root.after(0, lambda: self._done_err(err))

    def _done_ok(self, out_dir, verdict):
        self.progress.stop()
        self.progress.pack_forget()
        self.repeat_btn.config(state="normal")
        self.status.config(text=f"Done.  Verdict: {verdict}\n{out_dir.name}")
        self._show_all_stages(out_dir)
        self._set_done_state()

    def _done_err(self, err):
        self.progress.stop()
        self.progress.pack_forget()
        self.repeat_btn.config(state="normal")
        self.run_btn.config(state="normal", bg=ACCENT, fg="#222222")
        self.status.config(text="ERROR – see popup")
        messagebox.showerror("Pipeline failed", err)

    def open_folder(self):
        if self.last_out_dir is None:
            return
        import os
        os.startfile(str(self.last_out_dir))


def main():
    root = Tk()
    InHouseDigitiser(root)
    root.mainloop()


if __name__ == "__main__":
    main()
