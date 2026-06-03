ArmourCore CDS Rectifier - Standalone Drag-and-Drop Tool
=========================================================

What it is
----------
A self-contained folder you can MOVE ANYWHERE on the machine.  Drop
PNG / JPG / TIFF / PDF scans of a GLOBAL CDS sheet onto Rectify.bat
and you get a perspective-corrected PNG of the calibrated design area
written into a "rectified\" subfolder next to the first input.

This is JUST the rectifier - the full pipeline (colour cleanup +
vectorising + SVG output) lives in the main In-House CDS Digitiser GUI.

How to use it
-------------
1.  Single file: drag a scan onto Rectify.bat.
2.  Many files: select them all in Explorer and drag the lot.
3.  From a terminal:

        Rectify.bat scan.png
        Rectify.bat scan1.png scan2.pdf
        Rectify.bat scan.png --xlarge          (use X-Large 900x500)
        Rectify.bat scan.png --paper-w 800 --paper-h 400   (custom)
        Rectify.bat scan.png --debug           (also write _corners.png)

Output
------
For each input <name>.<ext> you get:
    rectified\<name>_rectified.png
With --debug also:
    rectified\<name>_corners.png      (markers drawn on the source)

Defaults
--------
Paper size : Large CDS, 600 x 500 mm calibrated design area
Resolution : 6 px / mm  (~3600 px on the long side of a Large sheet)

Requirements
------------
- Python 3.10+ on PATH (or a venv at .venv\ alongside this folder).
- Packages:
    pip install opencv-python numpy pillow
- For PDF input ALSO:
    pip install pymupdf

If you have the main repo checked out, this script will auto-pick up
its venv at ..\..\.venv\ or ..\..\venv\ so you don't need a second one.

Files in this folder
--------------------
Rectify.bat        drag-and-drop launcher
rectify.py         CLI driver (parse args, load image, write PNG)
_rectify_lib.py    self-contained rectifier (red-marker detector +
                   minAreaRect inner-corner + perspective warp)
README.txt         this file

Sync
----
_rectify_lib.py is a hand-synced COPY of
    src\armourcore_cds\phase1\marker_rectify_scan.py
from the main repo, with the two cross-module constants inlined so the
folder has zero project imports.  If you fix a bug in either copy,
mirror it in the other.
