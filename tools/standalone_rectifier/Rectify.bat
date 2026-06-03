@echo off
REM ------------------------------------------------------------------
REM ArmourCore CDS Rectifier - drag-and-drop launcher
REM
REM Drop one or more PNG / JPG / PDF files onto this .bat and it will
REM rectify each one into a 'rectified\' folder next to the first file.
REM
REM No GUI.  No project install.  Just needs Python with opencv-python
REM + numpy + pillow installed.  (Plus pymupdf if you drop PDFs.)
REM ------------------------------------------------------------------
setlocal
set HERE=%~dp0

REM Prefer a portable venv inside this folder, then a sibling repo's
REM venv, then system python.
set PY=python
if exist "%HERE%.venv\Scripts\python.exe" set PY=%HERE%.venv\Scripts\python.exe
if exist "%HERE%..\..\.venv\Scripts\python.exe" set PY=%HERE%..\..\.venv\Scripts\python.exe
if exist "%HERE%..\..\venv\Scripts\python.exe" set PY=%HERE%..\..\venv\Scripts\python.exe

if "%~1"=="" (
    echo.
    echo  ArmourCore CDS Rectifier - drag-and-drop tool
    echo  ---------------------------------------------
    echo  Drop one or more PNG / JPG / PDF files onto this .bat.
    echo  Or run from a terminal:
    echo      Rectify.bat path\to\scan.png
    echo      Rectify.bat scan1.png scan2.pdf --xlarge --debug
    echo.
    echo  Defaults: Large CDS  ^(600 x 500 mm^)
    echo  Add --xlarge for X-Large ^(900 x 500 mm^).
    echo.
    pause
    exit /b 2
)

echo Rectifying %* ...
"%PY%" "%HERE%rectify.py" %*
set ERR=%ERRORLEVEL%

echo.
if "%ERR%"=="0" (
    echo Done.
) else (
    echo *** Some files failed - see above. ***
)
pause
endlocal & exit /b %ERR%
