@echo off
REM ------------------------------------------------------------------
REM  In-House CDS Digitiser launcher
REM
REM  Double-click to start the GUI as a windowed application.
REM  Uses pythonw.exe so NO console window stays open in the
REM  background - the GUI looks like a normal Windows app.
REM
REM  If the GUI fails to start, run this file from a CMD prompt
REM  instead - that surfaces any traceback.
REM ------------------------------------------------------------------
setlocal
set REPO=%~dp0
cd /d "%REPO%"

REM Prefer the venv's pythonw (no console).  Fall back to system
REM pythonw.  Final fallback: console python (will show CMD window
REM but at least the user sees errors).
set PYW=pythonw
if exist "%REPO%.venv\Scripts\pythonw.exe" set PYW=%REPO%.venv\Scripts\pythonw.exe
if exist "%REPO%venv\Scripts\pythonw.exe"  set PYW=%REPO%venv\Scripts\pythonw.exe

REM `start "" /B` detaches us from the CMD window so the .bat exits
REM immediately and Windows treats the GUI as a standalone app.
start "" /B "%PYW%" tools\in_house_digitiser.py
endlocal
