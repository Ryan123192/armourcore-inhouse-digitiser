@echo off
REM ------------------------------------------------------------------
REM In-House Digitiser launcher
REM Double-click this file to start the GUI.
REM ------------------------------------------------------------------
setlocal
set REPO=%~dp0
cd /d "%REPO%"

REM Try venv python first, fall back to system python
set PY=python
if exist "%REPO%.venv\Scripts\python.exe" set PY=%REPO%.venv\Scripts\python.exe
if exist "%REPO%venv\Scripts\python.exe" set PY=%REPO%venv\Scripts\python.exe

echo Launching In-House Digitiser...
"%PY%" tools\in_house_digitiser.py

if errorlevel 1 (
    echo.
    echo *** Pipeline failed - see message above. ***
    pause
)
endlocal
