@echo off
REM ------------------------------------------------------------------
REM  In-House CDS Digitiser - one-shot installer.
REM
REM  Creates a local .venv inside this folder and pip-installs
REM  requirements.txt into it.  Re-run any time to refresh deps after
REM  a `git pull`.
REM
REM  Needs: Python 3.10 or newer on PATH.
REM         (Microsoft Store "Python 3.12" works, or python.org.)
REM ------------------------------------------------------------------
setlocal
set REPO=%~dp0
cd /d "%REPO%"

echo.
echo  ArmourCore In-House CDS Digitiser - installer
echo  ----------------------------------------------
echo.

REM Verify Python
where python >nul 2>&1
if errorlevel 1 (
    echo *** Python is not on PATH.  Install Python 3.10+ from
    echo     https://www.python.org/downloads/windows/  or the
    echo     Microsoft Store, then re-run this installer.
    pause
    exit /b 1
)

python --version

REM Create venv if missing
if not exist "%REPO%.venv\Scripts\python.exe" (
    echo Creating .venv ...
    python -m venv .venv
    if errorlevel 1 (
        echo *** Failed to create .venv.  See message above.
        pause
        exit /b 1
    )
)

echo Upgrading pip ...
"%REPO%.venv\Scripts\python.exe" -m pip install --upgrade pip

echo Installing requirements ...
"%REPO%.venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo *** pip install failed.  See message above.
    pause
    exit /b 1
)

echo.
echo  Install complete.  Double-click Launch_InHouse_Digitiser.bat to start.
echo.
pause
endlocal
