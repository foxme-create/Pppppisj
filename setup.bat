@echo off
REM One-time setup on Windows. Double-click to run.
REM Calls the venv's python directly so it works even when PowerShell blocks
REM script activation and the global pip launcher is broken.
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
  echo.
  echo [!] Python not found. Install Python 3.11+ from https://www.python.org/downloads/
  echo     During install, TICK the box "Add python.exe to PATH".
  echo.
  pause
  exit /b 1
)

echo Creating virtual environment...
python -m venv .venv

echo Installing dependencies (this may take a minute)...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt

if not exist config.yaml (
  copy config.example.yaml config.yaml
  echo Created config.yaml from the example.
)

echo.
echo ============================================================
echo  Setup complete!
echo  Next: double-click  run_select.bat
echo ============================================================
echo.
pause
