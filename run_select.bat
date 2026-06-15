@echo off
REM Find the best strategy on real market data (out-of-sample validated).
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Please run setup.bat first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" run_select.py --config config.yaml
echo.
pause
