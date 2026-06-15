@echo off
REM Backtest the strategy in config.yaml and save an equity chart (equity.png).
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Please run setup.bat first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" run_backtest.py --config config.yaml --plot equity.png
echo.
echo Chart saved as equity.png in this folder.
pause
