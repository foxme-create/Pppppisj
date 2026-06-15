@echo off
REM Scan several pairs and timeframes for any out-of-sample edge.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Please run setup.bat first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" run_scan.py --symbols BTC/USDT,ETH/USDT,SOL/USDT --timeframes 1h,4h,1d
echo.
pause
