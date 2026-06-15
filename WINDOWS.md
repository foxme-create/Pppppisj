# Running on Windows

## Easiest way (double-click)

1. Install **Python 3.11+** from https://www.python.org/downloads/
   — during install, tick **"Add python.exe to PATH"**.
2. Double-click **`setup.bat`** (one time — installs everything).
3. Double-click **`run_select.bat`** to find the best strategy on real data.
4. Double-click **`run_backtest.bat`** to backtest + save `equity.png`.

## Manual way (PowerShell)

PowerShell often blocks venv activation and the global `pip` can be broken.
Avoid both by calling the venv's Python directly — **no activation needed**:

```powershell
cd C:\Users\ruste\Desktop\bots-trading

python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

copy config.example.yaml config.yaml
.\.venv\Scripts\python.exe run_select.py --config config.yaml
```

Always launch scripts with `.\.venv\Scripts\python.exe <script>.py` so they use
the virtual environment (where the dependencies live), not the global Python.

## Common errors

- **`Activate.ps1 cannot be loaded ... running scripts is disabled`** — you
  don't need to activate. Use `.\.venv\Scripts\python.exe ...` directly (above).
  (Or, if you prefer activation: run once
  `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.)
- **`Fatal error in launcher ... pip.exe`** — the global `pip` shortcut is
  broken. Use `python -m pip` (or the venv's python) instead of bare `pip`.
- **`ModuleNotFoundError: No module named 'yaml'`** — dependencies weren't
  installed into the Python you're running. Re-run the install line with the
  venv's python, then run the script with the same venv python.
