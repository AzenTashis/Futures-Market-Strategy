@echo off
REM ====================================================================
REM  Futures Market Strategy Backtesting & Risk Analysis
REM  One-click setup and run for Windows.
REM  Double-click this file, or run it from a command prompt.
REM ====================================================================

cd /d "%~dp0"

echo.
echo [1/4] Creating virtual environment (first run only)...
if not exist ".venv" (
    python -m venv .venv
    if errorlevel 1 (
        echo.
        echo ERROR: Could not create a virtual environment.
        echo Check that Python is installed and on your PATH: try "python --version".
        pause
        exit /b 1
    )
) else (
    echo       .venv already exists, skipping.
)

echo.
echo [2/4] Installing dependencies...
call .venv\Scripts\activate.bat
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt
if errorlevel 1 (
    echo.
    echo ERROR: Dependency install failed. Check your internet connection.
    pause
    exit /b 1
)

echo.
echo [3/4] Running sanity checks...
python tests\test_sanity.py
if errorlevel 1 (
    echo.
    echo ERROR: Sanity checks failed. Do not trust any results until this passes.
    pause
    exit /b 1
)

echo.
echo [4/4] Running the full study (this downloads ~10 years of data, please wait)...
python main.py

echo.
echo ====================================================================
echo  Done. Results are in:
echo    outputs\results\   (CSV files and final_report.xlsx)
echo    outputs\plots\     (charts)
echo ====================================================================
pause
