@echo off
REM ====================================================================
REM  Re-run the study on data ALREADY downloaded (data\raw\).
REM  No internet needed. Takes about 30 seconds.
REM  Use this after a code change, or to regenerate the result files.
REM ====================================================================

cd /d "%~dp0"

if not exist ".venv" (
    echo.
    echo ERROR: No .venv folder found. Run run.bat first - it does the setup.
    pause
    exit /b 1
)

if not exist "data\raw\ES_F_raw.csv" (
    echo.
    echo ERROR: No downloaded data found in data\raw\.
    echo Run run.bat first to download it.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat

echo.
echo Re-running on the data already in data\raw\ ...
echo.
python main.py --use-cache

echo.
echo ====================================================================
echo  Done. Updated files are in:
echo    outputs\results\   (CSV files and final_report.xlsx)
echo    outputs\plots\     (charts)
echo    outputs\run_log.txt
echo ====================================================================
pause
