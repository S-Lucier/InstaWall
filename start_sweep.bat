@echo off
:: ============================================================
:: InstaWall Sweep Runner
:: Double-click to start (or drag onto it: start_sweep.bat --tier 0)
:: ============================================================

:: --- EDIT THIS to match your conda environment name ---
set CONDA_ENV=instawall

:: --- Project directory is wherever this bat file lives ---
set PROJECT_DIR=%~dp0
cd /d "%PROJECT_DIR%"

:: Activate conda
call conda activate %CONDA_ENV% 2>nul
if errorlevel 1 (
    echo.
    echo ERROR: Could not activate conda env "%CONDA_ENV%"
    echo Make sure conda is installed and run:
    echo     conda create -n %CONDA_ENV% python=3.10
    echo     conda activate %CONDA_ENV%
    echo     pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

:: Install rich if missing (needed for the live dashboard)
python -c "from rich.live import Live" 2>nul
if errorlevel 1 (
    echo Installing rich...
    pip install rich
)

echo.
echo ============================================================
echo  InstaWall Hyperparameter Sweep
echo  Output: %PROJECT_DIR%outputs\sweep
echo ============================================================
echo.
echo Commands you can pass (drag bat file onto cmd or edit below):
echo   --tier 0          run only tier 0 (sanity check)
echo   --tier 0 1        run tiers 0 and 1
echo   --only tex_b0     run experiments containing "tex_b0"
echo   --dry-run         list all experiments without running
echo   --summary         show results table
echo.

:: Pass any arguments the user dragged/typed through
python run_sweep.py %*

echo.
if errorlevel 1 (
    echo Sweep exited with an error. Check the log above.
) else (
    echo Sweep complete.
)
pause
