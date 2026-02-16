@echo off
echo Setting GPU power limit to 70W for training...
nvidia-smi -pl 70
if %errorlevel% neq 0 (
    echo FAILED - Right-click this file and "Run as administrator"
    pause
    exit /b 1
)
echo Done.
pause
