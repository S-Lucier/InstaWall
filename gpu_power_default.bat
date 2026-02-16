@echo off
echo Restoring GPU power limit to default (80W)...
nvidia-smi -pl 80
if %errorlevel% neq 0 (
    echo FAILED - Right-click this file and "Run as administrator"
    pause
    exit /b 1
)
echo Done.
pause
