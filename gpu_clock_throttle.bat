@echo off
echo Current GPU clocks:
nvidia-smi -q -d CLOCK
echo.
echo Locking GPU to max 1800 MHz...
nvidia-smi -lgc 210,1800
echo.Rel
echo Done. To reset after training, run: nvidia-smi -rgc
pause
