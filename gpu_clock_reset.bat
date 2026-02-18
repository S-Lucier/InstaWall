@echo off
echo Resetting GPU clocks to default...
nvidia-smi -rgc
echo Done.
pause
