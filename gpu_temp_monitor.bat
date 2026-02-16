@echo off
echo Monitoring GPU temperature (updates every 5s, Ctrl+C to stop)
echo.
nvidia-smi -l 5 --query-gpu=timestamp,temperature.gpu,power.draw,power.limit --format=csv
