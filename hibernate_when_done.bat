@echo off
echo Watching for sweep to finish...
echo Close this window to cancel hibernate.
echo.

:WAIT
if exist "outputs\hparam_sweep\20260219_155901\summary.md" goto FOUND
timeout /t 30 /nobreak >nul
goto WAIT

:FOUND
echo Sweep finished! Hibernating in 30 seconds...
echo Close this window to cancel.
timeout /t 30 /nobreak
shutdown /h
