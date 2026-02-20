@echo off
echo Disabling Modern Standby (requires reboot)...

reg add "HKLM\SYSTEM\CurrentControlSet\Control\Power" /v PlatformAoAcOverride /t REG_DWORD /d 0 /f
reg add "HKLM\SYSTEM\CurrentControlSet\Control\Power" /v CsEnabled /t REG_DWORD /d 0 /f

echo.
echo Done. Run powercfg /a to verify S3 is available after reboot.
echo Reboot now for changes to take effect.
pause
