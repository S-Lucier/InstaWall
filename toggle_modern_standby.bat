@echo off
:: Request admin elevation if not already running as admin
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo Requesting administrator privileges...
    powershell -Command "Start-Process '%~f0' -Verb RunAs"
    exit /b
)

echo ============================================
echo        Modern Standby Toggle
echo ============================================
echo.

:: Check current state
reg query "HKLM\SYSTEM\CurrentControlSet\Control\Power" /v PlatformAoAcOverride >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=3" %%a in ('reg query "HKLM\SYSTEM\CurrentControlSet\Control\Power" /v PlatformAoAcOverride') do set CURRENT=%%a
) else (
    set CURRENT=NOT SET
)

if "%CURRENT%"=="0x0" (
    echo Current status: DISABLED (screen-off won't sleep)
) else (
    echo Current status: ENABLED (Windows default)
)

echo.
echo What would you like to do?
echo   1 - Disable Modern Standby  (screen-off stays awake - good for training)
echo   2 - Enable Modern Standby   (restore Windows default)
echo   3 - Exit
echo.
set /p CHOICE="Enter choice (1/2/3): "

if "%CHOICE%"=="1" goto DISABLE
if "%CHOICE%"=="2" goto ENABLE
if "%CHOICE%"=="3" goto END
echo Invalid choice.
goto END

:DISABLE
echo.
echo Disabling Modern Standby...
reg add "HKLM\SYSTEM\CurrentControlSet\Control\Power" /v PlatformAoAcOverride /t REG_DWORD /d 0 /f
reg add "HKLM\SYSTEM\CurrentControlSet\Control\Power" /v CsEnabled /t REG_DWORD /d 0 /f
echo Done. Reboot for changes to take effect.
goto END

:ENABLE
echo.
echo Enabling Modern Standby (restoring Windows default)...
reg delete "HKLM\SYSTEM\CurrentControlSet\Control\Power" /v PlatformAoAcOverride /f 2>nul
reg delete "HKLM\SYSTEM\CurrentControlSet\Control\Power" /v CsEnabled /f 2>nul
echo Done. Reboot for changes to take effect.
goto END

:END
echo.
pause
