@echo off
REM install.bat — Windows convenience launcher
REM Double-click this file, or run from Command Prompt / PowerShell:
REM   install.bat
REM   install.bat --check
REM   install.bat --skip-torch

where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python not found in PATH.
    echo Install from https://python.org ^(tick "Add to PATH" during setup^)
    pause
    exit /b 1
)

echo Using Python:
python --version

python install.py %*
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Install failed. See errors above.
    pause
    exit /b 1
)
pause
