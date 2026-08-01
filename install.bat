@echo off
REM install.bat — Windows convenience launcher
REM
REM  HOW TO RUN
REM  ----------
REM  Command Prompt:  install.bat
REM  PowerShell:      .\install.bat        <-- the .\ prefix is required in PowerShell
REM  Double-click:    works from Explorer
REM
REM  Optional flags (append after the command above):
REM    --check        verify installs only, do not install anything
REM    --skip-torch   skip PyTorch (e.g. you have a custom CUDA build)
REM    --skip-wandb   skip optional wandb install

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
