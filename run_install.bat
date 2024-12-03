@echo off
:: Set the code page to UTF-8 for proper encoding of Russian characters
chcp 65001

:: Check if the script is not run with administrator rights
NET SESSION >nul 2>&1
if %errorlevel% NEQ 0 (
    echo Running with administrator rights...
    :: Relaunching itself with administrator rights
    powershell -Command "Start-Process cmd -ArgumentList '/c, %~s0' -Verb RunAs"
    exit /b
)

:: If administrator rights are available, get the absolute path to the current .bat file
set BAT_PATH=%~dp0
set SCRIPT_PATH="%BAT_PATH%/scripts/install_dependencies.ps1"

:: Check if the script exists before running it
if not exist %SCRIPT_PATH% (
    echo Script install_dependencies.ps1 not found at: %SCRIPT_PATH%
    pause
    exit /b
)

:: Informing what will be executed
echo Script found: %SCRIPT_PATH%
echo Running the PowerShell script...

:: Run the PowerShell script
powershell -ExecutionPolicy Bypass -File %SCRIPT_PATH%

:: Wait for key press before closing
pause
