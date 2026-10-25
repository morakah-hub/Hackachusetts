@echo off
rem ArmStrong launcher. Double-click to start. Nothing needs to be installed first.
rem The first run sets up Python and builds the web app (a few minutes, needs internet), then serves
rem http://127.0.0.1:8770 and opens the browser. Close this window (or press Ctrl+C) to stop the server.
rem Extra options go to start-studio.ps1, for example: start.bat -NoBrowser -Port 8771
cd /d "%~dp0"
set "POWERSHELL_EXE=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"
if not exist "%POWERSHELL_EXE%" set "POWERSHELL_EXE=powershell"
"%POWERSHELL_EXE%" -NoProfile -ExecutionPolicy Bypass -File "%~dp0start-studio.ps1" %*
if errorlevel 1 (
    echo.
    echo ArmStrong stopped with an error. See the messages above.
    pause
)
