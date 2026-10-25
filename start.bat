@echo off
rem Armstrong Studio: double-click to start (or use Start.exe). Close this window to stop the app.
title Armstrong Studio
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1" %*
if errorlevel 1 pause
