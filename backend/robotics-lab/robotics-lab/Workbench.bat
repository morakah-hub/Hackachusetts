@echo off
rem Robotics Lab Workbench (v0.7): double-click to start; close with the Quit button or Ctrl+C.
cd /d "%~dp0"
".venv\Scripts\python.exe" workbench.py %*
