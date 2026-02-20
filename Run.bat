@echo off
if not exist "%~dp0venv" (
    echo Venv not found! Please run Install.bat first.
    pause
    exit
)
call "%~dp0venv\Scripts\activate.bat"
py -3.11 Makimus-AI.py
if errorlevel 1 (
    py -3.10 Makimus-AI.py
)
if errorlevel 1 (
    python Makimus-AI.py
)
if errorlevel 1 (
    echo No compatible Python found! Please install Python 3.10 or 3.11 from python.org
    pause
)
pause