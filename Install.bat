@echo off
if not exist "%~dp0venv" (
    py -3.12 -m venv venv
    if errorlevel 1 (
        py -3.11 -m venv venv
    )
    if errorlevel 1 (
        py -3.10 -m venv venv
    )
    if errorlevel 1 (
        python -m venv venv
    )
    if errorlevel 1 (
        echo No compatible Python found! Please install Python 3.10, 3.11, or 3.12 from python.org
        pause
        exit
    )
)
call "%~dp0venv\Scripts\activate.bat"
pip install -r requirements.txt
python Makimus-AI.py
pause