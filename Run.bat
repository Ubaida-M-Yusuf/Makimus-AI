@echo off
call "%~dp0venv\Scripts\activate.bat"
py -3.10 Makimus-AI.py
if errorlevel 1 (
    python Makimus-AI.py
)
pause