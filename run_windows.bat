@echo off
REM WasteWise AI - one-click setup and launch for Windows
cd /d "%~dp0"
if not exist .venv (
    echo Creating virtual environment...
    python -m venv .venv || (echo Python not found. Install Python 3.11 or 3.12 from python.org and tick "Add to PATH". & pause & exit /b 1)
)
call .venv\Scripts\activate.bat
echo Installing dependencies (first run only takes a minute)...
python -m pip install --upgrade pip >nul
pip install -r requirements.txt || (echo Dependency install failed. & pause & exit /b 1)
echo Starting WasteWise AI at http://localhost:8501
streamlit run main.py
pause
