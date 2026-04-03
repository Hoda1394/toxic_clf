@echo off
REM Quick start script for Windows

echo =========================================
echo Toxic Comment Classifier - Setup Script
echo =========================================

REM Check Python version
echo Python version:
python --version

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Install package in development mode
echo Installing package...
pip install -e .

REM Run quick tests
echo.
echo Running quick tests...
python tests/test_quick.py

echo.
echo =========================================
echo Setup complete!
echo =========================================
echo.
echo Next steps:
echo 1. Prepare your data with 'text' and 'toxicity' columns
echo 2. Run: python scripts/train.py --data_path data/train.csv
echo 3. Or try the demo: python scripts/demo.py
echo.
echo To deactivate virtual environment, run: deactivate
