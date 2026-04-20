@echo off
setlocal

cd /d "C:\Users\RT01579\OneDrive - Telefonica\Escritorio\Proyects\UPM\irrigation_decision_support_system"

call ".venv\Scripts\activate.bat"
python --version
python main.py

pause
endlocal