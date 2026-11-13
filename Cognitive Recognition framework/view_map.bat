@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_EXE=.\07_environment_and_project_meta\.venv-gpu311\Scripts\python.exe"
set "SCRIPT=.\01_codebase\09_semantic_map\view_semantic_map.py"

if "%~1"=="" (
    "%PYTHON_EXE%" "%SCRIPT%" --menu --rerun
) else if /I "%~1"=="--html" (
    "%PYTHON_EXE%" "%SCRIPT%" --menu --html %2 %3 %4 %5 %6 %7 %8 %9
) else (
    "%PYTHON_EXE%" "%SCRIPT%" --db "%~1" --rerun %2 %3 %4 %5 %6 %7 %8 %9
)
