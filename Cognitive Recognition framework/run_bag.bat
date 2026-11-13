@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_EXE=.\07_environment_and_project_meta\.venv-gpu311\Scripts\python.exe"
set "SCRIPT=.\01_codebase\09_semantic_map\build_semantic_map_from_bag.py"

if "%~1"=="" (
    "%PYTHON_EXE%" "%SCRIPT%" --frame-stride 4 --min-consensus-hits 3 --reset-map
) else (
    "%PYTHON_EXE%" "%SCRIPT%" --bag "%~1" --frame-stride 4 --min-consensus-hits 3 --reset-map %2 %3 %4 %5 %6 %7 %8 %9
)
