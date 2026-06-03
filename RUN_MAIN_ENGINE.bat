@echo off
setlocal

cd /d "%~dp0"
set OMP_NUM_THREADS=4

if not exist "build\atlas_laamo.exe" (
    echo Missing build\atlas_laamo.exe
    echo Build it first with:
    echo g++ -O3 -std=c++20 -I. -fopenmp APP/main.cpp CORE/lie_operator.cpp CORE/pathological_suite.cpp -o build/atlas_laamo.exe
    pause
    exit /b 1
)

echo Running FlexMesh main engine...
echo Output will be saved to main_engine_output.txt
echo.

"build\atlas_laamo.exe" > "main_engine_output.txt" 2>&1
type "main_engine_output.txt"

echo.
echo Done. Results saved to main_engine_output.txt
pause
