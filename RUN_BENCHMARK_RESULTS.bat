@echo off
setlocal

cd /d "%~dp0"
set OMP_NUM_THREADS=4

if not exist "build\atlas_bench.exe" (
    echo Missing build\atlas_bench.exe
    echo Build it first with:
    echo g++ -O3 -std=c++20 -I. -fopenmp BENCH/benchmark_suite.cpp CORE/lie_operator.cpp -o build/atlas_bench.exe
    pause
    exit /b 1
)

echo Running FlexMesh benchmark suite...
echo Output will be saved to benchmark_output.txt
echo.

"build\atlas_bench.exe" --threads 4 > "benchmark_output.txt" 2>&1
type "benchmark_output.txt"

echo.
echo Done. Results saved to benchmark_output.txt
pause
