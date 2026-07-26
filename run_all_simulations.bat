@echo off
setlocal
python simulate_and_run_python.py
if errorlevel 1 exit /b %errorlevel%

"c:\Program Files\r\R-4.5.2\bin\Rscript.exe" run_simulations_r_justvsn.R vsn_simulation_results
if errorlevel 1 exit /b %errorlevel%

python compare_simulation_results.py vsn_simulation_results
if errorlevel 1 exit /b %errorlevel%

echo Done. Upload the TSV, JSON, and session-info files listed above.
