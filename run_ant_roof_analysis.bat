@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "PY_SCRIPT=%SCRIPT_DIR%ant_roof_analysis.py"

if not exist "%PY_SCRIPT%" (
  echo Error: Could not find ant_roof_analysis.py in %SCRIPT_DIR%
  exit /b 1
)

python "%PY_SCRIPT%" %*
if errorlevel 1 (
  echo.
  echo Analysis failed.
  exit /b %errorlevel%
)

echo.
echo Done. Report generated at:
echo %SCRIPT_DIR%ant_roof_cooling_report.html

endlocal
