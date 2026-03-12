@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "PY_SCRIPT=%SCRIPT_DIR%simple_roof_analysis.py"

if not exist "%PY_SCRIPT%" (
  echo Error: Could not find simple_roof_analysis.py in %SCRIPT_DIR%
  exit /b 1
)

pushd "%SCRIPT_DIR%"
python "%PY_SCRIPT%" %*
if errorlevel 1 (
  popd
  echo.
  echo Analysis failed.
  exit /b %errorlevel%
)
popd

echo.
echo Done. Report generated at:
echo %SCRIPT_DIR%ant_roof_cooling_report.html

endlocal
