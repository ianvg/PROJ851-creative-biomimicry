@echo off
setlocal

set "PYTHON_EXE=C:\Users\ianva\.pyenv\pyenv-win\versions\3.10.5\python.exe"
set "SCRIPT_DIR=%~dp0"

"%PYTHON_EXE%" "%SCRIPT_DIR%ant_roof_analysis.py" ^
  --solar 900 ^
  --ambient-c 42 ^
  --sky-c 15 ^
  --f-sky 0.7 ^
  --f-ground 0.2 ^
  --f-air 0.05 ^
  --f-surrounding 0.05 ^
  --ground-c 48 ^
  --surrounding-c 38 ^
  --html "%SCRIPT_DIR%Reports\interactive_roof_report.html"

if errorlevel 1 (
  echo.
  echo Interactive roof report generation failed.
  exit /b 1
)

echo.
echo Interactive roof report created:
echo %SCRIPT_DIR%Reports\interactive_roof_report.html

endlocal
