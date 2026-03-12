@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%"
python simple_roof_analysis.py --use-pvgis --cities marseille,cairo %*
set "RC=%ERRORLEVEL%"
popd
exit /b %RC%
