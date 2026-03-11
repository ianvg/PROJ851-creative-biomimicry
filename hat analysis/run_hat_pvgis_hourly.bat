@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%"
python hat_thermal_analysis.py --use-pvgis --cities marseille,cairo %*
set "RC=%ERRORLEVEL%"
popd
exit /b %RC%
