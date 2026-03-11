@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%"
python interactive_map_analysis.py
set "RC=%ERRORLEVEL%"
popd
exit /b %RC%
