@echo off
REM PM2 Python launcher. Does NOT change directory — PM2 app `cwd` controls that.
setlocal

if defined FACE_REC_PYTHON (
  "%FACE_REC_PYTHON%" -u %*
  exit /b %ERRORLEVEL%
)

where python >nul 2>&1
if %ERRORLEVEL%==0 (
  python -u %*
  exit /b %ERRORLEVEL%
)

REM Fallback: known Store Python 3.10 on this machine
set "STORE_PY=%LOCALAPPDATA%\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.10_qbz5n2kfra8p0\python.exe"
if exist "%STORE_PY%" (
  "%STORE_PY%" -u %*
  exit /b %ERRORLEVEL%
)

echo ERROR: No Python found. Set FACE_REC_PYTHON to your python.exe >&2
exit /b 1
