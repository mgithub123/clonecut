@echo off
REM Double-click this file to start the app on Windows.
cd /d "%~dp0"

where ffmpeg >nul 2>&1
if errorlevel 1 (
  echo   ! ffmpeg is not on PATH. Install it from https://ffmpeg.org/download.html
  echo     and make sure ffmpeg.exe is on your PATH, then run this again.
  echo.
)

where uv >nul 2>&1
if errorlevel 1 (
  echo   ! uv is not installed. Install it with:
  echo       powershell -c "irm https://astral.sh/uv/install.ps1 ^| iex"
  echo     then close this window and double-click again.
  echo.
  pause
  exit /b 1
)

uv run app.py
echo.
pause
