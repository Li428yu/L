@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Missing .venv. Please run setup_project.bat first.
  pause
  exit /b 1
)

if not exist "frontend\node_modules" (
  echo Missing frontend node_modules. Please run setup_project.bat first.
  pause
  exit /b 1
)

echo Starting backend and frontend...
echo Backend:  http://127.0.0.1:8000
echo Frontend: http://127.0.0.1:5173
echo.

start "Paper Assistant Backend" cmd /k call "%~dp0start_backend.bat"
start "Paper Assistant Frontend" cmd /k call "%~dp0start_frontend.bat"

timeout /t 3 /nobreak >nul
start "" "http://127.0.0.1:5173"

endlocal
