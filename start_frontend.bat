@echo off
setlocal
cd /d "%~dp0frontend"

if not exist "node_modules" (
  echo Missing node_modules. Please run setup_project.bat first.
  exit /b 1
)

call npm.cmd run dev
endlocal

