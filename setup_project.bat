@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Creating Python virtual environment...
  python -m venv .venv
)

echo Installing backend dependencies...
".venv\Scripts\python.exe" -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --timeout 120 --retries 5
if errorlevel 1 (
  echo Backend dependency installation failed.
  exit /b 1
)

echo Installing frontend dependencies...
cd /d "%~dp0frontend"
call npm.cmd config set registry https://registry.npmmirror.com
call npm.cmd install
if errorlevel 1 (
  echo Frontend dependency installation failed.
  exit /b 1
)

echo.
echo Setup completed.
echo Start backend:  start_backend.bat
echo Start frontend: start_frontend.bat
endlocal

