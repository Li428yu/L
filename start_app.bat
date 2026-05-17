@echo off
setlocal

set "PROJECT_ROOT=%~dp0"
set "PYTHON_EXE=%PROJECT_ROOT%.venv\Scripts\python.exe"
set "APP_FILE=%PROJECT_ROOT%app.py"

if not exist "%PYTHON_EXE%" (
  echo 未找到项目虚拟环境：%PYTHON_EXE%
  echo 请先在项目目录执行：python -m venv .venv
  exit /b 1
)

"%PYTHON_EXE%" -m streamlit run "%APP_FILE%"
