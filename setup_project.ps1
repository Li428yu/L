$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonExe = Join-Path $projectRoot ".venv\Scripts\python.exe"
$frontendDir = Join-Path $projectRoot "frontend"

Set-Location $projectRoot

if (-not (Test-Path $pythonExe)) {
    Write-Host "正在创建 Python 虚拟环境..."
    python -m venv .venv
}

Write-Host "正在安装后端依赖..."
& $pythonExe -m pip install -r requirements.txt

Write-Host "正在安装前端依赖..."
Set-Location $frontendDir
npm.cmd install

Write-Host ""
Write-Host "安装完成。接下来打开两个终端分别运行："
Write-Host "1. .\start_backend.ps1"
Write-Host "2. .\start_frontend.ps1"

