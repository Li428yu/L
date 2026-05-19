$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonExe = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $pythonExe)) {
    Write-Error "未找到项目虚拟环境：$pythonExe`n请先在项目目录执行：python -m venv .venv"
}

Set-Location $projectRoot
& $pythonExe -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000

