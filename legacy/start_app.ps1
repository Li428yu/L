$ErrorActionPreference = "Stop"

$legacyRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $legacyRoot
$pythonExe = Join-Path $projectRoot ".venv\Scripts\python.exe"
$appFile = Join-Path $legacyRoot "app.py"

if (-not (Test-Path $pythonExe)) {
    Write-Error "未找到项目虚拟环境：$pythonExe`n请先在项目目录执行：python -m venv .venv"
}

Set-Location $legacyRoot
& $pythonExe -m streamlit run $appFile
