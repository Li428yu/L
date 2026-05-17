$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonExe = Join-Path $projectRoot ".venv\Scripts\python.exe"
$appFile = Join-Path $projectRoot "app.py"

if (-not (Test-Path $pythonExe)) {
    Write-Error "未找到项目虚拟环境：$pythonExe`n请先在项目目录执行：python -m venv .venv"
}

& $pythonExe -m streamlit run $appFile
