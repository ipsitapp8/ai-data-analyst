<#
.SYNOPSIS
  Start (or restart) the whole app locally: API on :8001 and Streamlit on :8501.

.DESCRIPTION
  Stops whatever is already listening on those two ports first, so you never end up talking
  to a stale server that was started with an older .env. Both servers read .env as usual
  (DATABASE_URL, BACKEND_BASE_URL, ...). Logs go to .run-local/.

.EXAMPLE
  ./run-local.ps1          # restart both and open the browser
  ./run-local.ps1 -Stop    # just stop them
  ./run-local.ps1 -NoOpen  # don't open the browser
#>
param(
    [switch]$Stop,
    [switch]$NoOpen,
    [int]$ApiPort = 8001,
    [int]$UiPort = 8501
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"
$logs = Join-Path $root ".run-local"

if (-not (Test-Path $python)) {
    throw "No virtualenv at $python. Create it first: python -m venv .venv; .venv\Scripts\pip install -r backend\requirements.txt -r frontend\requirements.txt"
}

function Stop-Port([int]$Port) {
    $pids = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty OwningProcess -Unique
    foreach ($p in $pids) {
        Write-Host "Stopping process $p on port $Port"
        Stop-Process -Id $p -Force -ErrorAction SilentlyContinue
    }
}

function Wait-Url([string]$Url, [int]$Seconds, [string]$Name) {
    $deadline = (Get-Date).AddSeconds($Seconds)
    while ((Get-Date) -lt $deadline) {
        try {
            if ((Invoke-WebRequest $Url -UseBasicParsing -TimeoutSec 3).StatusCode -eq 200) { return }
        } catch { Start-Sleep -Seconds 2 }
    }
    throw "$Name did not come up within $Seconds s. See $logs for its log."
}

Stop-Port $ApiPort
Stop-Port $UiPort
if ($Stop) { Write-Host "Stopped."; return }

New-Item -ItemType Directory -Force $logs | Out-Null
# A DATABASE_URL left in this shell would override .env; start from the file's values.
Remove-Item Env:DATABASE_URL -ErrorAction SilentlyContinue
$env:BACKEND_BASE_URL = "http://127.0.0.1:$ApiPort"

Write-Host "Starting API on :$ApiPort ..."
Start-Process -FilePath $python -WorkingDirectory $root -WindowStyle Hidden `
    -ArgumentList "-m", "uvicorn", "app.main:app", "--app-dir", "backend", "--host", "127.0.0.1", "--port", $ApiPort `
    -RedirectStandardOutput (Join-Path $logs "api.out.log") -RedirectStandardError (Join-Path $logs "api.err.log")
Wait-Url "http://127.0.0.1:$ApiPort/api/health" 90 "The API"

Write-Host "Starting Streamlit on :$UiPort ..."
Start-Process -FilePath $python -WorkingDirectory $root -WindowStyle Hidden `
    -ArgumentList "-m", "streamlit", "run", "frontend/Home.py", "--server.port", $UiPort, "--server.headless", "true", "--browser.gatherUsageStats", "false" `
    -RedirectStandardOutput (Join-Path $logs "ui.out.log") -RedirectStandardError (Join-Path $logs "ui.err.log")
Wait-Url "http://localhost:$UiPort/_stcore/health" 60 "Streamlit"

$health = (Invoke-WebRequest "http://127.0.0.1:$ApiPort/api/health" -UseBasicParsing).Content
Write-Host ""
Write-Host "Ready:  http://localhost:$UiPort   (API http://127.0.0.1:$ApiPort)"
Write-Host "Health: $health"
Write-Host "Logs:   $logs   |   Stop with: ./run-local.ps1 -Stop"
if (-not $NoOpen) { Start-Process "http://localhost:$UiPort" }
