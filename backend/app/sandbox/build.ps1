# Builds the sandbox Docker image used to run all AI-generated analysis code.
# Run from anywhere; paths are resolved relative to this script.
$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
docker build -t ai-data-analyst-sandbox:latest -f "$scriptDir\Dockerfile" "$scriptDir"
Write-Host "Built image: ai-data-analyst-sandbox:latest"
