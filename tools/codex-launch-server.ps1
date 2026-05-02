param(
    [switch]$Sync
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

if ($Sync) {
    uv sync
}

$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (Test-Path $venvPython) {
    & $venvPython -c "from expresso_calib.server import main; main()"
} else {
    uv run expresso-calib
}
