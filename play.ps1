param([switch]$ManualAll)
$ErrorActionPreference = 'Stop'
$projectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) {
    throw 'Create project/.venv and install requirements first; see README.md.'
}
Push-Location $PSScriptRoot
try {
    if ($ManualAll) { & $projectPython -m territory_rl --manual-all }
    else { & $projectPython -m territory_rl }
} finally { Pop-Location }
