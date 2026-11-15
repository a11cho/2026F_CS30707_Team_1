param(
    [switch]$ManualAll,
    [ValidateSet('independent', 'sequential', 'feedback')]
    [string]$Strategy = 'independent'
)
$ErrorActionPreference = 'Stop'
$projectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) {
    throw 'Create project/.venv and install requirements first; see README.md.'
}
Push-Location $PSScriptRoot
try {
    if ($ManualAll) { & $projectPython -m territory_rl --strategy $Strategy --manual-all }
    else { & $projectPython -m territory_rl --strategy $Strategy }
} finally { Pop-Location }
