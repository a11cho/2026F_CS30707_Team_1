param(
    [switch]$ManualAll,
    [ValidateSet('independent', 'sequential', 'feedback')]
    [string]$Strategy = 'independent',
    [ValidateCount(3, 3)]
    [ValidateSet(0, 1)]
    [int[]]$RewardMask = @(1, 0, 0),
    [int]$ObservationWindowSize = 15,
    [switch]$RandomizeObservationRotation
)
$ErrorActionPreference = 'Stop'
$projectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) {
    throw 'Create project/.venv and install requirements first; see README.md.'
}
Push-Location $PSScriptRoot
try {
    $playArguments = @('-m', 'territory_rl', '--strategy', $Strategy, '--reward-mask') + $RewardMask + @(
        '--observation-window-size', $ObservationWindowSize)
    if ($ManualAll) { $playArguments += '--manual-all' }
    if ($RandomizeObservationRotation) { $playArguments += '--randomize-observation-rotation' }
    & $projectPython @playArguments
} finally { Pop-Location }
