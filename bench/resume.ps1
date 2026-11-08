# Re-runs a benchmark until its result file exists. Finished episodes are
# cached under runs/, so every restart resumes where the last one stopped.
#   powershell -ExecutionPolicy Bypass -File bench/resume.ps1 -Suite slack -Defense provguard
#   powershell -ExecutionPolicy Bypass -File bench/resume.ps1 -Suite slack -Defense provguard -Policies bench/policies-v3
param(
    [string]$Suite = "banking",
    [string]$Defense = "provguard",
    [string]$Model = "qwen3-8k",
    [string]$Policies = "",
    [int]$Repeat = 0,
    [int]$MaxAttempts = 20,
    # Default: the first three injection tasks of the suite (Slack numbers them from 1).
    [string[]]$InjectionTasks = @()
)
if (-not $InjectionTasks) {
    $first = if ($Suite -eq "slack") { 1 } else { 0 }
    $InjectionTasks = $first..($first + 2) | ForEach-Object { "injection_task_$_" }
}
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
$py = Join-Path $root ".venv\Scripts\python.exe"

# Same naming as bench/run.py: non-default policies get their own label.
$label = $Defense
$extra = @()
if ($Policies -and $Defense -eq "provguard") {
    $label = "provguard-" + (Split-Path $Policies -Leaf)
    $extra = @("--policies", $Policies)
}
$result = Join-Path $root "results\$Suite\$Model-nothink\$label-r$Repeat.json"
$log = Join-Path $root "overnight-$Suite-$label.log"

Add-Content $log "injection tasks: $($InjectionTasks -join ', ')"
for ($i = 1; $i -le $MaxAttempts -and -not (Test-Path $result); $i++) {
    Add-Content $log "attempt $i started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    & $py -m bench.run --suite $Suite --defense $Defense --no-think --model $Model --repeats ($Repeat + 1) `
        --injection-tasks @InjectionTasks @extra *>> $log
    Add-Content $log "attempt $i exited with $LASTEXITCODE at $(Get-Date -Format 'HH:mm:ss')"
}
Add-Content $log "finished: $(Test-Path $result)"
