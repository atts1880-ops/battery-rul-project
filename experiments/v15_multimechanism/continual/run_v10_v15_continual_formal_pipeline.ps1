$ErrorActionPreference = "Stop"

$workspace = "D:\basilisk-2.11.0"
$targetDomain = Join-Path $workspace "battery_target_domain"
$modelRoot = Join-Path $workspace "battery_tcn_lstm_reproduction"
$basiliskPython = Join-Path $workspace "basilisk-2.11.0\.venv\Scripts\python.exe"
$trainingPython = "D:\miniconda3\envs\rul_gpu\python.exe"
$statusPath = Join-Path $modelRoot "bhump_v10_v15_continual_pipeline\pipeline_status.json"

function Write-PipelineStatus([string]$stage, [string]$state) {
    $status = [ordered]@{
        stage = $stage
        state = $state
        updated_at = (Get-Date).ToString("o")
        process_id = $PID
    }
    $status | ConvertTo-Json | Set-Content -LiteralPath $statusPath -Encoding UTF8
}

New-Item -ItemType Directory -Force -Path (Split-Path $statusPath) | Out-Null

try {
    Write-PipelineStatus "generate_microdomains" "running"
    Push-Location $targetDomain
    & $basiliskPython ".\generate_battery_v15_microdomains.py" `
        --formal --workers 4 --resume
    if ($LASTEXITCODE -ne 0) { throw "Microdomain generation failed with exit code $LASTEXITCODE" }
    Pop-Location

    Write-PipelineStatus "extract_frozen_features" "running"
    Push-Location $modelRoot
    & $trainingPython ".\prepare_bhump_v15_microdomains.py" --mode formal
    if ($LASTEXITCODE -ne 0) { throw "Feature extraction failed with exit code $LASTEXITCODE" }

    Write-PipelineStatus "continual_cuda_training" "running"
    & $trainingPython ".\train_bhump_v10_v15_continual.py" `
        --parent-nasa-dir ".\bhump_v10_nasa_dynamics_full320_runs" `
        --parent-target-dir ".\bhump_v10_full320_target_control_runs" `
        --v15-train-units 40 `
        --microdomains "knee_spectrum,thermal_load,decoupled_aging,path_nonstationary" `
        --v10-anchor-weight 0.5 `
        --methods "inherited_ft,inherited_mldg,inherited_mldg_groupdro" `
        --seeds "52,53,54" `
        --swad-best `
        --device cuda `
        --resume
    if ($LASTEXITCODE -ne 0) { throw "Continual training failed with exit code $LASTEXITCODE" }
    Pop-Location

    Write-PipelineStatus "complete" "complete"
}
catch {
    Write-PipelineStatus "failed" $_.Exception.Message
    throw
}
finally {
    while ((Get-Location).Path -ne $workspace -and (Get-Location).Path -ne $modelRoot) {
        Pop-Location -ErrorAction SilentlyContinue
    }
}
