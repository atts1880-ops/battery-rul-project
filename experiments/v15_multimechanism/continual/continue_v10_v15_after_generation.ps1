$ErrorActionPreference = "Stop"

$workspace = "D:\basilisk-2.11.0"
$modelRoot = Join-Path $workspace "battery_tcn_lstm_reproduction"
$trainingPython = "D:\miniconda3\envs\rul_gpu\python.exe"
$generationManifest = Join-Path $workspace "battery_target_domain\output\v1.5_incremental_microdomains\formal\microdomain_manifest.json"
$statusPath = Join-Path $modelRoot "bhump_v10_v15_continual_pipeline\pipeline_status.json"
$deadline = (Get-Date).AddHours(24)

function Write-PipelineStatus([string]$stage, [string]$state) {
    [ordered]@{
        stage = $stage
        state = $state
        updated_at = (Get-Date).ToString("o")
        process_id = $PID
    } | ConvertTo-Json | Set-Content -LiteralPath $statusPath -Encoding UTF8
}

New-Item -ItemType Directory -Force -Path (Split-Path $statusPath) | Out-Null
Write-PipelineStatus "waiting_for_microdomain_generation" "running"
while (-not (Test-Path -LiteralPath $generationManifest)) {
    if ((Get-Date) -ge $deadline) {
        Write-PipelineStatus "failed" "Timed out waiting 24 hours for microdomain generation"
        throw "Timed out waiting for $generationManifest"
    }
    Start-Sleep -Seconds 10
}

Push-Location $modelRoot
try {
    Write-PipelineStatus "extract_frozen_features" "running"
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
    Write-PipelineStatus "complete" "complete"
}
catch {
    Write-PipelineStatus "failed" $_.Exception.Message
    throw
}
finally {
    Pop-Location
}
