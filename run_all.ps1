# Run all experiments and save output logs

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$results = Join-Path $root "results"

if (-not (Test-Path $results)) {
    New-Item -ItemType Directory -Path $results | Out-Null
}

$classificationLog = Join-Path $results "classification.txt"
$triangleLog = Join-Path $results "triangle.txt"

Write-Host "Running classification experiment..."
python (Join-Path $root "experiments\Classification_Experiment.py") *> $classificationLog

Write-Host "Running triangle inequality experiment..."
python (Join-Path $root "experiments\Triangle_Inequality_Analysis_On_Synthetic_dataser.py") *> $triangleLog

Write-Host "Done. Logs saved to:"
Write-Host "  $classificationLog"
Write-Host "  $triangleLog"
