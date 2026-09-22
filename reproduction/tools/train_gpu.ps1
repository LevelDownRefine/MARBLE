param([string]$OutputDirectory = '')
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUNBUFFERED = '1'
$env:MPLBACKEND = 'Agg'
$env:MPLCONFIGDIR = Join-Path (Get-Location) '.cache/matplotlib'
$env:OMP_NUM_THREADS = '4'
$env:OPENBLAS_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
$env:NUMBA_NUM_THREADS = '4'
if ($OutputDirectory -eq '') {
    $OutputDirectory = Join-Path (Get-Location) "results/train-gpu-$(Get-Date -Format yyyyMMdd-HHmmss)"
}
if (Test-Path -LiteralPath $OutputDirectory) { throw 'Choose a new output directory' }
& .venv/Scripts/python.exe -u src/prepare_gpu_training.py --output $OutputDirectory
if ($LASTEXITCODE -ne 0) { throw 'Original-code preprocessing/sampling failed' }
& gpu/.venv/Scripts/python.exe -u src/modern_gpu.py --directory $OutputDirectory 2>&1 |
    Tee-Object -FilePath (Join-Path $OutputDirectory 'gpu.log')
if ($LASTEXITCODE -ne 0) { throw 'GPU validation/training failed; inspect gpu.log' }
& .venv/Scripts/python.exe -u src/prepare_gpu_training.py --output $OutputDirectory --finish
if ($LASTEXITCODE -ne 0) { throw 'Decoding/original-code cross-check failed' }
