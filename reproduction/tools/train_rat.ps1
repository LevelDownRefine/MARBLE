param([string]$OutputDirectory = "", [ValidateSet('cpu', 'cuda')][string]$Device = 'cuda')
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
if ($Device -eq 'cuda') {
    & (Join-Path $PSScriptRoot 'train_gpu.ps1') -OutputDirectory $OutputDirectory
    exit $LASTEXITCODE
} else {
    . .\.venv\Scripts\Activate.ps1
}
$env:PYTHONPATH = Join-Path (Get-Location) "src"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"
$env:MPLBACKEND = "Agg"
$env:MPLCONFIGDIR = Join-Path (Get-Location) ".cache\matplotlib"
$env:OMP_NUM_THREADS = "4"
$env:OPENBLAS_NUM_THREADS = "4"
$env:MKL_NUM_THREADS = "4"
$env:NUMBA_NUM_THREADS = "4"
# Upstream reads verified local checkpoints using the pre-PyTorch-2.6 API.
$env:TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD = "1"
if ($OutputDirectory -eq "") {
    $OutputDirectory = Join-Path (Get-Location) "results\train-rat-$(Get-Date -Format yyyyMMdd-HHmmss)"
}
if (Test-Path -LiteralPath $OutputDirectory) { throw "Output directory already exists" }
python -u src/train_rat.py --output $OutputDirectory --seeds 0 1 2 --device $Device
if ($LASTEXITCODE -ne 0) { throw "Training failed; inspect the result directory" }
