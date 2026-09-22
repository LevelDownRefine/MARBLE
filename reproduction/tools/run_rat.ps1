param([string]$OutputDirectory = "")
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
. .\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = Join-Path (Get-Location) "src"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"
$env:MPLBACKEND = "Agg"
$env:MPLCONFIGDIR = Join-Path (Get-Location) ".cache\matplotlib"
$env:OMP_NUM_THREADS = "4"
$env:OPENBLAS_NUM_THREADS = "4"
$env:MKL_NUM_THREADS = "4"
$env:NUMBA_NUM_THREADS = "4"
if ($OutputDirectory -eq "") {
    $OutputDirectory = Join-Path (Get-Location) "results\rat-$(Get-Date -Format yyyyMMdd-HHmmss)"
}
if (Test-Path -LiteralPath $OutputDirectory) { throw "Output directory already exists" }
New-Item -ItemType Directory -Path $OutputDirectory | Out-Null
python -u src/rat_decoding.py --output $OutputDirectory 2>&1 |
    Tee-Object -FilePath (Join-Path $OutputDirectory "run.log")
if ($LASTEXITCODE -ne 0) { throw "Decoding failed; inspect run.log" }
