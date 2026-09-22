$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
. .\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = Join-Path (Get-Location) "src"
$env:MPLBACKEND = "Agg"
$env:MPLCONFIGDIR = Join-Path (Get-Location) ".cache\matplotlib"
$env:OMP_NUM_THREADS = "4"
$env:OPENBLAS_NUM_THREADS = "4"
$env:NUMBA_NUM_THREADS = "4"
python -m unittest discover -s tests -p "test*.py"
if ($LASTEXITCODE -ne 0) { throw "Reproduction tests failed" }
ruff check src tests tools
if ($LASTEXITCODE -ne 0) { throw "Lint failed" }
python -m pytest ..\tests -q -o cache_dir=.pytest_cache
if ($LASTEXITCODE -ne 0) { throw "Upstream tests failed" }
