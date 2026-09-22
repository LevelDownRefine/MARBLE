param([string]$Python = "python")
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:UV_CACHE_DIR = Join-Path (Get-Location) ".uv-cache"
$env:UV_LINK_MODE = "copy"
uv sync --python $Python --locked
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed" }
. .\.venv\Scripts\Activate.ps1
uv pip install --python .venv\Scripts\python.exe --no-deps --no-build-isolation --editable ..
if ($LASTEXITCODE -ne 0) { throw "MARBLE build failed; Windows requires MSVC C++ Build Tools" }
uv pip check --python .venv\Scripts\python.exe
if ($LASTEXITCODE -ne 0) { throw "Dependency verification failed" }
