param([string]$Python = 'python')
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:UV_CACHE_DIR = Join-Path (Get-Location) '.uv-cache'
$env:UV_LINK_MODE = 'hardlink'
uv sync --project gpu --python $Python --locked
if ($LASTEXITCODE -ne 0) { throw 'GPU dependency installation failed' }
# Original graph construction remains in the existing CPU environment.
uv pip check --python gpu/.venv/Scripts/python.exe
if ($LASTEXITCODE -ne 0) { throw 'GPU dependency verification failed' }
