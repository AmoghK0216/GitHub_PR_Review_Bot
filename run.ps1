$ErrorActionPreference = "Stop"

Set-Location -Path $PSScriptRoot

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    Write-Error "Virtual environment not found at .venv. Create it first: python -m venv .venv"
}

& .\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8080 --reload
