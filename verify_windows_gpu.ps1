#Requires -Version 5.1
<#
.SYNOPSIS
    Native Windows 11 Verification & Live CUDA GPU Benchmark Launcher for Kinesis.
.DESCRIPTION
    Windows counterpart to verify_bare_metal.sh. Executes the full 24-suite
    build + verification protocol (22 host suites + G2 live CUDA GPU suite + H1 hygiene)
    and runs the empirical C++20 AVX2 CPU roofline benchmark.
#>

param(
    [string]$PythonExe = ""
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $Root

try {
    if (-not $PythonExe) {
        $cands = @(
            'C:\cuas_venv\Scripts\python.exe',
            'D:\Coding_Projects\Local model tuning\.venv\Scripts\python.exe',
            'D:\Coding_Projects\.venv\Scripts\python.exe'
        )
        foreach ($c in $cands) {
            if (Test-Path -LiteralPath $c) {
                $PythonExe = $c
                break
            }
        }
        if (-not $PythonExe) {
            $PythonExe = (Get-Command python -ErrorAction Stop).Source
        }
    }

    Write-Host "================================================================================" -ForegroundColor Cyan
    Write-Host "  KINESIS WINDOWS 11 + LIVE CUDA GPU VERIFICATION PROTOCOL" -ForegroundColor Cyan
    Write-Host "================================================================================" -ForegroundColor Cyan
    Write-Host "Python : $PythonExe"

    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        Write-Host "`n[GPU Telemetry]" -ForegroundColor Yellow
        nvidia-smi --query-gpu=index,name,driver_version,compute_cap,memory.total,clocks.sm,power.draw --format=csv
    }

    Write-Host "`n[Step 1/2] Running 24-suite master verification (including live CUDA Suite G2)..." -ForegroundColor Yellow
    & $PythonExe run_all_tests.py --gpu
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }

    Write-Host "`n[Step 2/2] Running C++20 AVX2 CPU Roofline benchmark..." -ForegroundColor Yellow
    if (Test-Path -LiteralPath '.\kinesis_unified\bench_cpu_roofline.exe') {
        & '.\kinesis_unified\bench_cpu_roofline.exe'
        if ($LASTEXITCODE -ne 0) {
            exit $LASTEXITCODE
        }
    }
} finally {
    Pop-Location
}
