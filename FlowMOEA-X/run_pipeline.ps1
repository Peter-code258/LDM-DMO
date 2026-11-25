<#
.SYNOPSIS
    FlowMOEA-X v4.0 Windows 一键全自动化流程运行脚本。

.DESCRIPTION
    依次执行：
      1. 环境与硬件检测（Python / PyTorch / CUDA）
      2. 全量 pytest 验收测试（可用 -SkipTests 跳过）
      3. 全张量批量动态多目标优化基准实验 + Wilcoxon/Holm + Friedman/Nemenyi 统计检验 + 自包含 HTML 报告编译
      4. 严格函数评估账本（FE Ledger）审计与 Friedman 排行榜输出
      5. 可选自动在浏览器打开 HTML 报告或启动 Streamlit 交互式流形工作台

.EXAMPLE
    .\run_pipeline.ps1
    .\run_pipeline.ps1 -Mode standard -OpenReport
    .\run_pipeline.ps1 -Mode paper -Device cuda -SkipTests
    .\run_pipeline.ps1 -Mode quick -LaunchDashboard
#>
[CmdletBinding()]
param(
    [ValidateSet("quick", "standard", "paper")]
    [string]$Mode = "quick",

    [string]$Config = "",

    [ValidateSet("auto", "cpu", "cuda", "mps")]
    [string]$Device = "auto",

    [string]$OutputDir = "",

    [switch]$SkipTests,

    [switch]$OpenReport,

    [switch]$LaunchDashboard
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $ProjectRoot

try {
    $LogDir = Join-Path $ProjectRoot "results\logs"
    if (-not (Test-Path $LogDir)) {
        New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
    }
    $Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $LogFile = Join-Path $LogDir "pipeline_${Mode}_${Timestamp}.log"

    Write-Host "[FlowMOEA-X] Starting automated pipeline (Mode=$Mode, Device=$Device)" -ForegroundColor Cyan
    Write-Host "[FlowMOEA-X] Log file: $LogFile" -ForegroundColor DarkGray

    $pyArgs = @("scripts/run_all.py", "--mode", $Mode, "--device", $Device)
    if ($Config -ne "")       { $pyArgs += @("--config", $Config) }
    if ($OutputDir -ne "")    { $pyArgs += @("--output-dir", $OutputDir) }
    if ($SkipTests)           { $pyArgs += "--skip-tests" }
    if ($OpenReport)          { $pyArgs += "--open-report" }
    if ($LaunchDashboard)     { $pyArgs += "--launch-dashboard" }

    & python @pyArgs 2>&1 | Tee-Object -FilePath $LogFile
    if ($LASTEXITCODE -ne 0) {
        throw "Pipeline failed with exit code $LASTEXITCODE. See log: $LogFile"
    }
    Write-Host "[FlowMOEA-X] Pipeline completed cleanly. Log saved to $LogFile" -ForegroundColor Green
}
finally {
    Pop-Location
}
