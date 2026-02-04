# Semantic-Cluster-WebUI 环境配置脚本（Windows PowerShell）
# 用法：在项目根目录执行 .\scripts\setup_env.ps1

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectRoot

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Semantic-Cluster-WebUI 环境配置" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "项目目录: $ProjectRoot" -ForegroundColor Gray

# 1. 创建虚拟环境（若不存在）
$venvPath = Join-Path $ProjectRoot ".venv"
if (-not (Test-Path $venvPath)) {
    Write-Host "`n[1/4] 创建虚拟环境 .venv ..." -ForegroundColor Yellow
    python -m venv .venv
} else {
    Write-Host "`n[1/4] 虚拟环境 .venv 已存在，跳过。" -ForegroundColor Gray
}

# 2. 激活并升级 pip
Write-Host "`n[2/4] 激活虚拟环境并升级 pip ..." -ForegroundColor Yellow
& (Join-Path $venvPath "Scripts\Activate.ps1")
python -m pip install --upgrade pip -q

# 3. 安装依赖
Write-Host "`n[3/4] 安装 requirements.txt ..." -ForegroundColor Yellow
pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# 4. 检查 VLM 可用性
Write-Host "`n[4/4] 检查 VLM（Qwen2-VL）可用性 ..." -ForegroundColor Yellow
$vlmCheck = python -c @"
import sys
sys.path.insert(0, r'$ProjectRoot')
try:
    from models.vlm_models import is_vlm_available
    ok = is_vlm_available()
    print('OK' if ok else 'FAIL')
except Exception as e:
    print('FAIL:', e)
"@
if ($vlmCheck -match "^OK") {
    Write-Host "  VLM 可用（transformers 支持 Qwen2-VL）" -ForegroundColor Green
} else {
    Write-Host "  VLM 不可用: $vlmCheck" -ForegroundColor Red
    Write-Host "  请确认已执行: pip install -U transformers" -ForegroundColor Yellow
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "配置完成。使用前请激活虚拟环境：" -ForegroundColor Cyan
Write-Host "  .\.venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "首次使用 VLM 需下载模型（约 15GB），可执行：" -ForegroundColor Cyan
Write-Host "  huggingface-cli download Qwen/Qwen2-VL-7B-Instruct" -ForegroundColor White
Write-Host "========================================" -ForegroundColor Cyan
