# 强制设置脚本编码为UTF-8
#Requires -Version 5.1

# 更全面的编码设置以解决中文乱码问题
$OutputEncoding = [System.Text.UTF8Encoding]::new()
[System.Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
[System.Console]::InputEncoding = [System.Text.UTF8Encoding]::new()
$PSDefaultParameterValues['Out-File:Encoding'] = 'utf8'
$PSDefaultParameterValues['Write-Output:Encoding'] = 'utf8'

Write-Host "正在启动Grounding DINO Web UI..." -ForegroundColor Green

# 切换到项目根目录（脚本所在目录）
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Definition
$projectRoot = $scriptPath
Set-Location -Path $projectRoot
Write-Host "当前工作目录: $projectRoot" -ForegroundColor Green

# 检查并激活虚拟环境
Write-Host "正在激活虚拟环境..." -ForegroundColor Green
$venvPath = Join-Path $projectRoot "venv_groundingdino\Scripts\Activate.ps1"
if (Test-Path -Path $venvPath) {
    & $venvPath
} else {
    Write-Host "错误：虚拟环境不存在，请先创建虚拟环境。" -ForegroundColor Red
    Read-Host -Prompt "按Enter键退出"
    exit 1
}

# 启动Web UI程序
Write-Host "正在启动Web UI程序..." -ForegroundColor Green
Write-Host "Web UI将在浏览器中打开，默认地址: http://localhost:7860" -ForegroundColor Yellow
Write-Host "Web UI默认使用GPU模式（如果可用）" -ForegroundColor Yellow
python grounding_dino_webui.py

# 检查是否成功启动
if ($LASTEXITCODE -ne 0) {
    Write-Host "错误：Web UI程序启动失败，请检查错误信息。" -ForegroundColor Red
    Read-Host -Prompt "按Enter键退出"
    exit 1
}