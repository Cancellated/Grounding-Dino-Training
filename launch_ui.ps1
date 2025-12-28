# 统一启动Grounding DINO UI脚本
# 支持CPU和GPU模式选择
# 使用PowerShell以获得更好的兼容性和UTF-8支持

# 强制设置脚本编码为UTF-8
#Requires -Version 5.1

# 更全面的编码设置以解决中文乱码问题
$OutputEncoding = [System.Text.UTF8Encoding]::new()
[System.Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
[System.Console]::InputEncoding = [System.Text.UTF8Encoding]::new()
$PSDefaultParameterValues['Out-File:Encoding'] = 'utf8'
$PSDefaultParameterValues['Write-Output:Encoding'] = 'utf8'

Write-Host "正在启动Grounding DINO UI..." -ForegroundColor Green

# 切换到项目根目录
$projectRoot = "d:\Projects\Grounding-Dino-Training"
Set-Location -Path $projectRoot

# 检查并激活虚拟环境
Write-Host "正在激活虚拟环境..." -ForegroundColor Green
$venvPath = "venv_groundingdino\Scripts\Activate.ps1"
if (Test-Path -Path $venvPath) {
    & $venvPath
} else {
    Write-Host "错误：虚拟环境不存在，请先创建虚拟环境。" -ForegroundColor Red
    Read-Host -Prompt "按Enter键退出"
    exit 1
}

# 启动UI程序（UI程序已移至项目根目录）
Write-Host "正在启动UI程序..." -ForegroundColor Green
Write-Host "请在界面中选择要使用的设备(CPU/GPU)" -ForegroundColor Yellow
python grounding_dino_ui.py

# 检查是否成功启动
if ($LASTEXITCODE -ne 0) {
    Write-Host "错误：UI程序启动失败，请检查错误信息。" -ForegroundColor Red
    Read-Host -Prompt "按Enter键退出"
    exit 1
}
