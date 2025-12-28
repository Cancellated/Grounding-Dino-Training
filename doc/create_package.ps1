# Grounding DINO 项目打包脚本
# 支持三种打包模式：minimal、standard、full

param(
    [ValidateSet("minimal", "standard", "full")]
    [string]$PackageType = "minimal",
    
    [string]$OutputPath = ".\GroundingDINO_Package",
    
    [switch]$Force
)

# 设置编码为UTF-8
$OutputEncoding = [System.Text.UTF8Encoding]::new()
[System.Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()

# 项目根目录
$ProjectRoot = "d:\Projects\Grounding-Dino-Training"

# 检查项目根目录是否存在
if (-not (Test-Path -Path $ProjectRoot)) {
    Write-Host "错误：项目根目录不存在: $ProjectRoot" -ForegroundColor Red
    exit 1
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Grounding DINO 项目打包工具" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 显示打包模式信息
Write-Host "打包模式: $PackageType" -ForegroundColor Green
switch ($PackageType) {
    "minimal" {
        Write-Host "  - 包含核心代码和配置文件" -ForegroundColor Yellow
        Write-Host "  - 不包含预训练模型权重（约1.7GB）" -ForegroundColor Yellow
        Write-Host "  - 不包含Hugging Face缓存（约400MB）" -ForegroundColor Yellow
        Write-Host "  - 预估大小: 50-100 MB" -ForegroundColor Yellow
    }
    "standard" {
        Write-Host "  - 包含核心代码和配置文件" -ForegroundColor Yellow
        Write-Host "  - 包含Hugging Face缓存（约400MB）" -ForegroundColor Yellow
        Write-Host "  - 不包含预训练模型权重（约1.7GB）" -ForegroundColor Yellow
        Write-Host "  - 预估大小: 450-600 MB" -ForegroundColor Yellow
    }
    "full" {
        Write-Host "  - 包含核心代码和配置文件" -ForegroundColor Yellow
        Write-Host "  - 包含Hugging Face缓存（约400MB）" -ForegroundColor Yellow
        Write-Host "  - 包含预训练模型权重（约1.7GB）" -ForegroundColor Yellow
        Write-Host "  - 预估大小: 2.1-2.3 GB" -ForegroundColor Yellow
    }
}
Write-Host ""

# 检查输出目录
if (Test-Path -Path $OutputPath) {
    if ($Force) {
        Write-Host "警告：输出目录已存在，将删除并重新创建" -ForegroundColor Yellow
        Remove-Item -Path $OutputPath -Recurse -Force
    } else {
        Write-Host "错误：输出目录已存在: $OutputPath" -ForegroundColor Red
        Write-Host "使用 -Force 参数强制覆盖" -ForegroundColor Yellow
        exit 1
    }
}

# 创建输出目录
Write-Host "创建输出目录: $OutputPath" -ForegroundColor Green
New-Item -Path $OutputPath -ItemType Directory | Out-Null

# 定义要复制的文件和目录
$filesToCopy = @()

# 1. 根目录文档文件
$filesToCopy += @{
    Source = "$ProjectRoot\环境配置指南.md"
    Destination = "$OutputPath\环境配置指南.md"
}

$filesToCopy += @{
    Source = "$ProjectRoot\README_启动说明.md"
    Destination = "$OutputPath\README_启动说明.md"
}

$filesToCopy += @{
    Source = "$ProjectRoot\训练步骤指南.md"
    Destination = "$OutputPath\训练步骤指南.md"
}

$filesToCopy += @{
    Source = "$ProjectRoot\自动标注原理与操作流程文档.md"
    Destination = "$OutputPath\自动标注原理与操作流程文档.md"
}

$filesToCopy += @{
    Source = "$ProjectRoot\必要文件清单.md"
    Destination = "$OutputPath\必要文件清单.md"
}

# 2. 启动脚本
$filesToCopy += @{
    Source = "$ProjectRoot\launch_ui.bat"
    Destination = "$OutputPath\launch_ui.bat"
}

$filesToCopy += @{
    Source = "$ProjectRoot\launch_ui.ps1"
    Destination = "$OutputPath\launch_ui.ps1"
}

# 3. GroundingDINO 核心文件
$filesToCopy += @{
    Source = "$ProjectRoot\GroundingDINO\requirements.txt"
    Destination = "$OutputPath\GroundingDINO\requirements.txt"
}

$filesToCopy += @{
    Source = "$ProjectRoot\GroundingDINO\environment.yaml"
    Destination = "$OutputPath\GroundingDINO\environment.yaml"
}

$filesToCopy += @{
    Source = "$ProjectRoot\GroundingDINO\setup.py"
    Destination = "$OutputPath\GroundingDINO\setup.py"
}

$filesToCopy += @{
    Source = "$ProjectRoot\GroundingDINO\README.md"
    Destination = "$OutputPath\GroundingDINO\README.md"
}

# 4. GroundingDINO 核心代码目录
$directoriesToCopy = @(
    "$ProjectRoot\GroundingDINO\groundingdino",
    "$ProjectRoot\GroundingDINO\demo"
)

# 5. 训练脚本
$filesToCopy += @{
    Source = "$ProjectRoot\GroundingDINO\train_grounding_dino.py"
    Destination = "$OutputPath\GroundingDINO\train_grounding_dino.py"
}

$filesToCopy += @{
    Source = "$ProjectRoot\GroundingDINO\verify_training_environment.py"
    Destination = "$OutputPath\GroundingDINO\verify_training_environment.py"
}

# 6. UI 程序（已移至项目根目录）
$filesToCopy += @{
    Source = "$ProjectRoot\grounding_dino_ui.py"
    Destination = "$OutputPath\grounding_dino_ui.py"
}

# 根据打包模式添加额外文件
if ($PackageType -eq "standard" -or $PackageType -eq "full") {
    # 添加 Hugging Face 缓存
    $filesToCopy += @{
        Source = "$ProjectRoot\huggingface_cache"
        Destination = "$OutputPath\huggingface_cache"
        IsDirectory = $true
    }
}

if ($PackageType -eq "full") {
    # 添加预训练模型权重
    $weightsPath = "$ProjectRoot\GroundingDINO\weights\groundingdino_swint_ogc.pth"
    if (Test-Path -Path $weightsPath) {
        $filesToCopy += @{
            Source = $weightsPath
            Destination = "$OutputPath\GroundingDINO\weights\groundingdino_swint_ogc.pth"
        }
    } else {
        Write-Host "警告：预训练模型权重文件不存在: $weightsPath" -ForegroundColor Yellow
        Write-Host "将跳过此文件" -ForegroundColor Yellow
    }
}

# 开始复制文件
Write-Host ""
Write-Host "开始复制文件..." -ForegroundColor Green
Write-Host ""

$copiedFiles = 0
$failedFiles = 0

foreach ($item in $filesToCopy) {
    $source = $item.Source
    $destination = $item.Destination
    $isDirectory = $item.IsDirectory

    if (-not (Test-Path -Path $source)) {
        Write-Host "  [跳过] 源文件不存在: $source" -ForegroundColor Red
        $failedFiles++
        continue
    }

    try {
        # 创建目标目录
        $destDir = Split-Path -Path $destination -Parent
        if (-not (Test-Path -Path $destDir)) {
            New-Item -Path $destDir -ItemType Directory -Force | Out-Null
        }

        if ($isDirectory) {
            # 复制目录
            Write-Host "  [目录] $(Split-Path $source -Leaf)" -ForegroundColor Cyan
            Copy-Item -Path $source -Destination $destination -Recurse -Force
        } else {
            # 复制文件
            Write-Host "  [文件] $(Split-Path $source -Leaf)" -ForegroundColor Cyan
            Copy-Item -Path $source -Destination $destination -Force
        }
        $copiedFiles++
    } catch {
        Write-Host "  [错误] 复制失败: $source" -ForegroundColor Red
        Write-Host "         错误信息: $($_.Exception.Message)" -ForegroundColor Red
        $failedFiles++
    }
}

# 复制目录
foreach ($dir in $directoriesToCopy) {
    if (-not (Test-Path -Path $dir)) {
        Write-Host "  [跳过] 源目录不存在: $dir" -ForegroundColor Red
        $failedFiles++
        continue
    }

    try {
        $destDir = $dir -replace [regex]::Escape($ProjectRoot), $OutputPath
        Write-Host "  [目录] $(Split-Path $dir -Parent)\$(Split-Path $dir -Leaf)" -ForegroundColor Cyan
        Copy-Item -Path $dir -Destination $destDir -Recurse -Force
        $copiedFiles++
    } catch {
        Write-Host "  [错误] 复制失败: $dir" -ForegroundColor Red
        Write-Host "         错误信息: $($_.Exception.Message)" -ForegroundColor Red
        $failedFiles++
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "打包完成！" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "复制成功: $copiedFiles" -ForegroundColor Green
if ($failedFiles -gt 0) {
    Write-Host "复制失败: $failedFiles" -ForegroundColor Red
}
Write-Host ""
Write-Host "输出目录: $OutputPath" -ForegroundColor Yellow
Write-Host ""

# 计算目录大小
try {
    $size = (Get-ChildItem -Path $OutputPath -Recurse -File | Measure-Object -Property Length -Sum).Sum
    $sizeMB = [math]::Round($size / 1MB, 2)
    $sizeGB = [math]::Round($size / 1GB, 2)
    
    if ($sizeGB -ge 1) {
        Write-Host "目录大小: $sizeGB GB" -ForegroundColor Yellow
    } else {
        Write-Host "目录大小: $sizeMB MB" -ForegroundColor Yellow
    }
} catch {
    Write-Host "无法计算目录大小" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "下一步操作：" -ForegroundColor Cyan
Write-Host "1. 检查输出目录中的文件" -ForegroundColor White
Write-Host "2. 压缩目录（可选）：" -ForegroundColor White
Write-Host "   Compress-Archive -Path '$OutputPath' -DestinationPath '$OutputPath.zip'" -ForegroundColor Gray
Write-Host "3. 发送给新成员" -ForegroundColor White
Write-Host ""

if ($PackageType -ne "full") {
    Write-Host "注意：" -ForegroundColor Yellow
    Write-Host "- 新成员需要下载预训练模型权重" -ForegroundColor White
    Write-Host "- 下载地址: https://github.com/IDEA-Research/GroundingDINO/releases" -ForegroundColor Gray
    Write-Host "- 文件名: groundingdino_swint_ogc.pth" -ForegroundColor Gray
    Write-Host "- 放置位置: GroundingDINO/weights/" -ForegroundColor Gray
    Write-Host ""
}

if ($PackageType -eq "minimal") {
    Write-Host "注意：" -ForegroundColor Yellow
    Write-Host "- 新成员首次运行时需要下载BERT模型" -ForegroundColor White
    Write-Host "- 可以手动下载或让程序自动下载" -ForegroundColor Gray
    Write-Host ""
}

Write-Host "新成员使用步骤：" -ForegroundColor Cyan
Write-Host "1. 解压压缩包" -ForegroundColor White
Write-Host "2. 阅读 '环境配置指南.md' 配置环境" -ForegroundColor White
Write-Host "3. 阅读 'README_启动说明.md' 启动项目" -ForegroundColor White
Write-Host ""
