# Rust环境搭建与测试脚本
# 自动检查环境、安装依赖、编译Rust扩展并测试Python-Rust通信

# 强制设置脚本编码为UTF-8
#Requires -Version 5.1
$OutputEncoding = [System.Text.UTF8Encoding]::new()
[System.Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
[System.Console]::InputEncoding = [System.Text.UTF8Encoding]::new()
$PSDefaultParameterValues['Out-File:Encoding'] = 'utf8'

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Grounding DINO Rust环境搭建与测试" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# 检查函数
function Test-Command {
    param (
        [string]$Command,
        [string]$Description = "命令"
    )
    
    try {
        $parts = $Command -split ' '
        $cmd = $parts[0]
        $args = $parts[1..$parts.Length] -join ' '
        
        # 使用Get-Command检查命令是否存在
        $cmdInfo = Get-Command $cmd -ErrorAction SilentlyContinue
        
        if ($null -eq $cmdInfo) {
            Write-Host "❌ $Description 未安装" -ForegroundColor Red
            return $false
        }
        
        # 尝试运行命令获取版本信息
        $output = & $cmd $args 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ $Description 已安装" -ForegroundColor Green
            return $true
        }
        else {
            Write-Host "❌ $Description 未安装: $output" -ForegroundColor Red
            return $false
        }
    }
    catch {
        Write-Host "❌ $Description 未安装: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

# 步骤1：检查Rust环境
Write-Host "📋 步骤1: 检查Rust环境" -ForegroundColor Yellow
Write-Host "------------------------------------------------" -ForegroundColor Yellow

$rustInstalled = Test-Command "rustc --version" "Rust编译器"
$cargoInstalled = Test-Command "cargo --version" "Cargo包管理器"

if (-not $rustInstalled -or -not $cargoInstalled) {
    Write-Host ""
    Write-Host "⚠️  Rust环境未完全安装，请先安装Rust" -ForegroundColor Yellow
    Write-Host "   下载地址: https://rustup.rs/" -ForegroundColor Cyan
    Write-Host "   或运行: Invoke-WebRequest -Uri https://win.rustup.rs/ -OutFile rustup-init.exe; .\rustup-init.exe" -ForegroundColor Cyan
    Write-Host ""
    exit 1
}

Write-Host ""

# 步骤2：检查Python环境
Write-Host "📋 步骤2: 检查Python环境" -ForegroundColor Yellow
Write-Host "------------------------------------------------" -ForegroundColor Yellow

$pythonInstalled = Test-Command "python --version" "Python"
$pipInstalled = Test-Command "pip --version" "pip包管理器"

if (-not $pythonInstalled -or -not $pipInstalled) {
    Write-Host ""
    Write-Host "❌ Python环境未安装，请先安装Python 3.8+" -ForegroundColor Red
    exit 1
}

# 检查Python版本
$pythonVersion = python --version 2>&1
Write-Host "ℹ️  Python版本: $pythonVersion" -ForegroundColor Cyan

Write-Host ""

# 步骤3：检查并安装Python依赖
Write-Host "📋 步骤3: 检查Python依赖" -ForegroundColor Yellow
Write-Host "------------------------------------------------" -ForegroundColor Yellow

$requiredPackages = @(
    "maturin",
    "numpy"
)

$missingPackages = @()

foreach ($package in $requiredPackages) {
    $installed = pip show $package 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ $package 已安装" -ForegroundColor Green
    }
    else {
        Write-Host "⚠️  $package 未安装" -ForegroundColor Yellow
        $missingPackages += $package
    }
}

if ($missingPackages.Count -gt 0) {
    Write-Host ""
    Write-Host "📦 正在安装缺失的包..." -ForegroundColor Cyan
    foreach ($package in $missingPackages) {
        Write-Host "   安装 $package..." -ForegroundColor Cyan
        pip install $package
        if ($LASTEXITCODE -eq 0) {
            Write-Host "   ✅ $package 安装成功" -ForegroundColor Green
        }
        else {
            Write-Host "   ❌ $package 安装失败" -ForegroundColor Red
            exit 1
        }
    }
}

Write-Host ""

# 步骤4：检查Rust项目结构
Write-Host "📋 步骤4: 检查Rust项目结构" -ForegroundColor Yellow
Write-Host "------------------------------------------------" -ForegroundColor Yellow

$rustDir = "rust"
$cargoToml = "$rustDir\Cargo.toml"
$libRs = "$rustDir\src\lib.rs"

if (-not (Test-Path $cargoToml)) {
    Write-Host "❌ 未找到 Cargo.toml: $cargoToml" -ForegroundColor Red
    exit 1
}
else {
    Write-Host "✅ 找到 Cargo.toml" -ForegroundColor Green
}

if (-not (Test-Path $libRs)) {
    Write-Host "❌ 未找到 lib.rs: $libRs" -ForegroundColor Red
    exit 1
}
else {
    Write-Host "✅ 找到 lib.rs" -ForegroundColor Green
}

Write-Host ""

# 步骤5：检查虚拟环境
Write-Host "📋 步骤5: 检查虚拟环境" -ForegroundColor Yellow
Write-Host "------------------------------------------------" -ForegroundColor Yellow

$venvPaths = @(
    "venv",
    ".venv",
    "venv_groundingdino",
    "env"
)

$venvFound = $false
$venvPath = $null

foreach ($path in $venvPaths) {
    if (Test-Path $path) {
        $venvFound = $true
        $venvPath = $path
        break
    }
}

if ($venvFound) {
    Write-Host "✅ 找到虚拟环境: $venvPath" -ForegroundColor Green
    
    # 激活虚拟环境
    $activateScript = Join-Path $venvPath "Scripts\Activate.ps1"
    if (Test-Path $activateScript) {
        Write-Host "🔧 正在激活虚拟环境..." -ForegroundColor Cyan
        & $activateScript
        Write-Host "✅ 虚拟环境已激活" -ForegroundColor Green
    }
    else {
        Write-Host "⚠️  虚拟环境激活脚本不存在: $activateScript" -ForegroundColor Yellow
        Write-Host "   将使用 maturin build 方式编译" -ForegroundColor Yellow
        $venvFound = $false
    }
}
else {
    Write-Host "⚠️  未找到虚拟环境" -ForegroundColor Yellow
    Write-Host "   将使用 maturin build + pip install 方式编译" -ForegroundColor Yellow
}

Write-Host ""

# 步骤6：编译Rust扩展
Write-Host "📋 步骤6: 编译Rust扩展" -ForegroundColor Yellow
Write-Host "------------------------------------------------" -ForegroundColor Yellow

Set-Location $rustDir

try {
    if ($venvFound) {
        # 使用 maturin develop（需要虚拟环境）
        Write-Host "🔨 正在编译Rust扩展（开发模式）..." -ForegroundColor Cyan
        maturin develop
    }
    else {
        # 使用 maturin build + pip install（不需要虚拟环境）
        Write-Host "🔨 正在编译Rust扩展（构建模式）..." -ForegroundColor Cyan
        maturin build --release
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ Rust扩展构建成功" -ForegroundColor Green
            
            # 查找生成的wheel文件
            $wheelFiles = Get-ChildItem -Path "target\wheels" -Filter "*.whl" -ErrorAction SilentlyContinue
            
            if ($wheelFiles.Count -gt 0) {
                $latestWheel = $wheelFiles | Sort-Object LastWriteTime -Descending | Select-Object -First 1
                Write-Host "📦 正在安装wheel包: $($latestWheel.Name)" -ForegroundColor Cyan
                
                Set-Location ..
                pip install $latestWheel.FullName
                
                if ($LASTEXITCODE -eq 0) {
                    Write-Host "✅ Wheel包安装成功" -ForegroundColor Green
                }
                else {
                    Write-Host "❌ Wheel包安装失败" -ForegroundColor Red
                    exit 1
                }
            }
            else {
                Write-Host "❌ 未找到生成的wheel文件" -ForegroundColor Red
                Set-Location ..
                exit 1
            }
        }
    }
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Rust扩展编译成功" -ForegroundColor Green
    }
    else {
        Write-Host "❌ Rust扩展编译失败" -ForegroundColor Red
        Write-Host "请检查编译错误信息" -ForegroundColor Yellow
        Set-Location ..
        exit 1
    }
}
catch {
    Write-Host "❌ 编译过程出错: $($_.Exception.Message)" -ForegroundColor Red
    Set-Location ..
    exit 1
}

# 切换回项目根目录
Set-Location ..
Write-Host ""

# 步骤7：运行集成测试
Write-Host "📋 步骤7: 运行Python-Rust通信测试" -ForegroundColor Yellow
Write-Host "------------------------------------------------" -ForegroundColor Yellow

$testScript = "test_rust_integration.py"

if (-not (Test-Path $testScript)) {
    Write-Host "❌ 未找到测试脚本: $testScript" -ForegroundColor Red
    exit 1
}

Write-Host "🧪 正在运行集成测试..." -ForegroundColor Cyan
python $testScript
$testResult = $LASTEXITCODE

Write-Host ""

# 步骤8：输出测试结果
Write-Host "📋 步骤8: 测试结果摘要" -ForegroundColor Yellow
Write-Host "------------------------------------------------" -ForegroundColor Yellow

if ($testResult -eq 0) {
    Write-Host ""
    Write-Host "🎉 恭喜！所有测试通过！" -ForegroundColor Green
    Write-Host "✅ Python-Rust通信正常" -ForegroundColor Green
    Write-Host "✅ Rust环境搭建成功" -ForegroundColor Green
    Write-Host ""
    Write-Host "📚 后续步骤：" -ForegroundColor Cyan
    Write-Host "   1. 在 rust/src/lib.rs 中添加新的Rust函数" -ForegroundColor White
    Write-Host "   2. 运行 'maturin build --release' 重新编译" -ForegroundColor White
    Write-Host "   3. 使用 'pip install target/wheels/*.whl' 安装" -ForegroundColor White
    Write-Host "   4. 在Python中导入并测试新函数" -ForegroundColor White
    Write-Host "   5. 参考 doc/Rust环境搭建指南.md 进行开发" -ForegroundColor White
}
else {
    Write-Host ""
    Write-Host "⚠️  部分测试失败，请检查错误信息" -ForegroundColor Yellow
    Write-Host "💡 可能的解决方案：" -ForegroundColor Cyan
    Write-Host "   1. 检查Rust和Python版本是否兼容" -ForegroundColor White
    Write-Host "   2. 确认所有依赖已正确安装" -ForegroundColor White
    Write-Host "   3. 查看编译错误信息并修复" -ForegroundColor White
    Write-Host "   4. 参考 doc/Rust环境搭建指南.md 排查问题" -ForegroundColor White
    exit 1
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "环境搭建与测试完成" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan