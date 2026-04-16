# Rust环境搭建与测试指南

本文档指导您完成Rust环境搭建和Python-Rust通信测试。

## 环境要求

### 必需软件

1. **Rust工具链**
   - Rust 1.70.0 或更高版本
   - Cargo（Rust包管理器）

2. **Python环境**
   - Python 3.8 或更高版本
   - pip 包管理器

3. **构建工具**
   - Microsoft C++ Build Tools（Windows）
   - 或 GCC/Clang（Linux/macOS）

## 安装步骤

### 1. 安装Rust工具链

#### Windows系统
```powershell
# 方法一：使用官方安装器（推荐）
# 下载地址：https://rustup.rs/
# 运行下载的rustup-init.exe

# 方法二：使用PowerShell
Invoke-WebRequest -Uri https://win.rustup.rs/ -OutFile rustup-init.exe
.\rustup-init.exe

# 重启终端使环境变量生效
```

#### 验证安装
```powershell
rustc --version
cargo --version
```

### 2. 安装Python依赖

```powershell
# 升级pip
python -m pip install --upgrade pip

# 安装maturin（PyO3构建工具）
pip install maturin

# 安装其他依赖
pip install numpy
```

### 3. 编译Rust扩展

#### 开发模式（推荐用于开发）
```powershell
# 进入Rust项目目录
cd rust

# 开发模式编译（支持快速迭代）
maturin develop
```

#### 发布模式（推荐用于生产）
```powershell
# 发布模式编译（性能优化）
maturin build --release

# 安装编译好的wheel包
pip install target/wheels/groundingdino_rust-*.whl
```

## 测试步骤

### 1. 基本功能测试

```powershell
# 运行集成测试脚本
python test_rust_integration.py
```

### 2. 手动测试

```python
# 在Python中手动测试
from groundingdino_rust import hello_rust, test_computation, test_array_sum

# 测试字符串处理
print(hello_rust("测试"))

# 测试数值计算
result = test_computation(10.0, 5.0)
print(f"计算结果: {result}")

# 测试数组操作
import numpy as np
values = [1.0, 2.0, 3.0, 4.0, 5.0]
sum_result = test_array_sum(values)
print(f"数组求和: {sum_result}")
```

## 常见问题解决

### 问题1：Rust编译失败

**错误信息**：
```
error: linker `link.exe` not found
```

**解决方案**：
```powershell
# 安装Microsoft C++ Build Tools
# 下载地址：https://visualstudio.microsoft.com/visual-cpp/
# 选择 "Desktop development with C++" 工作负载
```

### 问题2：Python导入失败

**错误信息**：
```
ImportError: DLL load failed
```

**解决方案**：
```powershell
# 重新编译Rust扩展
cd rust
cargo clean
maturin develop --release
```

### 问题3：NumPy版本不兼容

**错误信息**：
```
ImportError: numpy.core.multiarray failed to import
```

**解决方案**：
```powershell
# 升级NumPy到兼容版本
pip install --upgrade numpy
```

### 问题4：权限错误

**错误信息**：
```
Permission denied: 'rust'
```

**解决方案**：
```powershell
# 以管理员身份运行PowerShell
# 或修改目录权限
icacls rust /grant Everyone:F /T
```

## 开发工作流

### 日常开发流程

1. **修改Rust代码**
   ```powershell
   # 编辑 rust/src/lib.rs
   notepad rust\src\lib.rs
   ```

2. **重新编译**
   ```powershell
   cd rust
   maturin develop  # 开发模式，快速编译
   ```

3. **Python测试**
   ```python
   # 立即测试修改
   from groundingdino_rust import your_function
   your_function()
   ```

4. **迭代优化**
   - 根据测试结果调整代码
   - 重复步骤1-3

### 性能优化流程

1. **Release编译**
   ```powershell
   cd rust
   maturin build --release
   ```

2. **性能测试**
   ```python
   import time
   from groundingdino_rust import your_function
   
   start = time.time()
   for _ in range(1000):
       your_function()
   end = time.time()
   
   print(f"平均耗时: {(end-start)/1000:.6f}秒")
   ```

3. **优化调整**
   - 根据性能测试结果优化算法
   - 使用Rust性能分析工具
   - 调整编译优化选项

## 项目结构说明

```
rust/
├── Cargo.toml              # Rust项目配置
├── src/
│   └── lib.rs          # Rust源代码（当前包含测试函数）
└── target/              # 编译输出目录
    ├── debug/           # 开发模式输出
    └── release/         # 发布模式输出
```

## 下一步

完成环境搭建和测试后，您可以：

1. **开始功能开发**
   - 在`rust/src/lib.rs`中添加新函数
   - 参考现有函数的格式

2. **性能优化**
   - 使用Release模式编译
   - 进行性能基准测试

3. **集成到Grounding DINO**
   - 替换Python中的对应函数
   - 保持API兼容性

## 技术支持

如遇到问题，请检查：

1. **Rust环境**：`rustc --version` 和 `cargo --version`
2. **Python环境**：`python --version` 和 `pip list`
3. **编译日志**：查看详细的编译错误信息
4. **测试输出**：运行`test_rust_integration.py`查看详细错误

祝您开发顺利！