# Grounding DINO 训练项目

## 项目简介

Grounding DINO 是一个强大的计算机视觉模型，用于图像检测和分割。本项目提供了一个基于 Tkinter 的用户界面，支持 CPU 和 GPU 两种运行模式，方便用户使用 Grounding DINO 模型进行图像分析和训练。

## 快速开始

### 启动 UI 界面

我们提供了统一的启动脚本，支持在 UI 界面中选择 CPU 或 GPU 模式：

#### PowerShell 版本（推荐）
```powershell
cd d:\Projects\Grounding-Dino-Training
.\launch_ui.ps1
```

#### Batch 版本
```cmd
cd d:\Projects\Grounding-Dino-Training
launch_ui.bat
```

## 文档索引

详细的文档请查看 [doc](doc/) 文件夹：

- **[环境配置指南](doc/环境配置指南.md)** - CPU 和 GPU 环境配置步骤
- **[训练步骤指南](doc/训练步骤指南.md)** - 训练流程说明
- **[自动标注原理与操作流程](doc/自动标注原理与操作流程文档.md)** - 自动标注功能的原理和操作说明
- **[必要文件清单](doc/必要文件清单.md)** - 项目文件结构说明和打包指南

## 项目结构

```
Grounding-Dino-Training/
├── doc/                          # 项目文档
│   ├── 环境配置指南.md
│   ├── 训练步骤指南.md
│   ├── 自动标注原理与操作流程文档.md
│   ├── 必要文件清单.md
│   └── create_package.ps1        # 项目打包脚本
├── GroundingDINO/                 # Grounding DINO 核心代码
│   ├── groundingdino/            # 核心模块
│   ├── config/                   # 配置文件
│   ├── weights/                  # 模型权重（需单独下载）
│   ├── requirements.txt          # Python 依赖
│   └── environment.yaml          # Conda 环境配置
├── Images/                       # 训练图像数据
├── venv_groundingdino/           # Python 虚拟环境
├── launch_ui.ps1                 # UI 启动脚本（PowerShell）
└── launch_ui.bat                 # UI 启动脚本（Batch）
```

## 环境要求

### 最低配置（CPU 模式）
- 操作系统：Windows 10/11 或 Linux (Ubuntu 22.04+)
- Python：3.9 或更高版本
- 内存：至少 8GB RAM
- 磁盘空间：至少 10GB 可用空间

### 推荐配置（GPU 模式）
- 操作系统：Windows 10/11 或 Linux (Ubuntu 18.04+)
- Python：3.9 或更高版本
- 内存：至少 16GB RAM
- 磁盘空间：至少 20GB 可用空间
- GPU：NVIDIA GPU（支持 CUDA 11.7 或更高版本）
- CUDA：11.7 或更高版本
- cuDNN：8.9 或更高版本

## 常见问题

### 首次使用
首次运行时，系统会尝试从 Hugging Face Hub 下载 `bert-base-uncased` 模型。若本地已缓存该模型，无需重新下载。

### 环境配置问题
详细的配置步骤和常见问题解决请参考 [环境配置指南](doc/环境配置指南.md)。

### 训练相关问题
详细的训练步骤和参数说明请参考 [训练步骤指南](doc/训练步骤指南.md)。


