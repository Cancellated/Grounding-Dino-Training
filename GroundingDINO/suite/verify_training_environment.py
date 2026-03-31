#!/usr/bin/env python3
"""Grounding DINO训练环境验证脚本

该脚本用于验证训练环境的所有必要组件是否正确配置，包括：
1. CUDA和PyTorch的兼容性
2. 必要的依赖项是否已安装
3. 模型配置和权重文件是否存在
4. GPU是否可用
5. 基本的模型加载功能是否正常
"""

import torch
import importlib
import os
import sys

def check_pytorch_cuda():
    """检查PyTorch和CUDA配置
    
    Returns:
        dict: 包含PyTorch和CUDA信息的字典
    """
    print("=== 检查PyTorch和CUDA配置 ===")
    
    result = {
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else "N/A",
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
    }
    
    print(f"PyTorch版本: {result['pytorch_version']}")
    print(f"CUDA可用: {result['cuda_available']}")
    print(f"CUDA版本: {result['cuda_version']}")
    print(f"cuDNN版本: {result['cudnn_version']}")
    print(f"GPU名称: {result['gpu_name']}")
    print(f"GPU数量: {result['gpu_count']}")
    
    return result

def check_dependencies():
    """检查必要的依赖项是否已安装
    
    Returns:
        dict: 包含依赖项检查结果的字典
    """
    print("\n=== 检查必要的依赖项 ===")
    
    dependencies = [
        "torch",
        "torchvision",
        "transformers",
        "addict",
        "yapf",
        "timm",
        "numpy",
        "opencv-python",
        "supervision",
        "pycocotools",
        "typer",
        "tqdm",
        "fiftyone"
    ]
    
    results = {}
    all_installed = True
    
    for dep in dependencies:
        try:
            importlib.import_module(dep)
            results[dep] = True
            print(f"✓ {dep}: 已安装")
        except ImportError:
            results[dep] = False
            all_installed = False
            print(f"✗ {dep}: 未安装")
    
    return {
        "all_installed": all_installed,
        "dependencies": results
    }

def check_model_files():
    """检查模型配置和权重文件是否存在
    
    Returns:
        dict: 包含模型文件检查结果的字典
    """
    print("\n=== 检查模型文件 ===")
    
    config_files = [
        "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        "GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py"
    ]
    
    weight_files = [
        "GroundingDINO/weights/groundingdino_swint_ogc.pth"
    ]
    
    results = {
        "config_files": {},
        "weight_files": {}
    }
    
    # 获取项目根目录（suite的父目录）
    project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
    
    for config_file in config_files:
        full_path = os.path.join(project_root, config_file)
        exists = os.path.exists(full_path)
        results["config_files"][config_file] = exists
        print(f"{'✓' if exists else '✗'} 配置文件: {config_file} {'(存在)' if exists else '(不存在)'}")
    
    for weight_file in weight_files:
        full_path = os.path.join(project_root, weight_file)
        exists = os.path.exists(full_path)
        results["weight_files"][weight_file] = exists
        print(f"{'✓' if exists else '✗'} 权重文件: {weight_file} {'(存在)' if exists else '(不存在)'}")
    
    return results

def check_model_loading():
    """检查基本的模型加载功能是否正常
    
    Returns:
        dict: 包含模型加载检查结果的字典
    """
    print("\n=== 检查模型加载功能 ===")
    
    try:
        # 获取项目根目录（suite的父目录）
        project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
        
        # 确保可以导入groundingdino模块
        sys.path.append(project_root)
        from groundingdino.models.GroundingDINO.groundingdino import GroundingDINO
        from groundingdino.util.slconfig import SLConfig
        
        # 检查配置文件是否存在
        config_path = os.path.join(project_root, "GroundingDINO", "groundingdino", "config", "GroundingDINO_SwinT_OGC.py")
        if not os.path.exists(config_path):
            print(f"✗ 配置文件不存在: {config_path}")
            return {"success": False, "error": "配置文件不存在"}
        
        # 加载配置
        config = SLConfig.fromfile(config_path)
        print("✓ 配置文件加载成功")
        
        # 初始化模型（不加载权重）
        model = GroundingDINO(config)
        print("✓ 模型初始化成功")
        
        # 检查是否可以将模型移动到GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        print(f"✓ 模型成功移动到设备: {device}")
        
        return {"success": True, "message": "模型加载功能正常"}
        
    except Exception as e:
        print(f"✗ 模型加载失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

def main():
    """主函数"""
    print("Grounding DINO 训练环境验证脚本")
    print("=" * 50)
    
    # 检查当前工作目录
    print(f"当前工作目录: {os.getcwd()}")
    
    # 1. 检查PyTorch和CUDA
    pytorch_cuda_result = check_pytorch_cuda()
    
    # 2. 检查依赖项
    dependencies_result = check_dependencies()
    
    # 3. 检查模型文件
    model_files_result = check_model_files()
    
    # 4. 检查模型加载功能
    model_loading_result = check_model_loading()
    
    # 生成总结
    print("\n" + "=" * 50)
    print("=== 训练环境验证总结 ===")
    
    all_checks_passed = True
    
    # PyTorch和CUDA检查
    if pytorch_cuda_result["cuda_available"]:
        print("✓ PyTorch和CUDA配置正确")
    else:
        print("⚠ PyTorch未配置CUDA支持，将使用CPU训练")
        all_checks_passed = False
    
    # 依赖项检查
    if dependencies_result["all_installed"]:
        print("✓ 所有必要的依赖项已安装")
    else:
        print("✗ 缺少必要的依赖项，请安装缺失的包")
        all_checks_passed = False
    
    # 模型文件检查
    all_configs_exist = all(model_files_result["config_files"].values())
    all_weights_exist = all(model_files_result["weight_files"].values())
    
    if all_configs_exist:
        print("✓ 所有必要的配置文件已存在")
    else:
        print("✗ 缺少必要的配置文件")
        all_checks_passed = False
    
    if all_weights_exist:
        print("✓ 所有必要的权重文件已存在")
    else:
        print("✗ 缺少必要的权重文件")
        all_checks_passed = False
    
    # 模型加载检查
    if model_loading_result["success"]:
        print("✓ 模型加载功能正常")
    else:
        print("✗ 模型加载功能异常")
        all_checks_passed = False
    
    print("\n" + "=" * 50)
    
    if all_checks_passed:
        print("🎉 训练环境验证通过！您可以开始训练了。")
        return 0
    else:
        print("❌ 训练环境验证未通过，请修复上述问题后再开始训练。")
        return 1

if __name__ == "__main__":
    sys.exit(main())