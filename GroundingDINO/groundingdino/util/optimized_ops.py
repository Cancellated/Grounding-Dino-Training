"""
优化操作模块，使用tch-rs实现核心操作，并提供回退到PyTorch的机制
"""
import torch
import logging

# 尝试导入Rust扩展
try:
    import groundingdino_rust
    HAS_RUST = True
    logging.info("成功加载Rust扩展")
except ImportError:
    HAS_RUST = False
    logging.warning("Rust扩展未加载，使用PyTorch实现")


def optimized_zeros(shape, dtype=torch.float32):
    """
    优化的零张量创建
    
    Args:
        shape (tuple): 张量形状
        dtype (torch.dtype): 数据类型
    
    Returns:
        torch.Tensor: 零张量
    """
    if HAS_RUST:
        try:
            # 转换dtype为字符串
            dtype_str = str(dtype).split('.')[-1]
            # 转换shape为列表
            shape_list = list(shape)
            # 调用Rust实现
            rust_tensor = groundingdino_rust.optimized_zeros(shape_list, dtype_str)
            # 转换为PyTorch张量
            return torch.as_tensor(rust_tensor)
        except Exception as e:
            logging.warning(f"Rust实现失败，回退到PyTorch: {e}")
    
    # 回退到PyTorch实现
    return torch.zeros(shape, dtype=dtype)


def optimized_to(tensor, device, non_blocking=False):
    """
    优化的数据传输
    
    Args:
        tensor (torch.Tensor): 输入张量
        device (str): 目标设备
        non_blocking (bool): 是否非阻塞传输
    
    Returns:
        torch.Tensor: 传输后的张量
    """
    # 检查输入是否为张量，如果不是，直接返回原始值
    if not isinstance(tensor, torch.Tensor):
        return tensor
    
    if HAS_RUST:
        try:
            # 调用Rust实现
            rust_tensor = groundingdino_rust.optimized_to(tensor, device, non_blocking)
            # 转换为PyTorch张量
            return torch.as_tensor(rust_tensor)
        except Exception as e:
            logging.warning(f"Rust实现失败，回退到PyTorch: {e}")
    
    # 回退到PyTorch实现
    return tensor.to(device, non_blocking=non_blocking)


def optimized_eq(tensor, value):
    """
    优化的元素比较
    
    Args:
        tensor (torch.Tensor): 输入张量
        value (float): 比较值
    
    Returns:
        torch.Tensor: 比较结果
    """
    if HAS_RUST:
        try:
            # 调用Rust实现
            rust_tensor = groundingdino_rust.optimized_eq(tensor, value)
            # 转换为PyTorch张量
            return torch.as_tensor(rust_tensor)
        except Exception as e:
            logging.warning(f"Rust实现失败，回退到PyTorch: {e}")
    
    # 回退到PyTorch实现
    return tensor == value


def optimized_bitwise_or(tensor1, tensor2):
    """
    优化的位或操作
    
    Args:
        tensor1 (torch.Tensor): 第一个张量
        tensor2 (torch.Tensor): 第二个张量
    
    Returns:
        torch.Tensor: 位或结果
    """
    if HAS_RUST:
        try:
            # 调用Rust实现
            rust_tensor = groundingdino_rust.optimized_bitwise_or(tensor1, tensor2)
            # 转换为PyTorch张量
            return torch.as_tensor(rust_tensor)
        except Exception as e:
            logging.warning(f"Rust实现失败，回退到PyTorch: {e}")
    
    # 回退到PyTorch实现
    return tensor1 | tensor2


def optimized_nonzero(tensor):
    """
    优化的非零元素查找
    
    Args:
        tensor (torch.Tensor): 输入张量
    
    Returns:
        torch.Tensor: 非零元素的索引
    """
    if HAS_RUST:
        try:
            # 调用Rust实现
            rust_tensor = groundingdino_rust.optimized_nonzero(tensor)
            # 转换为PyTorch张量
            return torch.as_tensor(rust_tensor)
        except Exception as e:
            logging.warning(f"Rust实现失败，回退到PyTorch: {e}")
    
    # 回退到PyTorch实现
    return tensor.nonzero()


def parallel_process_batch(batch):
    """
    并行处理批次数据
    
    Args:
        batch (dict): 批次数据
    
    Returns:
        dict: 处理后的批次数据
    """
    if HAS_RUST:
        try:
            # 调用Rust实现
            rust_batch = groundingdino_rust.parallel_process_batch(batch)
            # 转换为Python字典
            result = {}
            for key in rust_batch:
                result[key] = rust_batch[key]
            return result
        except Exception as e:
            logging.warning(f"Rust实现失败，回退到Python: {e}")
    
    # 回退到Python实现
    result = {}
    result["images"] = batch["images"]
    result["targets"] = []
    result["captions"] = batch["captions"]
    
    for i, target in enumerate(batch["targets"]):
        new_target = target.copy()
        new_target["caption"] = batch["captions"][i]
        result["targets"].append(new_target)
    
    return result


def is_cuda_available():
    """
    检查CUDA是否可用
    
    Returns:
        bool: CUDA是否可用
    """
    if HAS_RUST:
        try:
            return groundingdino_rust.is_cuda_available()
        except Exception as e:
            logging.warning(f"Rust实现失败，回退到PyTorch: {e}")
    
    # 回退到PyTorch实现
    return torch.cuda.is_available()


def clear_memory_pool():
    """
    清空内存池
    """
    if HAS_RUST:
        try:
            groundingdino_rust.clear_memory_pool()
        except Exception as e:
            logging.warning(f"清空内存池失败: {e}")


# 导出所有函数
__all__ = [
    "optimized_zeros",
    "optimized_to",
    "optimized_eq",
    "optimized_bitwise_or",
    "optimized_nonzero",
    "parallel_process_batch",
    "is_cuda_available",
    "clear_memory_pool",
    "HAS_RUST"
]
