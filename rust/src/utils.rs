use pyo3::prelude::*;

/// 简单的测试函数
#[pyfunction]
pub fn test_function() -> PyResult<String> {
    Ok("Rust扩展测试成功!".to_string())
}

/// 检查CUDA是否可用（通过Python API）
#[pyfunction]
pub fn is_cuda_available() -> PyResult<bool> {
    Python::with_gil(|py| {
        let torch = py.import("torch")?;
        let cuda = torch.getattr("cuda")?;
        let is_available = cuda.getattr("is_available")?;
        let result = is_available.call0()?;
        Ok(result.is_truthy()?)
    })
}
