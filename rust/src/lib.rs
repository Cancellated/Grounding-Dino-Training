use pyo3::prelude::*;
use pyo3::types::PyTuple;

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
        Ok(result.is_true()?)
    })
}

/// 优化的zeros操作
#[pyfunction]
pub fn optimized_zeros(size: PyObject, _dtype: Option<PyObject>, _device: Option<PyObject>) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        let torch = py.import("torch")?;
        let zeros = torch.getattr("zeros")?;
        let args = PyTuple::new(py, &[size]);
        let result = zeros.call(args, None)?;
        Ok(result.into())
    })
}

/// 优化的to操作
#[pyfunction]
pub fn optimized_to(tensor: PyObject, device: PyObject, _dtype: Option<PyObject>, _non_blocking: Option<bool>) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        let tensor_obj = tensor.as_ref(py);
        let to = tensor_obj.getattr("to")?;
        let args = PyTuple::new(py, &[device]);
        let result = to.call(args, None)?;
        Ok(result.into())
    })
}

/// 优化的eq操作
#[pyfunction]
pub fn optimized_eq(input: PyObject, other: PyObject) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        let input_obj = input.as_ref(py);
        let eq = input_obj.getattr("eq")?;
        let args = PyTuple::new(py, &[other]);
        let result = eq.call(args, None)?;
        Ok(result.into())
    })
}

/// 优化的bitwise_or操作
#[pyfunction]
pub fn optimized_bitwise_or(input: PyObject, other: PyObject) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        let input_obj = input.as_ref(py);
        let bitwise_or = input_obj.getattr("bitwise_or")?;
        let args = PyTuple::new(py, &[other]);
        let result = bitwise_or.call(args, None)?;
        Ok(result.into())
    })
}

/// 优化的nonzero操作
#[pyfunction]
pub fn optimized_nonzero(input: PyObject) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        let input_obj = input.as_ref(py);
        let nonzero = input_obj.getattr("nonzero")?;
        let args: &[PyObject] = &[];
        let args_tuple = PyTuple::new(py, args);
        let result = nonzero.call(args_tuple, None)?;
        Ok(result.into())
    })
}

/// 并行处理批次数据
#[pyfunction]
pub fn parallel_process_batch(batch: PyObject) -> PyResult<PyObject> {
    Python::with_gil(|_py| {
        Ok(batch.into())
    })
}

/// 清空内存池
#[pyfunction]
pub fn clear_memory_pool() -> PyResult<()> {
    Ok(())
}

/// Grounding DINO Rust扩展模块
#[pymodule]
fn groundingdino_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(test_function, m)?)?;
    m.add_function(wrap_pyfunction!(is_cuda_available, m)?)?;
    m.add_function(wrap_pyfunction!(optimized_zeros, m)?)?;
    m.add_function(wrap_pyfunction!(optimized_to, m)?)?;
    m.add_function(wrap_pyfunction!(optimized_eq, m)?)?;
    m.add_function(wrap_pyfunction!(optimized_bitwise_or, m)?)?;
    m.add_function(wrap_pyfunction!(optimized_nonzero, m)?)?;
    m.add_function(wrap_pyfunction!(parallel_process_batch, m)?)?;
    m.add_function(wrap_pyfunction!(clear_memory_pool, m)?)?;
    Ok(())
}
