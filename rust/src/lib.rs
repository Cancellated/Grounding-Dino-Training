use pyo3::prelude::*;

/// 简单的测试函数，用于验证Python-Rust通信
#[pyfunction]
pub fn hello_rust(name: &str) -> PyResult<String> {
    Ok(format!("Hello from Rust, {}!", name))
}

/// 测试数值计算
#[pyfunction]
pub fn test_computation(a: f64, b: f64) -> PyResult<f64> {
    Ok(a * b + 42.0)
}

/// 测试数组操作
#[pyfunction]
pub fn test_array_sum(values: Vec<f64>) -> PyResult<f64> {
    let sum: f64 = values.iter().sum();
    Ok(sum)
}

/// Grounding DINO Rust扩展模块
#[pymodule]
fn groundingdino_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello_rust, m)?)?;
    m.add_function(wrap_pyfunction!(test_computation, m)?)?;
    m.add_function(wrap_pyfunction!(test_array_sum, m)?)?;
    Ok(())
}