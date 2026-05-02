use pyo3::prelude::*;

mod tensor_ops;
mod memory_pool;
mod parallel;
mod utils;

/// Grounding DINO Rust扩展模块，从python到rust的接口
#[pymodule]
fn groundingdino_rust(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // 注册工具函数
    m.add_function(wrap_pyfunction!(utils::test_function, m)?)?;
    m.add_function(wrap_pyfunction!(utils::is_cuda_available, m)?)?;
    
    // 注册张量操作优化函数
    m.add_function(wrap_pyfunction!(tensor_ops::optimized_zeros, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_ops::optimized_ones, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_ops::optimized_to, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_ops::optimized_eq, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_ops::optimized_bitwise_or, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_ops::optimized_nonzero, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_ops::batch_zeros, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_ops::optimized_fill, m)?)?;
    
    // 注册内存管理函数
    m.add_function(wrap_pyfunction!(memory_pool::clear_memory_pool, m)?)?;
    
    // 注册并行处理函数
    m.add_function(wrap_pyfunction!(parallel::parallel_process_batch, m)?)?;
    
    Ok(())
}
