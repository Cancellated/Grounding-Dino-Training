use pyo3::prelude::*;

/// 清空内存池
/// 这部分暂未实现，后续会根据需要添加
#[pyfunction]
pub fn clear_memory_pool() -> PyResult<()> {
    Ok(())
}
