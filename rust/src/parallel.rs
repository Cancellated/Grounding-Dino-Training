use pyo3::prelude::*;

/// 并行处理批次数据
/// 这部分暂未实现，后续会根据需要添加
#[pyfunction]
pub fn parallel_process_batch(batch: PyObject) -> PyResult<PyObject> {
    Python::with_gil(|_py| {
        Ok(batch.into())
    })
}
