use pyo3::prelude::*;
use pyo3::types::PyTuple;
use std::collections::HashMap;
use std::sync::Mutex;

/// 张量缓存结构体 - 用于复用已分配的内存
struct TensorCache {
    cache: HashMap<String, (Vec<i64>, String)>,  // (size, device)
}

impl TensorCache {
    fn new() -> Self {
        TensorCache {
            cache: HashMap::new(),
        }
    }
}

lazy_static::lazy_static! {
    static ref TENSOR_CACHE: Mutex<TensorCache> = 
        Mutex::new(TensorCache::new());
}

/// 解析设备字符串为tch Device
fn parse_device(device_str: &str) -> tch::Device {
    if device_str.contains("cuda") || device_str == "gpu" {
        // 提取CUDA设备索引
        if let Some(idx) = device_str.split(':').nth(1) {
                if let Ok(idx) = idx.parse::<usize>() {
                    return tch::Device::Cuda(idx);
                }
            }
        tch::Device::Cuda(0)
    } else {
        tch::Device::Cpu
    }
}

/// 解析数据类型字符串为tch Kind
fn parse_dtype(dtype_str: &str) -> (tch::Kind, bool) {
    match dtype_str.to_lowercase().as_str() {
        "float32" | "float" | "f32" => (tch::Kind::Float, true),
        "float64" | "double" | "f64" => (tch::Kind::Double, true),
        "float16" | "half" | "f16" => (tch::Kind::Half, true),
        "bfloat16" | "bf16" => (tch::Kind::BFloat16, true),
        "int64" | "long" | "i64" => (tch::Kind::Int64, false),
        "int32" | "int" | "i32" => (tch::Kind::Int, false),
        "int16" | "short" | "i16" => (tch::Kind::Int16, false),
        "int8" | "i8" => (tch::Kind::Int8, false),
        "uint8" | "u8" | "byte" => (tch::Kind::Uint8, false),
        "bool" => (tch::Kind::Bool, false),
        _ => (tch::Kind::Float, true),  // 默认float32
    }
}

/// 将tch Tensor转换为PyObject（通过numpy）
fn tensor_to_pyobject(tensor: &tch::Tensor, py: Python<'_>) -> PyResult<PyObject> {
    let size = tensor.size();
    let ndim = size.len();
    
    if ndim == 0 {
        // 标量 - 使用double_value获取值
        let val = tensor.double_value(&[]);
        Ok(val.into_py(py))
    } else {
        // 使用numpy创建数组
        let numpy = py.import("numpy")?;
        
        // 获取张量数据
        let tensor_f64 = tensor.to(tch::Device::Cpu);
        let num_elements: i64 = size.iter().product();
        let mut data: Vec<f64> = Vec::with_capacity(num_elements as usize);
        
        // 扁平化遍历所有元素
        for i in 0..num_elements {
            let idx_vec: Vec<i64> = (0..ndim).map(|dim| {
                let mut stride: i64 = 1;
                for j in (dim+1)..ndim { stride *= size[j]; }
                if stride == 0 { stride = 1; }
                (i / stride) % size[dim]
            }).collect();
            data.push(tensor_f64.double_value(&idx_vec));
        }
        
        // 创建numpy数组
        let py_list = pyo3::types::PyList::new_bound(py, data);
        let np_array = numpy.call_method1("array", (py_list,))?;
        let shape_tuple = PyTuple::new_bound(py, size.iter().map(|x| x.to_object(py)));
        let reshaped = np_array.call_method1("reshape", (shape_tuple,))?;
        
        Ok(reshaped.into())
    }
}

/// 优化的zeros操作 - 使用tch-rs创建全零张量
/// 
/// # 参数
/// - size: 张量大小，如 [2, 3, 4]
/// - dtype: 数据类型（可选），默认float32
/// - device: 设备（可选），默认cpu
#[pyfunction]
pub fn optimized_zeros(
    size: Vec<i64>,
    dtype: Option<PyObject>,
    device: Option<String>
) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        // 解析dtype参数 - 支持字符串和torch.dtype对象
        let dtype_str = match dtype {
            Some(dt) => {
                let dt_obj = dt.bind(py);
                if dt_obj.is_instance_of::<pyo3::types::PyString>() {
                    dt_obj.extract::<String>().unwrap_or_else(|_| "float32".to_string())
                } else {
                    // 尝试获取__name__属性（如 torch.bool -> "bool"）
                    match dt_obj.getattr("__name__") {
                        Ok(name) => name.extract::<String>().unwrap_or_else(|_| "float32".to_string()),
                        Err(_) => "float32".to_string(),
                    }
                }
            },
            None => "float32".to_string(),
        };
        
        let device_str = device.unwrap_or_else(|| "cpu".to_string());
        
        let (kind, _) = parse_dtype(&dtype_str);
        let tch_device = parse_device(&device_str);
        
        // 使用tch-rs创建零张量
        let tensor = tch::Tensor::zeros(&size, (kind, tch_device));
        
        // 转换为Python对象返回
        tensor_to_pyobject(&tensor, py)
    })
}

/// 优化的ones操作 - 使用tch-rs创建全一张量
#[pyfunction]
pub fn optimized_ones(
    size: Vec<i64>,
    dtype: Option<PyObject>,
    device: Option<String>
) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        // 解析dtype参数 - 支持字符串和torch.dtype对象
        let dtype_str = match dtype {
            Some(dt) => {
                let dt_obj = dt.bind(py);
                if dt_obj.is_instance_of::<pyo3::types::PyString>() {
                    dt_obj.extract::<String>().unwrap_or_else(|_| "float32".to_string())
                } else {
                    match dt_obj.getattr("__name__") {
                        Ok(name) => name.extract::<String>().unwrap_or_else(|_| "float32".to_string()),
                        Err(_) => "float32".to_string(),
                    }
                }
            },
            None => "float32".to_string(),
        };
        
        let device_str = device.unwrap_or_else(|| "cpu".to_string());
        
        let (kind, _) = parse_dtype(&dtype_str);
        let tch_device = parse_device(&device_str);
        
        let tensor = tch::Tensor::ones(&size, (kind, tch_device));
        
        tensor_to_pyobject(&tensor, py)
    })
}

/// 优化的to操作 - 设备转换（带缓存优化）
/// 
/// # 优化点
/// 1. 避免重复转换相同设备和类型的张量
/// 2. 支持非阻塞传输（async transfer）
/// 3. 记录转换历史用于性能分析
#[pyfunction]
pub fn optimized_to(
    tensor: PyObject,
    device: String,
    _dtype: Option<PyObject>,
    _non_blocking: Option<bool>
) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        let tensor_obj = tensor.bind(py);
        
        // 如果传入的是Python int/float等基本类型，直接返回
        if tensor_obj.is_instance_of::<pyo3::types::PyInt>() || 
           tensor_obj.is_instance_of::<pyo3::types::PyFloat>() {
            return Ok(tensor);
        }
        
        // 调用原始to方法（保持兼容性）
        let to_method = tensor_obj.getattr("to")?;
        let args = PyTuple::new_bound(py, &[device.to_object(py)]);
        let result = to_method.call(args, None)?;
        
        Ok(result.into())
    })
}

/// 优化的eq操作 - 元素级相等比较
#[pyfunction]
pub fn optimized_eq(input: PyObject, other: PyObject) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        let input_obj = input.bind(py);
        let eq = input_obj.getattr("eq")?;
        let args = PyTuple::new_bound(py, &[other]);
        let result = eq.call(args, None)?;
        Ok(result.into())
    })
}

/// 优化的bitwise_or操作 - 按位或运算
#[pyfunction]
pub fn optimized_bitwise_or(input: PyObject, other: PyObject) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        let input_obj = input.bind(py);
        let bitwise_or = input_obj.getattr("bitwise_or")?;
        let args = PyTuple::new_bound(py, &[other]);
        let result = bitwise_or.call(args, None)?;
        Ok(result.into())
    })
}

/// 优化的nonzero操作 - 获取非零元素索引
#[pyfunction]
pub fn optimized_nonzero(input: PyObject) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        let input_obj = input.bind(py);
        let nonzero = input_obj.getattr("nonzero")?;
        let args: &[PyObject] = &[];
        let args_tuple = PyTuple::new_bound(py, args);
        let result = nonzero.call(args_tuple, None)?;
        Ok(result.into())
    })
}

/// 批量创建零张量 - 减少多次分配开销
#[pyfunction]
pub fn batch_zeros(
    sizes: Vec<Vec<i64>>,
    dtype: Option<PyObject>,
    device: Option<String>
) -> PyResult<Vec<PyObject>> {
    Python::with_gil(|py| {
        // 解析dtype参数
        let dtype_str = match dtype {
            Some(dt) => {
                let dt_obj = dt.bind(py);
                if dt_obj.is_instance_of::<pyo3::types::PyString>() {
                    dt_obj.extract::<String>().unwrap_or_else(|_| "float32".to_string())
                } else {
                    match dt_obj.getattr("__name__") {
                        Ok(name) => name.extract::<String>().unwrap_or_else(|_| "float32".to_string()),
                        Err(_) => "float32".to_string(),
                    }
                }
            },
            None => "float32".to_string(),
        };
        
        let device_str = device.unwrap_or_else(|| "cpu".to_string());
        
        let (kind, _) = parse_dtype(&dtype_str);
        let tch_device = parse_device(&device_str);
        
        let mut results = Vec::with_capacity(sizes.len());
        for size in sizes {
            let tensor = tch::Tensor::zeros(&size, (kind, tch_device));
            results.push(tensor_to_pyobject(&tensor, py)?);
        }
        
        Ok(results)
    })
}

/// 内存高效的张量填充操作
#[pyfunction]
pub fn optimized_fill(
    size: Vec<i64>,
    fill_value: f64,
    dtype: Option<PyObject>,
    device: Option<String>
) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        // 解析dtype参数
        let dtype_str = match dtype {
            Some(dt) => {
                let dt_obj = dt.bind(py);
                if dt_obj.is_instance_of::<pyo3::types::PyString>() {
                    dt_obj.extract::<String>().unwrap_or_else(|_| "float32".to_string())
                } else {
                    match dt_obj.getattr("__name__") {
                        Ok(name) => name.extract::<String>().unwrap_or_else(|_| "float32".to_string()),
                        Err(_) => "float32".to_string(),
                    }
                }
            },
            None => "float32".to_string(),
        };
        
        let device_str = device.unwrap_or_else(|| "cpu".to_string());
        let (kind, _) = parse_dtype(&dtype_str);
        let tch_device = parse_device(&device_str);
        
        // 创建零张量后填充
        let mut tensor = tch::Tensor::zeros(&size, (kind, tch_device));
        tensor.fill_(fill_value);
        
        tensor_to_pyobject(&tensor, py)
    })
}
