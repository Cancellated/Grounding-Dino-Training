"""
Rust扩展测试脚本
用于验证Python-Rust通信是否正常
"""
import sys
import traceback

def test_basic_import():
    """测试基本导入"""
    try:
        from groundingdino_rust import hello_rust, test_computation, test_array_sum
        print("✅ Rust扩展导入成功")
        return True
    except ImportError as e:
        print(f"❌ Rust扩展导入失败: {e}")
        print("请确保已运行: maturin develop")
        return False
    except Exception as e:
        print(f"❌ 未知错误: {e}")
        traceback.print_exc()
        return False

def test_hello_function():
    """测试hello函数"""
    try:
        from groundingdino_rust import hello_rust
        result = hello_rust("Grounding DINO")
        print(f"✅ hello_rust函数测试通过: {result}")
        return True
    except Exception as e:
        print(f"❌ hello_rust函数测试失败: {e}")
        traceback.print_exc()
        return False

def test_computation():
    """测试数值计算"""
    try:
        from groundingdino_rust import test_computation
        result = test_computation(10.0, 5.0)
        expected = 10.0 * 5.0 + 42.0
        if abs(result - expected) < 0.001:
            print(f"✅ 数值计算测试通过: {result} (期望: {expected})")
            return True
        else:
            print(f"❌ 数值计算结果不匹配: {result} (期望: {expected})")
            return False
    except Exception as e:
        print(f"❌ 数值计算测试失败: {e}")
        traceback.print_exc()
        return False

def test_array_operations():
    """测试数组操作"""
    try:
        from groundingdino_rust import test_array_sum
        import numpy as np
        
        test_values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = test_array_sum(test_values)
        expected = sum(test_values)
        
        if abs(result - expected) < 0.001:
            print(f"✅ 数组操作测试通过: {result} (期望: {expected})")
            return True
        else:
            print(f"❌ 数组操作结果不匹配: {result} (期望: {expected})")
            return False
    except Exception as e:
        print(f"❌ 数组操作测试失败: {e}")
        traceback.print_exc()
        return False

def test_numpy_integration():
    """测试NumPy数组集成"""
    try:
        from groundingdino_rust import test_array_sum
        import numpy as np
        
        # 创建NumPy数组
        np_array = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        
        # 转换为Python列表传递给Rust
        values = np_array.tolist()
        result = test_array_sum(values)
        
        expected = 15.0
        if abs(result - expected) < 0.001:
            print(f"✅ NumPy集成测试通过: {result}")
            return True
        else:
            print(f"❌ NumPy集成结果不匹配: {result} (期望: {expected})")
            return False
    except Exception as e:
        print(f"❌ NumPy集成测试失败: {e}")
        traceback.print_exc()
        return False

def test_performance():
    """性能测试"""
    try:
        from groundingdino_rust import test_array_sum
        import time
        import numpy as np
        
        # 创建大型数组进行性能测试
        large_array = list(range(10000))
        
        start_time = time.time()
        for _ in range(100):
            test_array_sum(large_array)
        end_time = time.time()
        
        elapsed = end_time - start_time
        print(f"✅ 性能测试完成: 100次迭代耗时 {elapsed:.4f}秒")
        print(f"   平均每次调用: {elapsed/100:.6f}秒")
        return True
    except Exception as e:
        print(f"❌ 性能测试失败: {e}")
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("=" * 60)
    print("Grounding DINO Rust扩展通信测试")
    print("=" * 60)
    print()
    
    results = []
    
    # 依次执行所有测试
    tests = [
        ("基本导入测试", test_basic_import),
        ("Hello函数测试", test_hello_function),
        ("数值计算测试", test_computation),
        ("数组操作测试", test_array_operations),
        ("NumPy集成测试", test_numpy_integration),
        ("性能测试", test_performance),
    ]
    
    for test_name, test_func in tests:
        print(f"🧪 执行: {test_name}")
        result = test_func()
        results.append((test_name, result))
        print()
    
    # 输出测试结果摘要
    print("=" * 60)
    print("测试结果摘要")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
    
    print()
    print(f"总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print()
        print("🎉 所有测试通过！Python-Rust通信正常！")
        return 0
    else:
        print()
        print("⚠️  部分测试失败，请检查错误信息")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)