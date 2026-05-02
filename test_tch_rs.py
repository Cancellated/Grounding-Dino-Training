import groundingdino_rust

print("=== tch-rs 实现测试 ===")
funcs = sorted([x for x in dir(groundingdino_rust) if not x.startswith('_')])
print(f"可用函数 ({len(funcs)}个):")
for f in funcs:
    print(f"  - {f}")

print()
print("测试 optimized_zeros (tch-rs实现):")
result = groundingdino_rust.optimized_zeros([2, 3], "float32", "cpu")
print(f"  结果: {result}")

print()
print("测试 optimized_ones:")
result = groundingdino_rust.optimized_ones([2, 2], "float32", "cpu")
print(f"  结果: {result}")

print()
print("测试 batch_zeros:")
results = groundingdino_rust.batch_zeros([[2,3], [4,5]], "float32", "cpu")
print(f"  结果数量: {len(results)}")

print()
print("✅ 所有tch-rs函数测试通过!")
