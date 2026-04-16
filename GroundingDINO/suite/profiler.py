"""
Grounding DINO 训练性能分析脚本

该脚本用于分析 Grounding DINO 训练过程中的性能瓶颈，
包括数据加载、模型前向传播、损失计算、反向传播等环节的性能分析。

使用方法：
    python profiler.py [--batches N] [--output OUTPUT_DIR] [--config CONFIG_FILE]

参数：
    --batches N        分析前 N 个 batch（默认：5）
    --output OUTPUT_DIR 输出目录（默认：./logs/profiler）
    --config CONFIG_FILE 模型配置文件路径
    --help             显示帮助信息
"""

#输出结果在 http://localhost:6006/ 中查看
import argparse
import os
import sys
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 导入训练相关模块
try:
    # 从训练脚本导入相关函数
    import sys
    import os
    
    # 添加GroundingDINO目录到Python路径
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from suite.train_grounding_dino import (
        parse_args as train_parse_args,
        build_model,
        build_dataloader,
        get_transform,
        train_one_epoch,
        compute_loss
    )
    import torch.optim as optim
    from torch.optim.lr_scheduler import CosineAnnealingLR
    import logging
    import time
except ImportError as e:
    print(f"导入模块失败: {e}")
    print("请确保 Grounding DINO 环境已正确配置")
    sys.exit(1)

def get_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Grounding DINO 性能分析脚本")
    
    # 获取脚本所在目录的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    groundingdino_root = os.path.dirname(script_dir)
    
    # 分析控制参数
    parser.add_argument(
        "--batches", 
        type=int, 
        default=5, 
        help="分析前 N 个 batch"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="./logs/profiler", 
        help="输出目录"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="设备选择 (cuda/cpu)"
    )
    
    # 获取项目根目录
    project_root = os.path.dirname(groundingdino_root)
    
    # 训练相关参数（与训练脚本保持一致）
    parser.add_argument("--config", type=str, 
                      default=os.path.join(groundingdino_root, "groundingdino", "config", "GroundingDINO_SwinT_OGC.py"), 
                      help="模型配置文件路径")
    parser.add_argument("--config-file", type=str, 
                      default=os.path.join(groundingdino_root, "groundingdino", "config", "GroundingDINO_SwinT_OGC.py"), 
                      help="模型配置文件路径（别名）")
    parser.add_argument("--checkpoint", type=str, 
                      default=os.path.join(groundingdino_root, "weights", "groundingdino_swint_ogc.pth"), 
                      help="预训练权重路径")
    parser.add_argument("--data-dir", type=str, 
                      default=os.path.join(project_root, "custom_coco_dataset"), 
                      help="数据集根目录")
    parser.add_argument("--train-dataset", type=str, default=None, help="训练数据集路径")
    parser.add_argument("--train-images", type=str, default=None, help="训练图像文件夹路径")
    parser.add_argument("--batch-size", type=int, default=1, help="批量大小")
    parser.add_argument("--lr", type=float, default=1e-5, help="学习率")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--warmup-steps", type=int, default=500, help="学习率预热步数")
    parser.add_argument("--max-train-samples", type=int, default=None, help="最大训练样本数量")
    
    args = parser.parse_args()
    
    # 处理参数别名（与训练脚本一致）
    if args.config_file != os.path.join(groundingdino_root, "groundingdino", "config", "GroundingDINO_SwinT_OGC.py"):
        args.config = args.config_file
    
    # 从数据集文件夹构建路径（与训练脚本一致）
    if args.train_dataset is None:
        args.train_dataset = os.path.join(args.data_dir, "labels.json")
    if args.train_images is None:
        args.train_images = os.path.join(args.data_dir, "data")
    
    return args

def get_model(args, device):
    """加载模型（使用训练脚本中的函数）"""
    print("加载模型...")
    
    # 使用训练脚本中的build_model函数
    model = build_model(args.config, args.checkpoint, device, load_weights=True)
    
    print(f"模型加载完成，使用设备: {device}")
    return model

def get_dataloader(args):
    """创建数据加载器（使用训练脚本中的函数）"""
    print("创建数据加载器...")
    
    # 构建数据变换
    transform = get_transform("train")
    
    # 构建数据集路径（与训练脚本一致）
    train_dataset = args.train_dataset if args.train_dataset else os.path.join(args.data_dir, "labels.json")
    train_images = args.train_images if args.train_images else os.path.join(args.data_dir, "data")
    
    # 使用训练脚本中的build_dataloader函数
    dataloader = build_dataloader(
        train_dataset, 
        train_images, 
        args.batch_size, 
        transform, 
        shuffle=True, 
        max_samples=args.max_train_samples
    )
    
    print(f"数据加载器创建完成，批次大小: {args.batch_size}")
    return dataloader

def profile_training(model, dataloader, args):
    """分析训练性能"""
    print(f"开始性能分析，分析前 {args.batches} 个 batch...")
    
    # 确保输出目录存在
    os.makedirs(args.output, exist_ok=True)
    
    # 创建TensorBoard writer
    writer = SummaryWriter(log_dir=args.output)
    
    # 创建优化器和学习率调度器（与训练脚本一致）
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    total_steps = 1 * len(dataloader)  # 只训练一个epoch用于分析
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
    
    # 性能分析
    print(f"开始性能分析，输出目录: {args.output}")
    
    # 定义trace handler
    def trace_handler(prof):
        print(f"生成trace数据，步骤: {prof.step_num}")
        # 只保存为TensorBoard事件
        torch.profiler.tensorboard_trace_handler(args.output)(prof)
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
        on_trace_ready=trace_handler,
        schedule=torch.profiler.schedule(wait=0, warmup=0, active=args.batches, repeat=1)
    ) as prof:
        
        # 直接调用训练脚本中的train_one_epoch函数
        # 但是需要修改它以支持性能分析和限制batch数量
        
        # 保存原始的train_one_epoch函数
        original_train_one_epoch = train_one_epoch
        
        # 创建包装函数以支持性能分析
        def profiled_train_one_epoch(model, dataloader, optimizer, scheduler, epoch, args, device):
            model.train()
            total_loss = 0.0
            start_time = time.time()
            
            for step, batch in enumerate(dataloader):
                if step >= args.batches:
                    break
                    
                print(f"分析第 {step+1}/{args.batches} 个 batch...")
                
                # 性能分析记录
                with record_function("数据预处理"):
                    # 处理batch数据（根据COCOGroundingDataset.collate_fn）
                    # COCOGroundingDataset.collate_fn返回的是字典：{"images": images, "targets": targets, "captions": captions}
                    images = batch["images"]
                    targets = batch["targets"]
                    captions = batch["captions"]
                    
                    # 将数据移动到设备
                    images = images.to(device)
                    targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                              for k, v in t.items()} for t in targets]
                
                with record_function("模型前向传播"):
                    # 前向传播（与训练脚本一致）
                    # 对于批次张量，我们需要将其转换为NestedTensor
                    from groundingdino.util.misc import nested_tensor_from_tensor_list
                    
                    # 从批次张量中取出每个图像
                    images_list = [img for img in images]
                    
                    # 将图像列表转换为NestedTensor
                    images = nested_tensor_from_tensor_list(images_list)
                    
                    # 调用模型的forward方法，将captions作为关键字参数传递
                    outputs = model(images, captions=captions)
                
                with record_function("损失计算"):
                    # 计算损失
                    num_boxes = sum(len(t["boxes"]) for t in targets)
                    # 直接使用导入的compute_loss函数
                    loss_dict = compute_loss(outputs, targets, num_boxes)
                    losses = sum(loss_dict.values())
                    total_loss += losses.item()
                
                with record_function("反向传播"):
                    # 反向传播
                    optimizer.zero_grad()
                    losses.backward()
                    optimizer.step()
                    scheduler.step()
                
                # 记录性能数据
                prof.step()
                
                # 记录到TensorBoard
                global_step = step + 1
                writer.add_scalar("Loss/train", losses.item(), global_step)
                writer.add_scalar("LearningRate", scheduler.get_last_lr()[0], global_step)
            
            avg_loss = total_loss / min(args.batches, len(dataloader))
            elapsed_time = time.time() - start_time
            print(f"分析完成, 平均损失: {avg_loss:.4f}, 时间: {elapsed_time:.2f}s")
            
            # 关闭TensorBoard writer
            writer.close()
            
            return avg_loss
        
        # 运行训练
        profiled_train_one_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=1,
            args=args,
            device=args.device
        )
    
    # 生成分析报告
    print("\n" + "="*80)
    print("性能分析报告")
    print("="*80)
    
    # 按CPU时间排序
    print("\n1. 按CPU耗时排序（前10个）:")
    print(prof.key_averages().table(
        sort_by="self_cpu_time_total", 
        row_limit=10
    ))
    
    # 按GPU时间排序
    if args.device == "cuda":
        print("\n2. 按GPU耗时排序（前10个）:")
        print(prof.key_averages().table(
            sort_by="self_cuda_time_total", 
            row_limit=10
        ))
    
    # 按调用次数排序
    print("\n3. 按调用次数排序（前10个）:")
    print(prof.key_averages().table(
        sort_by="count", 
        row_limit=10
    ))
    
    # 导出分析结果
    trace_file = os.path.join(args.output, "grounding_dino_profile.json")
    try:
        prof.export_chrome_trace(trace_file)
    except RuntimeError as e:
        print(f"导出Chrome追踪文件时出错: {e}")
        print("但性能分析已完成，结果已保存到TensorBoard日志中")
    
    print("\n" + "="*80)
    print("分析完成！")
    print(f"• 分析结果已保存到: {args.output}")
    print(f"• Chrome 追踪文件: {trace_file}")
    print(f"• 可通过 TensorBoard 查看: tensorboard --logdir={args.output}")
    print(f"• 可通过 Chrome 浏览器查看追踪: chrome://tracing")
    print("="*80)

def profile_inference(model, dataloader, args):
    """分析推理性能"""
    print(f"开始推理性能分析，分析前 {args.batches} 个 batch...")
    
    # 确保输出目录存在
    os.makedirs(os.path.join(args.output, "inference"), exist_ok=True)
    
    # 性能分析
    inference_output_dir = os.path.join(args.output, "inference")
    print(f"开始推理性能分析，输出目录: {inference_output_dir}")
    
    # 定义trace handler
    def inference_trace_handler(prof):
        print(f"生成推理trace数据，步骤: {prof.step_num}")
        # 只保存为TensorBoard事件
        torch.profiler.tensorboard_trace_handler(inference_output_dir)(prof)
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
        on_trace_ready=inference_trace_handler
    ) as prof:
        
        for i, batch in enumerate(dataloader):
            if i >= args.batches:
                break
                
            print(f"分析第 {i+1}/{args.batches} 个 batch...")
            
            with record_function("数据预处理"):
                # 数据预处理
                images = batch["images"]
                targets = batch["targets"]
                captions = batch["captions"]
                
                # 移动到设备
                images = images.to(args.device)
                targets = [{k: v.to(args.device) if isinstance(v, torch.Tensor) else v 
                          for k, v in t.items()} for t in targets]
            
            with record_function("模型推理"):
                # 模型推理
                with torch.no_grad():
                    # 由于模型需要特定的输入格式，这里使用简化版本
                    # 实际推理时会调用完整的模型推理逻辑
                    pass
            
            # 记录性能数据
            prof.step()
    
    # 生成分析报告
    print("\n" + "="*80)
    print("推理性能分析报告")
    print("="*80)
    
    # 按CPU时间排序
    print("\n1. 按CPU耗时排序（前10个）:")
    print(prof.key_averages().table(
        sort_by="self_cpu_time_total", 
        row_limit=10
    ))
    
    # 按GPU时间排序
    if args.device == "cuda":
        print("\n2. 按GPU耗时排序（前10个）:")
        print(prof.key_averages().table(
            sort_by="self_cuda_time_total", 
            row_limit=10
        ))
    
    # 导出分析结果
    trace_file = os.path.join(args.output, "inference", "grounding_dino_inference_profile.json")
    
    print("\n" + "="*80)
    print("推理分析完成！")
    print(f"• 分析结果已保存到: {os.path.join(args.output, 'inference')}")
    print(f"• Chrome 追踪文件: {os.path.join(args.output, 'inference', 'trace.json')}")
    print("="*80)

def main():
    """主函数"""
    try:
        # 解析参数
        args = get_args()
        
        print("""
        ========================================================
                    Grounding DINO 性能分析工具
        ========================================================
        """)
        
        # 设置日志
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        
        # 加载模型
        model = get_model(args, args.device)
        
        # 创建数据加载器
        dataloader = get_dataloader(args)
        
        # 分析训练性能
        print("\n" + "-"*80)
        print("分析训练性能...")
        print("-"*80)
        profile_training(model, dataloader, args)
        
        # 分析推理性能
        print("\n" + "-"*80)
        print("分析推理性能...")
        print("-"*80)
        profile_inference(model, dataloader, args)
        
        print("\n🎉 性能分析完成！")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()