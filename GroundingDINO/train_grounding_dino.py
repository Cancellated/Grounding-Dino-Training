import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from groundingdino.models.GroundingDINO.groundingdino import build_groundingdino
from groundingdino.datasets.cocogrounding_eval import COCOGroundingDataset, CocoGroundingEvaluator
from groundingdino.datasets.transforms import get_transform
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.misc import setup_for_distributed, nested_tensor_from_tensor_list
from groundingdino.util.logger import setup_logger
from groundingdino.util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou, box_iou
from groundingdino.models.GroundingDINO.utils import sigmoid_focal_loss
import time
import logging


def parse_args():
    """解析命令行参数
    
    Returns:
        argparse.Namespace: 包含所有命令行参数的命名空间
    """
    parser = argparse.ArgumentParser(description="Grounding DINO 单GPU训练脚本")
    parser.add_argument("--config", type=str, default="groundingdino/config/GroundingDINO_SwinT_OGC.py", help="模型配置文件路径")
    parser.add_argument("--config-file", type=str, default="groundingdino/config/GroundingDINO_SwinT_OGC.py", help="模型配置文件路径（别名）")
    parser.add_argument("--checkpoint", type=str, default="weights/groundingdino_swint_ogc.pth", help="预训练权重路径")
    parser.add_argument("--data-dir", type=str, default="demo/coco_dataset", help="数据集根目录")
    parser.add_argument("--train_dataset", type=str, default=None, help="训练数据集路径")
    parser.add_argument("--train_images", type=str, default=None, help="训练图像文件夹路径")
    parser.add_argument("--val_dataset", type=str, default=None, help="验证数据集路径")
    parser.add_argument("--val_images", type=str, default=None, help="验证图像文件夹路径")
    parser.add_argument("--output_dir", type=str, default="outputs", help="输出目录")
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录（别名）")
    parser.add_argument("--batch_size", type=int, default=1, help="批量大小")
    parser.add_argument("--batch-size", type=int, default=None, help="批量大小（别名）")
    parser.add_argument("--epochs", type=int, default=25, help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-5, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--warmup_steps", type=int, default=500, help="学习率预热步数")
    parser.add_argument("--log_interval", type=int, default=10, help="日志输出间隔")
    parser.add_argument("--save_interval", type=int, default=1, help="模型保存间隔")
    parser.add_argument("--max_train_samples", type=int, default=None, help="最大训练样本数量")
    args = parser.parse_args()
    
    # 处理参数别名
    if args.config_file != "groundingdino/config/GroundingDINO_SwinT_OGC.py":
        args.config = args.config_file
    if args.output_dir is not None:
        args.output_dir = args.output_dir
    if args.batch_size is not None:
        args.batch_size = args.batch_size
    
    # 从data-dir构建路径
    if args.train_dataset is None:
        args.train_dataset = os.path.join(args.data_dir, "data.json")
    if args.train_images is None:
        args.train_images = os.path.join(args.data_dir, "data")
    if args.val_dataset is None:
        args.val_dataset = os.path.join(args.data_dir, "data.json")
    if args.val_images is None:
        args.val_images = os.path.join(args.data_dir, "data")
    
    return args


def build_model(config_path, checkpoint_path, device):
    """构建模型
    
    Args:
        config_path (str): 配置文件路径
        checkpoint_path (str): 预训练权重路径
        device (torch.device): 设备
    
    Returns:
        nn.Module: 构建好的模型
    """
    # 加载配置
    config = SLConfig.fromfile(config_path)
    
    # 使用build_groundingdino函数初始化模型
    model = build_groundingdino(config)
    
    # 加载预训练权重
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=False)
    
    # 将模型移动到设备
    model.to(device)
    
    return model


def build_dataloader(dataset_path, images_path, batch_size, transform, shuffle=True, max_samples=None):
    """构建数据加载器
    
    Args:
        dataset_path (str): 数据集路径
        images_path (str): 图像文件夹路径
        batch_size (int): 批量大小
        transform (callable): 数据转换函数
        shuffle (bool): 是否打乱数据
        max_samples (int): 最大样本数量，用于限制训练数据量
    
    Returns:
        DataLoader: 数据加载器
    """
    dataset = COCOGroundingDataset(
        ann_file=dataset_path, 
        img_folder=images_path,
        transforms=transform,
        return_masks=False,
        return_tokens=True
    )
    
    # 如果指定了最大样本数量，则截取数据集
    if max_samples is not None and max_samples < len(dataset):
        from torch.utils.data import Subset
        indices = list(range(max_samples))
        dataset = Subset(dataset, indices)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
        collate_fn=COCOGroundingDataset.collate_fn
    )
    
    return dataloader


def compute_loss(outputs, targets, num_boxes):
    """计算损失
    
    Args:
        outputs (dict): 模型输出，包含pred_logits和pred_boxes
        targets (list): 目标列表
        num_boxes (int): 批次中的边界框数量
    
    Returns:
        dict: 损失字典，包含loss_ce和loss_bbox
    """
    pred_logits = outputs["pred_logits"]
    pred_boxes = outputs["pred_boxes"]
    
    batch_size = pred_logits.shape[0]
    num_queries = pred_logits.shape[1]
    
    # 调试信息
    logging.info(f"Debug: pred_logits shape: {pred_logits.shape}")
    logging.info(f"Debug: num_boxes: {num_boxes}")
    
    # 准备目标
    target_boxes = []
    
    for target in targets:
        boxes = target["boxes"]
        
        # 将边界框转换为xyxy格式
        if boxes.shape[-1] == 4:
            boxes = box_cxcywh_to_xyxy(boxes)
        
        target_boxes.append(boxes)
    
    # 计算分类损失
    # Grounding DINO使用对比学习，pred_logits是查询与文本token的相似度
    # 我们需要计算正样本（与目标匹配的查询）的相似度损失
    loss_ce = 0.0
    for i in range(batch_size):
        num_target_boxes = len(target_boxes[i])
        logging.info(f"Debug: batch {i}, num_target_boxes: {num_target_boxes}")
        
        if num_target_boxes > 0:
            # 对于每个目标框，选择最匹配的查询
            # 使用pred_boxes和target_boxes之间的IoU来匹配
            pred_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes[i])
            
            # 计算IoU矩阵
            iou_matrix, _ = box_iou(pred_boxes_xyxy, target_boxes[i])
            
            # 为每个目标框选择IoU最大的查询
            max_ious, matched_queries = iou_matrix.max(dim=0)
            
            # 计算分类损失
            # 对于匹配的查询，我们希望它们与文本token有高相似度
            # 对于未匹配的查询，我们希望它们与文本token有低相似度
            
            # 计算每个查询的最大相似度
            max_logits = pred_logits[i].max(dim=-1)[0]
            
            # 创建目标：匹配的查询应该有高相似度
            target_sim = torch.zeros(num_queries, device=pred_logits.device)
            target_sim[matched_queries] = 1.0
            
            # 使用二元交叉熵损失
            loss_ce_single = F.binary_cross_entropy_with_logits(
                max_logits,
                target_sim,
                reduction='mean'
            )
            logging.info(f"Debug: loss_ce_single: {loss_ce_single}")
            loss_ce += loss_ce_single
    
    loss_ce = loss_ce / batch_size
    
    # 计算边界框损失
    loss_bbox = 0.0
    for i in range(batch_size):
        num_target_boxes = len(target_boxes[i])
        if num_target_boxes > 0:
            # 将预测框转换为xyxy格式
            pred_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes[i])
            
            # 计算IoU矩阵
            iou_matrix, _ = box_iou(pred_boxes_xyxy, target_boxes[i])
            
            # 为每个目标框选择IoU最大的查询
            max_ious, matched_queries = iou_matrix.max(dim=0)
            
            # 计算GIoU损失
            for j, query_idx in enumerate(matched_queries):
                giou = generalized_box_iou(
                    pred_boxes_xyxy[query_idx:query_idx+1],
                    target_boxes[i][j:j+1]
                )
                loss_bbox += (1 - giou).mean()
    
    loss_bbox = loss_bbox / batch_size
    
    return {
        "loss_ce": loss_ce,
        "loss_bbox": loss_bbox
    }


def train_one_epoch(model, dataloader, optimizer, scheduler, epoch, args, device):
    """训练一个轮次
    
    Args:
        model (nn.Module): 模型
        dataloader (DataLoader): 数据加载器
        optimizer (Optimizer): 优化器
        scheduler (LRScheduler): 学习率调度器
        epoch (int): 当前轮次
        args (argparse.Namespace): 命令行参数
        device (torch.device): 设备
    
    Returns:
        float: 平均损失
    """
    model.train()
    total_loss = 0.0
    start_time = time.time()
    
    for step, batch in enumerate(dataloader):
        # 将数据移动到设备
        images = batch["images"].to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in batch["targets"]]
        captions = batch["captions"]
        
        # 将captions添加到targets中
        for i, target in enumerate(targets):
            target["caption"] = captions[i]
        
        # 将图像转换为NestedTensor
        nested_images = nested_tensor_from_tensor_list(images)
        
        # 前向传播
        outputs = model(nested_images, targets=targets)
        
        # 计算损失
        num_boxes = sum(len(t["boxes"]) for t in targets)
        loss_dict = compute_loss(outputs, targets, num_boxes)
        
        # 计算总损失
        losses = sum(loss_dict.values())
        total_loss += losses.item()
        
        # 反向传播
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        scheduler.step()
        
        # 输出日志
        if step % args.log_interval == 0:
            logging.info(f"Epoch [{epoch}/{args.epochs}], Step [{step}/{len(dataloader)}], Loss: {losses.item():.4f}, Loss_ce: {loss_dict['loss_ce'].item():.4f}, Loss_bbox: {loss_dict['loss_bbox'].item():.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
    
    avg_loss = total_loss / len(dataloader)
    elapsed_time = time.time() - start_time
    logging.info(f"Epoch [{epoch}/{args.epochs}] Completed, Average Loss: {avg_loss:.4f}, Time: {elapsed_time:.2f}s")
    
    return avg_loss


def validate(model, dataloader, epoch, args, device):
    """验证模型
    
    Args:
        model (nn.Module): 模型
        dataloader (DataLoader): 数据加载器
        epoch (int): 当前轮次
        args (argparse.Namespace): 命令行参数
        device (torch.device): 设备
    
    Returns:
        float: 平均损失
    """
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            # 将数据移动到设备
            images = batch["images"].to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in batch["targets"]]
            captions = batch["captions"]
            
            # 将captions添加到targets中
            for i, target in enumerate(targets):
                target["caption"] = captions[i]
            
            # 将图像转换为NestedTensor
            nested_images = nested_tensor_from_tensor_list(images)
            
            # 前向传播
            outputs = model(nested_images, targets=targets)
            
            # 计算损失
            num_boxes = sum(len(t["boxes"]) for t in targets)
            loss_dict = compute_loss(outputs, targets, num_boxes)
            
            # 计算总损失
            losses = sum(loss_dict.values())
            total_loss += losses.item()
    
    avg_loss = total_loss / len(dataloader)
    logging.info(f"Validation Epoch [{epoch}/{args.epochs}], Average Loss: {avg_loss:.4f}")
    
    return avg_loss


def save_checkpoint(model, optimizer, scheduler, epoch, args, filename="checkpoint.pth"):
    """保存检查点
    
    Args:
        model (nn.Module): 模型
        optimizer (Optimizer): 优化器
        scheduler (LRScheduler): 学习率调度器
        epoch (int): 当前轮次
        args (argparse.Namespace): 命令行参数
        filename (str): 文件名
    """
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "args": args
    }
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 保存检查点
    checkpoint_path = os.path.join(args.output_dir, filename)
    torch.save(checkpoint, checkpoint_path)
    logging.info(f"Checkpoint saved to {checkpoint_path}")


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 设置日志
    setup_logger(args.output_dir)
    
    # 配置日志立即输出到控制台
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    
    # 检查设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # 构建模型
    logging.info(f"Building model from {args.config}")
    model = build_model(args.config, args.checkpoint, device)
    logging.info(f"Model built successfully")
    
    # 构建数据转换
    logging.info(f"Building data transforms")
    train_transform = get_transform("train")
    val_transform = get_transform("val")
    
    # 构建数据加载器
    logging.info(f"Building dataloaders")
    logging.info(f"Train dataset: {args.train_dataset}")
    logging.info(f"Train images: {args.train_images}")
    if args.max_train_samples is not None:
        logging.info(f"Max train samples: {args.max_train_samples}")
    train_dataloader = build_dataloader(args.train_dataset, args.train_images, args.batch_size, train_transform, shuffle=True, max_samples=args.max_train_samples)
    logging.info(f"Train dataloader built: {len(train_dataloader)} batches")
    
    val_dataloader = build_dataloader(args.val_dataset, args.val_images, args.batch_size, val_transform, shuffle=False)
    logging.info(f"Val dataloader built: {len(val_dataloader)} batches")
    
    # 构建优化器
    logging.info(f"Building optimizer with lr={args.lr}, weight_decay={args.weight_decay}")
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # 构建学习率调度器
    total_steps = args.epochs * len(train_dataloader)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
    logging.info(f"Total training steps: {total_steps}")
    
    # 训练循环
    logging.info(f"Starting training for {args.epochs} epochs")
    logging.info(f"Batch size: {args.batch_size}")
    logging.info(f"Log interval: {args.log_interval}")
    logging.info("=" * 50)
    
    for epoch in range(1, args.epochs + 1):
        logging.info(f"Starting epoch {epoch}/{args.epochs}")
        
        # 训练一个轮次
        train_loss = train_one_epoch(model, train_dataloader, optimizer, scheduler, epoch, args, device)
        
        # 验证
        logging.info(f"Starting validation for epoch {epoch}")
        val_loss = validate(model, val_dataloader, epoch, args, device)
        
        # 保存检查点
        if epoch % args.save_interval == 0:
            logging.info(f"Saving checkpoint for epoch {epoch}")
            save_checkpoint(model, optimizer, scheduler, epoch, args, f"checkpoint_epoch_{epoch}.pth")
        
        logging.info("=" * 50)
    
    # 保存最终模型
    logging.info("Saving final model")
    save_checkpoint(model, optimizer, scheduler, args.epochs, args, "final_model.pth")
    logging.info("Training completed!")


if __name__ == "__main__":
    main()