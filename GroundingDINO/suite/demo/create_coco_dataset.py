import typer
from groundingdino.util.inference import load_model, load_image, predict
from tqdm import tqdm
import torchvision
import torch
import fiftyone as fo

##使用GroundingDINO模型创建COCO格式数据集的脚本
def main(
        image_directory: str = 'test_grounding_dino',  # 输入图像目录
        text_prompt: str = 'bus, car',  # 文本提示
        box_threshold: float = 0.15,  # 图像框阈值
        text_threshold: float = 0.10,  # 文本阈值
        export_dataset: bool = False,   # 是否导出数据集
        export_path: str = 'coco_dataset',  # 导出数据集的路径
        view_dataset: bool = False,  # 是否查看数据集
        export_annotated_images: bool = True,  # 是否导出标注图像
        annotated_images_path: str = '../../images_with_bounding_boxes',
        weights_path : str = "../../weights/groundingdino_swint_ogc.pth",
        config_path: str = "../../groundingdino/config/GroundingDINO_SwinT_OGC.py",
        subsample: int = None,  # 子采样数量，None表示处理所有样本
    ):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    model = load_model(config_path, weights_path, device=device)
    
    # 创建数据集并显式设置media_type为image
    dataset = fo.Dataset.from_images_dir(image_directory)
    dataset.media_type = 'image'

    # 处理subsample参数
    if subsample is not None and subsample > 0:
        if subsample < len(dataset):
            # 获取前subsample个样本
            samples_to_process = list(dataset.take(subsample))
            print(f"处理 {len(samples_to_process)} 个样本")
        else:
            samples_to_process = list(dataset)
            print(f"subsample值大于等于数据集大小，处理所有 {len(samples_to_process)} 个样本")
    else:
        samples_to_process = list(dataset)
        print(f"处理所有 {len(samples_to_process)} 个样本")
    
    for sample in tqdm(samples_to_process):

        image_source, image = load_image(sample.filepath)

        boxes, logits, phrases = predict(
            model=model, 
            image=image, 
            caption=text_prompt, 
            box_threshold=box_threshold, 
            text_threshold=text_threshold,
            device=device
        )

        detections = [] 

        for box, logit, phrase in zip(boxes, logits, phrases):
            rel_box = torchvision.ops.box_convert(box, 'cxcywh', 'xywh')
            detections.append(
                fo.Detection(
                    label=phrase, 
                    bounding_box=rel_box,
                    confidence=logit,
                )
            )

        # 存储检测结果到样本中
        sample["detections"] = fo.Detections(detections=detections)
        sample.save()

    # 加载数据集到FiftyOne UI
    if view_dataset:
        session = fo.launch_app(dataset)
        session.wait()
        
    # 导出数据集为COCO格式
    if export_dataset:
        dataset.export(
            export_path,
            dataset_type=fo.types.COCODetectionDataset,
            label_field="detections",  # 使用detections字段作为检测结果
        )
        
    # 保存标注后的图像到磁盘
    if export_annotated_images:
        dataset.draw_labels(
            annotated_images_path,
            label_fields=['detections']
        )


if __name__ == '__main__':
    typer.run(main)
