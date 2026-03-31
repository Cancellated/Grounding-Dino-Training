import typer
from groundingdino.util.inference import load_model, load_image, predict
from tqdm import tqdm
import torchvision
import torch
import fiftyone as fo


def main(
        image_directory: str = 'test_grounding_dino',
        text_prompt: str = 'bus, car',
        box_threshold: float = 0.15, 
        text_threshold: float = 0.10,
        export_dataset: bool = False,
        export_path: str = 'coco_dataset',
        view_dataset: bool = False,
        export_annotated_images: bool = True,
        annotated_images_path: str = '../../images_with_bounding_boxes',
        weights_path : str = "../../weights/groundingdino_swint_ogc.pth",
        config_path: str = "../../groundingdino/config/GroundingDINO_SwinT_OGC.py",
        subsample: int = None,
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

        # Store detections in a field name of your choice, ensure it's always set
        sample["detections"] = fo.Detections(detections=detections)
        sample.save()

    # loads the voxel fiftyone UI ready for viewing the dataset.
    if view_dataset:
        session = fo.launch_app(dataset)
        session.wait()
        
    # exports COCO dataset ready for training
    if export_dataset:
        dataset.export(
            export_path,
            dataset_type=fo.types.COCODetectionDataset,
            label_field="detections",  # 明确指定使用detections字段作为检测结果
        )
        
    # saves bounding boxes plotted on the input images to disk
    if export_annotated_images:
        dataset.draw_labels(
            annotated_images_path,
            label_fields=['detections']
        )


if __name__ == '__main__':
    typer.run(main)
