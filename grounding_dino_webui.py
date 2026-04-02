# -*- coding: utf-8 -*-
"""
Grounding DINO Web UI
将本地UI转换为Web端实现，支持模型配置、设备选择、图像处理等功能
"""
import os
import sys
import cv2
import numpy as np
from PIL import Image
import torch
import gradio as gr

# 添加调试信息和错误处理
print("Starting Grounding DINO Web UI...")
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# 禁用YAPF缓存以避免权限问题
os.environ['YAPF_CACHE_DIR'] = os.path.join(os.getcwd(), 'yapf_cache')
if not os.path.exists(os.environ['YAPF_CACHE_DIR']):
    os.makedirs(os.environ['YAPF_CACHE_DIR'], exist_ok=True)
    print(f"Created YAPF cache directory: {os.environ['YAPF_CACHE_DIR']}")

# 添加临时目录设置
import tempfile
tempfile.tempdir = os.path.join(os.getcwd(), 'temp')
if not os.path.exists(tempfile.tempdir):
    os.makedirs(tempfile.tempdir, exist_ok=True)
    print(f"Created temp directory: {tempfile.tempdir}")

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, 'GroundingDINO'))

# 导入Grounding DINO的推理相关模块
from groundingdino.util.inference import Model

class GroundingDINOWebUI:
    def __init__(self):
        """初始化Web UI"""
        self.model = None
        self.model_loaded = False
        
        # 默认模型路径
        self.default_config_path = os.path.join(project_root, 'GroundingDINO', 'groundingdino', 'config', 'GroundingDINO_SwinT_OGC.py')
        self.default_weights_path = os.path.join(project_root, 'GroundingDINO', 'weights', 'groundingdino_swint_ogc.pth')
        
        # 确保weights目录存在
        if not os.path.exists(os.path.dirname(self.default_weights_path)):
            os.makedirs(os.path.dirname(self.default_weights_path))
        
        # 检查CUDA可用性
        self.cuda_available = torch.cuda.is_available()
        self.default_device = "cuda" if self.cuda_available else "cpu"
    
    def load_model(self, config_path, weights_path, device):
        """加载模型"""
        try:
            # 检查文件是否存在
            if not os.path.exists(config_path):
                return f"错误: 配置文件不存在: {config_path}"
            if not os.path.exists(weights_path):
                return f"错误: 权重文件不存在: {weights_path}"
            
            # 添加本地缓存支持，避免在线下载依赖
            os.environ['TRANSFORMERS_OFFLINE'] = '1'  # 尝试使用离线模式
            os.environ['HF_HUB_OFFLINE'] = '1'  # HuggingFace Hub离线模式
            
            # 加载模型
            self.model = Model(
                model_config_path=config_path,
                model_checkpoint_path=weights_path,
                device=device
            )
            
            self.model_loaded = True
            return "模型加载成功，准备就绪"
        except Exception as e:
            error_msg = f"加载模型失败: {str(e)}"
            print(f"加载模型错误: {str(e)}")
            return error_msg
    
    def process_image(self, image, caption, box_threshold, text_threshold):
        """处理图像"""
        try:
            # 检查模型是否加载
            if not self.model_loaded or self.model is None:
                return "错误: 请先加载模型", None
            
            # 检查文本提示是否输入
            if not caption.strip():
                return "错误: 请输入文本提示", None
            
            # 使用PIL读取图像
            img_pil = image.convert('RGB')
            # 转换为numpy数组，然后转换为BGR格式
            image_np = np.array(img_pil)
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            
            # 使用模型进行预测
            detections, phrases = self.model.predict_with_caption(
                image=image_bgr,
                caption=caption,
                box_threshold=box_threshold,
                text_threshold=text_threshold
            )
            
            # 绘制边界框和标签
            annotated_image = self._annotate_image(image_bgr, detections, phrases)
            
            # 转换回RGB格式用于显示
            annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            
            return f"处理完成，找到 {len(detections)} 个对象", Image.fromarray(annotated_image_rgb)
        except Exception as e:
            error_msg = f"处理图像失败: {str(e)}"
            print(f"处理图像错误: {str(e)}")
            return error_msg, None
    
    def _annotate_image(self, image, detections, phrases):
        """在图像上绘制边界框和标签"""
        # 创建一个副本以避免修改原始图像
        annotated = image.copy()
        
        # 遍历所有检测结果
        for i in range(len(detections)):
            # 获取边界框坐标
            x1, y1, x2, y2 = detections.xyxy[i].astype(int)
            
            # 获取置信度和标签
            confidence = detections.confidence[i]
            label = f"{phrases[i]} ({confidence:.2f})"
            
            # 绘制边界框
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            try:
                from PIL import Image, ImageDraw, ImageFont
                import numpy as np
                
                # 将OpenCV图像转换为PIL图像
                pil_img = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil_img)
                
                # 尝试加载支持中文的字体
                font_path = "C:\\Windows\\Fonts\\simhei.ttf"  # 黑体字体路径
                try:
                    font = ImageFont.truetype(font_path, 12)
                except:
                    # 如果无法加载指定字体，使用默认字体
                    font = ImageFont.load_default()
                
                # 获取文本大小
                text_size = draw.textsize(label, font=font)
                
                # 绘制标签背景
                draw.rectangle(
                    [(x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1)],
                    fill=(0, 255, 0)
                )
                
                # 绘制标签文本
                draw.text(
                    (x1, y1 - text_size[1] - 5),
                    label,
                    font=font,
                    fill=(0, 0, 0)
                )
                
                # 将PIL图像转换回OpenCV图像
                annotated = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                
            except Exception as e:
                # 如果PIL方法失败，回退到OpenCV方法
                try:
                    # 绘制标签背景
                    (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(
                        annotated, 
                        (x1, y1 - label_height - 5), 
                        (x1 + label_width, y1), 
                        (0, 255, 0), 
                        -1
                    )
                    
                    # 绘制标签文本
                    cv2.putText(
                        annotated, 
                        label, 
                        (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        (0, 0, 0), 
                        1
                    )
                except:
                    # 如果所有方法都失败，跳过标签绘制
                    pass
        
        return annotated

def create_webui():
    """创建Web UI界面"""
    webui = GroundingDINOWebUI()
    
    # 创建Gradio界面
    with gr.Blocks(title="Grounding DINO Web UI") as demo:
        gr.Markdown("# Grounding DINO 图像标注工具")
        gr.Markdown("基于Web的Grounding DINO图像标注工具，支持本地模型加载和设备选择")
        
        # 模型设置按钮
        model_settings_btn = gr.Button("切换模型设置", variant="secondary")
        
        # 使用状态变量跟踪模型设置面板的可见性
        settings_visible = gr.State(False)
        
        # 模型设置部分
        model_settings = gr.Column(visible=False)  # 默认隐藏
        with model_settings:
            with gr.Row():
                # 配置文件路径
                config_path = gr.Textbox(
                    label="配置文件路径",
                    value=webui.default_config_path,
                    placeholder="输入配置文件路径"
                )
            
            with gr.Row():
                # 权重文件路径
                weights_path = gr.Textbox(
                    label="权重文件路径",
                    value=webui.default_weights_path,
                    placeholder="输入权重文件路径"
                )
            
            with gr.Row():
                # 设备选择
                device = gr.Radio(
                    label="设备选择",
                    choices=["cpu", "cuda"],
                    value=webui.default_device
                )
                
                # 显示CUDA可用性
                cuda_status = gr.Textbox(
                    label="CUDA状态",
                    value=f"CUDA可用: {webui.cuda_available}",
                    interactive=False
                )
            
            with gr.Row():
                # 加载模型按钮
                load_button = gr.Button("加载模型", variant="primary")
            
            # 加载状态
            load_status = gr.Textbox(
                label="加载状态",
                value="请加载模型...",
                interactive=False
            )
        
        # 图像处理部分
        with gr.Column():
                with gr.Row():
                    # 图像上传
                    input_image = gr.Image(
                        label="输入图像",
                        type="pil"
                    )
                    
                    # 结果显示
                    output_image = gr.Image(
                        label="结果图像",
                        type="pil"
                    )
                
                with gr.Row():
                    # 文本提示
                    caption = gr.Textbox(
                        label="文本提示",
                        placeholder="输入要检测的对象名称，如: person, dog, cat"
                    )
                
                with gr.Row():
                    # 阈值设置
                    box_threshold = gr.Slider(
                        label="边界框阈值",
                        minimum=0.0,
                        maximum=1.0,
                        value=0.35,
                        step=0.01
                    )
                    
                    text_threshold = gr.Slider(
                        label="文本阈值",
                        minimum=0.0,
                        maximum=1.0,
                        value=0.25,
                        step=0.01
                    )
                
                with gr.Row():
                    # 处理按钮
                    process_button = gr.Button("开始处理", variant="primary")
                
                # 处理状态
                process_status = gr.Textbox(
                    label="处理状态",
                    value="请上传图像并输入文本提示...",
                    interactive=False
                )
        
        # 模型设置按钮点击事件（切换显示/隐藏）
        def toggle_model_settings(current_visible):
            new_visible = not current_visible
            return [gr.update(visible=new_visible), new_visible]
        
        model_settings_btn.click(
            fn=toggle_model_settings,
            inputs=[settings_visible],
            outputs=[model_settings, settings_visible]
        )
        
        # 自动加载模型功能（页面打开时触发）
        def auto_load_model():
            # 使用默认配置自动加载模型
            return webui.load_model(webui.default_config_path, webui.default_weights_path, webui.default_device)
        
        # 页面加载时自动执行模型加载
        demo.load(
            fn=auto_load_model,
            inputs=[],
            outputs=load_status
        )
        
        # 手动加载模型功能
        load_button.click(
            fn=webui.load_model,
            inputs=[config_path, weights_path, device],
            outputs=load_status
        )
        
        # 处理图像功能
        process_button.click(
            fn=webui.process_image,
            inputs=[input_image, caption, box_threshold, text_threshold],
            outputs=[process_status, output_image]
        )
    
    return demo

def main():
    """主函数"""
    # 创建并启动Web UI
    demo = create_webui()
    demo.launch(
        server_name="0.0.0.0",  # 允许外部访问
        server_port=7860,        # Web UI端口
        share=False              # 是否生成公网链接
    )

if __name__ == "__main__":
    main()