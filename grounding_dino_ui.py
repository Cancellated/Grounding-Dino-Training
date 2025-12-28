# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import cv2
import numpy as np
import os
import sys
from PIL import Image, ImageTk
import threading

# 设置系统默认编码为utf-8
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, 'GroundingDINO'))

# 导入Grounding DINO的推理相关模块
from groundingdino.util.inference import Model

class GroundingDINOUI:
    def __init__(self, root):
        """初始化UI界面"""
        self.root = root
        self.root.title("Grounding DINO 图像标注工具")
        self.root.geometry("1000x700")
        self.root.resizable(True, True)
        
        # 设置字体，支持中文显示
        # 尝试几种常见的中文字体
        self.font = ('Microsoft YaHei UI', 10)
        try:
            # 测试字体是否可用
            temp_font = tk.font.Font(family=self.font[0], size=self.font[1])
        except:
            # 如果首选字体不可用，使用其他备选字体
            self.font = ('SimHei', 10)
            try:
                temp_font = tk.font.Font(family=self.font[0], size=self.font[1])
            except:
                # 最后使用默认字体
                self.font = ('Arial', 10)
        
        # 模型相关变量
        self.model = None
        self.model_loaded = False
        self.image_path = None
        self.processed_image = None
        
        # 设备选择相关变量
        self.device_choice = tk.StringVar(value="CPU")
        
        # 创建UI组件
        self._create_widgets()
        
        # 初始化状态变量并显示模型加载状态
        self.status_var = tk.StringVar()
        self.status_var.set("请加载模型和图像...")
        
    def _create_widgets(self):
        """创建UI组件"""
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 顶部控制区
        control_frame = ttk.LabelFrame(main_frame, text="控制区", padding="10")
        control_frame.pack(fill=tk.X, pady=5)
        
        # 模型路径设置
        model_frame = ttk.Frame(control_frame)
        model_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(model_frame, text="配置文件路径:", font=self.font).grid(row=0, column=0, sticky=tk.W, pady=2)
        self.config_path_var = tk.StringVar(value="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
        ttk.Entry(model_frame, textvariable=self.config_path_var, width=80).grid(row=0, column=1, padx=5, pady=2)
        ttk.Button(model_frame, text="浏览", command=self._browse_config).grid(row=0, column=2, padx=5, pady=2)
        
        ttk.Label(model_frame, text="权重文件路径:", font=self.font).grid(row=1, column=0, sticky=tk.W, pady=2)
        self.weights_path_var = tk.StringVar(value="GroundingDINO/weights/groundingdino_swint_ogc.pth")
        ttk.Entry(model_frame, textvariable=self.weights_path_var, width=80).grid(row=1, column=1, padx=5, pady=2)
        ttk.Button(model_frame, text="浏览", command=self._browse_weights).grid(row=1, column=2, padx=5, pady=2)
        
        # 设备选择
        device_frame = ttk.Frame(control_frame)
        device_frame.pack(fill=tk.X, pady=5)
        ttk.Label(device_frame, text="设备选择:", font=self.font).grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Radiobutton(device_frame, text="CPU", variable=self.device_choice, value="CPU").grid(row=0, column=1, padx=10, pady=2)
        ttk.Radiobutton(device_frame, text="GPU", variable=self.device_choice, value="GPU").grid(row=0, column=2, padx=10, pady=2)
        
        # 加载模型按钮
        ttk.Button(control_frame, text="加载模型", command=self._load_model_thread).pack(pady=5)
        
        # 阈值设置
        threshold_frame = ttk.Frame(control_frame)
        threshold_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(threshold_frame, text="边界框阈值:", font=self.font).grid(row=0, column=0, sticky=tk.W, padx=10, pady=2)
        self.box_threshold_var = tk.DoubleVar(value=0.35)
        ttk.Scale(threshold_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL, 
                  variable=self.box_threshold_var, length=200).grid(row=0, column=1, padx=5, pady=2)
        self.box_threshold_label = ttk.Label(threshold_frame, text="0.35", width=5)
        self.box_threshold_label.grid(row=0, column=2, padx=5, pady=2)
        self.box_threshold_var.trace_add("write", self._update_box_threshold_label)
        
        ttk.Label(threshold_frame, text="文本阈值:", font=self.font).grid(row=0, column=3, sticky=tk.W, padx=10, pady=2)
        self.text_threshold_var = tk.DoubleVar(value=0.25)
        ttk.Scale(threshold_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL, 
                  variable=self.text_threshold_var, length=200).grid(row=0, column=4, padx=5, pady=2)
        self.text_threshold_label = ttk.Label(threshold_frame, text="0.25", width=5)
        self.text_threshold_label.grid(row=0, column=5, padx=5, pady=2)
        self.text_threshold_var.trace_add("write", self._update_text_threshold_label)
        
        # 文本提示和图像上传
        input_frame = ttk.Frame(control_frame)
        input_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(input_frame, text="文本提示:", font=self.font).grid(row=0, column=0, sticky=tk.NW, padx=5, pady=5)
        self.caption_var = tk.StringVar(value="")
        ttk.Entry(input_frame, textvariable=self.caption_var, width=70).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(input_frame, text="上传图像", command=self._browse_image).grid(row=0, column=2, padx=5, pady=5)
        
        # 处理按钮
        ttk.Button(control_frame, text="开始处理", command=self._process_image_thread).pack(pady=5)
        
        # 状态条
        self.status_var = tk.StringVar()
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, font=self.font)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # 图像显示区域
        image_frame = ttk.Frame(main_frame)
        image_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 原始图像
        self.original_frame = ttk.LabelFrame(image_frame, text="原始图像", padding="5")
        self.original_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.original_canvas = tk.Canvas(self.original_frame, bg="lightgray")
        self.original_canvas.pack(fill=tk.BOTH, expand=True)
        
        # 处理后的图像
        self.result_frame = ttk.LabelFrame(image_frame, text="结果图像", padding="5")
        self.result_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        self.result_canvas = tk.Canvas(self.result_frame, bg="lightgray")
        self.result_canvas.pack(fill=tk.BOTH, expand=True)
    
    def _update_box_threshold_label(self, *args):
        """更新边界框阈值显示"""
        self.box_threshold_label.config(text=f"{self.box_threshold_var.get():.2f}")
    
    def _update_text_threshold_label(self, *args):
        """更新文本阈值显示"""
        self.text_threshold_label.config(text=f"{self.text_threshold_var.get():.2f}")
    
    def _browse_config(self):
        """浏览配置文件"""
        filename = filedialog.askopenfilename(
            title="选择配置文件",
            filetypes=[("Python文件", "*.py")]
        )
        if filename:
            self.config_path_var.set(filename)
    
    def _browse_weights(self):
        """浏览权重文件"""
        filename = filedialog.askopenfilename(
            title="选择权重文件",
            filetypes=[("模型权重", "*.pth")]
        )
        if filename:
            self.weights_path_var.set(filename)
    
    def _browse_image(self):
        """浏览图像文件"""
        filename = filedialog.askopenfilename(
            title="选择图像",
            filetypes=[("图像文件", "*.jpg;*.jpeg;*.png;*.bmp")]
        )
        if filename:
            self.image_path = filename
            self._display_image(self.image_path, self.original_canvas)
            self.status_var.set(f"已加载图像: {os.path.basename(filename)}")
    
    def _display_image(self, image_path, canvas):
        """在画布上显示图像"""
        try:
            # 使用PIL读取图像，更好地支持中文路径
            img_pil = Image.open(image_path)
            # 转换为RGB格式
            img_pil = img_pil.convert('RGB')
            # 转换为numpy数组，以便后续处理
            image = np.array(img_pil)
            
            # 获取画布大小
            canvas_width = canvas.winfo_width() or 400
            canvas_height = canvas.winfo_height() or 300
            
            # 调整图像大小以适应画布
            img_pil = Image.fromarray(image)
            img_pil.thumbnail((canvas_width-20, canvas_height-20), Image.LANCZOS)
            
            # 转换为Tkinter可用格式
            img_tk = ImageTk.PhotoImage(image=img_pil)
            
            # 清除并显示图像
            canvas.delete("all")
            canvas.img_tk = img_tk  # 保持引用，防止被垃圾回收
            
            # 计算居中位置
            x = (canvas_width - img_pil.width) // 2
            y = (canvas_height - img_pil.height) // 2
            
            canvas.create_image(x, y, anchor=tk.NW, image=img_tk)
            
        except Exception as e:
            messagebox.showerror("错误", f"无法显示图像: {str(e)}")
    
    def _load_model_thread(self):
        """在单独线程中加载模型"""
        self.status_var.set("正在加载模型，请稍候...")
        self.root.update_idletasks()
        
        # 创建并启动线程
        thread = threading.Thread(target=self._load_model)
        thread.daemon = True  # 设置为守护线程，确保主程序退出时线程也退出
        thread.start()
    
    def _load_model(self):
        """加载Grounding DINO模型"""
        try:
            config_path = self.config_path_var.get()
            weights_path = self.weights_path_var.get()
            
            # 检查文件是否存在
            if not os.path.exists(config_path):
                self.status_var.set(f"错误: 配置文件不存在: {config_path}")
                return
            if not os.path.exists(weights_path):
                self.status_var.set(f"错误: 权重文件不存在: {weights_path}")
                return
            
            # 添加本地缓存支持，避免在线下载依赖
            os.environ['TRANSFORMERS_OFFLINE'] = '1'  # 尝试使用离线模式
            os.environ['HF_HUB_OFFLINE'] = '1'  # HuggingFace Hub离线模式
            
            # 根据用户选择确定设备
            device = "cpu" if self.device_choice.get() == "CPU" else "cuda"
            
            # 加载模型
            self.model = Model(
                model_config_path=config_path,
                model_checkpoint_path=weights_path,
                device=device
            )
            
            self.model_loaded = True
            self.status_var.set("模型加载成功，准备就绪")
            
        except Exception as e:
            error_msg = f"加载模型失败: {str(e)}"
            self.status_var.set(error_msg)
            print(f"加载模型错误: {str(e)}")
            
            # 更详细的错误信息
            print("错误详情:")
            import traceback
            traceback.print_exc()
            
            # 检查是否是网络问题
            if 'Connection' in str(e) or 'requests' in str(e).lower() or 'download' in str(e).lower():
                self.status_var.set("网络连接问题，请检查网络或确保所有依赖已本地安装")
    
    def _process_image_thread(self):
        """在单独线程中处理图像，避免UI冻结"""
        # 检查模型是否加载
        if not self.model_loaded or self.model is None:
            messagebox.showwarning("警告", "请先加载模型")
            return
        
        # 检查图像是否加载
        if self.image_path is None:
            messagebox.showwarning("警告", "请先上传图像")
            return
        
        # 检查文本提示是否输入
        caption = self.caption_var.get().strip()
        if not caption:
            messagebox.showwarning("警告", "请输入文本提示")
            return
        
        # 更新状态
        self.status_var.set("正在处理图像，请稍候...")
        self.root.update_idletasks()
        
        # 创建并启动线程
        thread = threading.Thread(target=self._process_image)
        thread.daemon = True  # 设置为守护线程，确保主程序退出时线程也退出
        thread.start()
    
    def _process_image(self):
        """处理图像并显示结果"""
        try:
            # 获取参数
            caption = self.caption_var.get().strip()
            box_threshold = self.box_threshold_var.get()
            text_threshold = self.text_threshold_var.get()
            
            # 使用PIL读取图像，更好地支持中文路径
            img_pil = Image.open(self.image_path)
            # 转换为RGB格式
            img_pil = img_pil.convert('RGB')
            # 转换为numpy数组
            image = np.array(img_pil)
            # 转换为BGR格式，因为OpenCV处理需要
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # 使用模型进行预测
            detections, phrases = self.model.predict_with_caption(
                image=image,
                caption=caption,
                box_threshold=box_threshold,
                text_threshold=text_threshold
            )
            
            # 绘制边界框和标签
            annotated_image = self._annotate_image(image, detections, phrases)
            
            # 创建 processed_image 文件夹（如果不存在）
            output_dir = "processed_image"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # 保存处理后的图像
            processed_path = os.path.join(output_dir, f"processed_{os.path.basename(self.image_path)}")
            cv2.imwrite(processed_path, annotated_image)
            
            # 显示结果
            self._display_result(processed_path)
            
            self.status_var.set(f"处理完成，找到 {len(detections)} 个对象")
            
        except Exception as e:
            self.status_var.set(f"处理图像失败: {str(e)}")
            print(f"处理图像错误: {str(e)}")
    
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
            
            # 尝试使用支持中文的字体

            try:
                from PIL import Image, ImageDraw, ImageFont
                import numpy as np
                
                # 将OpenCV图像转换为PIL图像
                pil_img = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil_img)
                
                # 尝试加载黑体字体
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
                # 如果PIL方法失败，回退到OpenCV方法并尝试使用其他字体选项
                try:
                    # 尝试使用可能支持中文的字体
                    font_face = cv2.FONT_HERSHEY_COMPLEX  # 可能更好地支持中文的字体
                    
                    # 绘制标签背景
                    (label_width, label_height), _ = cv2.getTextSize(label, font_face, 0.5, 1)
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
                        font_face, 
                        0.5, 
                        (0, 0, 0), 
                        1
                    )
                except:
                    # 如果所有方法都失败，使用原始方法
                    (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(
                        annotated, 
                        (x1, y1 - label_height - 5), 
                        (x1 + label_width, y1), 
                        (0, 255, 0), 
                        -1
                    )
                    cv2.putText(
                        annotated, 
                        label, 
                        (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        (0, 0, 0), 
                        1
                    )
        
        return annotated
    
    def _display_result(self, image_path):
        """在结果画布上显示处理后的图像"""
        self._display_image(image_path, self.result_canvas)

# 窗口大小变化时重新显示图像
def on_resize(event, canvas, image_path):
    if image_path:
        # 延迟更新以避免频繁重绘
        canvas.after(100, lambda: _display_resized_image(canvas, image_path))

def _display_resized_image(canvas, image_path):
    if os.path.exists(image_path) and canvas.winfo_exists():
        # 重新加载并显示图像
        try:
            # 使用PIL读取图像，更好地支持中文路径
            img_pil = Image.open(image_path)
            # 转换为RGB格式
            img_pil = img_pil.convert('RGB')
            # 转换为numpy数组
            image = np.array(img_pil)
            
            canvas_width = canvas.winfo_width() or 400
            canvas_height = canvas.winfo_height() or 300
            
            img_pil = Image.fromarray(image)
            img_pil.thumbnail((canvas_width-20, canvas_height-20), Image.LANCZOS)
            
            img_tk = ImageTk.PhotoImage(image=img_pil)
            
            canvas.delete("all")
            canvas.img_tk = img_tk
            
            x = (canvas_width - img_pil.width) // 2
            y = (canvas_height - img_pil.height) // 2
            
            canvas.create_image(x, y, anchor=tk.NW, image=img_tk)
        except Exception as e:
            print(f"调整图像大小时出错: {str(e)}")

def main():
    """主函数"""
    # 设置中文字体支持
    import matplotlib
    matplotlib.use('Agg')  # 非交互式后端，避免显示问题
    
    root = tk.Tk()
    app = GroundingDINOUI(root)
    
    # 绑定窗口大小变化事件
    def on_root_resize(event):
        if app.image_path:
            _display_resized_image(app.original_canvas, app.image_path)
        if app.processed_image:
            _display_resized_image(app.result_canvas, app.processed_image)
    
    root.bind("<Configure>", on_root_resize)
    
    # 运行主循环
    root.mainloop()

if __name__ == "__main__":
    main()