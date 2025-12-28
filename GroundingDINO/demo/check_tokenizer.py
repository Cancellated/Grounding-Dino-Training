import torch
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath('../../'))

# 导入Grounding DINO相关模块
from groundingdino.util.inference import load_model

# 配置文件和权重路径
config_path = '../groundingdino/config/GroundingDINO_SwinT_OGC.py'
weights_path = '../weights/groundingdino_swint_ogc.pth'

# 加载模型
model = load_model(config_path, weights_path, device='cpu')

# 获取模型自带的tokenizer
tokenizer = model.tokenizer
caption = 'gypsum'

# 进行tokenization
tokenized = tokenizer(caption)

# 输出结果
print("tokenized类型:", type(tokenized))
print("tokenized.keys():", tokenized.keys())
print("tokenized['input_ids']类型:", type(tokenized['input_ids']))
print("tokenized['input_ids']内容:", tokenized['input_ids'])

# 检查是否是张量
if isinstance(tokenized['input_ids'], torch.Tensor):
    print("tokenized['input_ids']设备:", tokenized['input_ids'].device)
else:
    print("tokenized['input_ids']不是张量，是列表类型")
