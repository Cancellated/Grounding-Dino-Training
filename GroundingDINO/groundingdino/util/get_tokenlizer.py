from transformers import AutoTokenizer, BertModel, BertTokenizer, RobertaModel, RobertaTokenizerFast
import os

# 获取项目根目录
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
# 定义本地缓存目录
CACHE_DIR = os.path.join(PROJECT_ROOT, "huggingface_cache", "transformers")

# 确保缓存目录存在
os.makedirs(CACHE_DIR, exist_ok=True)

def get_tokenlizer(text_encoder_type):
    if not isinstance(text_encoder_type, str):
        # print("text_encoder_type is not a str")
        if hasattr(text_encoder_type, "text_encoder_type"):
            text_encoder_type = text_encoder_type.text_encoder_type
        elif text_encoder_type.get("text_encoder_type", False):
            text_encoder_type = text_encoder_type.get("text_encoder_type")
        elif os.path.isdir(text_encoder_type) and os.path.exists(text_encoder_type):
            pass
        else:
            raise ValueError(
                "Unknown type of text_encoder_type: {}".format(type(text_encoder_type))
            )
    print("final text_encoder_type: {}".format(text_encoder_type))

    # 优先使用本地缓存
    try:
        tokenizer = AutoTokenizer.from_pretrained(text_encoder_type, cache_dir=CACHE_DIR, local_files_only=True)
    except (OSError, ValueError):
        # 如果本地缓存不存在，尝试下载
        tokenizer = AutoTokenizer.from_pretrained(text_encoder_type, cache_dir=CACHE_DIR, local_files_only=False)
    return tokenizer


def get_pretrained_language_model(text_encoder_type):
    if text_encoder_type == "bert-base-uncased" or (os.path.isdir(text_encoder_type) and os.path.exists(text_encoder_type)):
        # 优先使用本地缓存
        try:
            return BertModel.from_pretrained(text_encoder_type, cache_dir=CACHE_DIR, local_files_only=True)
        except (OSError, ValueError):
            # 如果本地缓存不存在，尝试下载
            return BertModel.from_pretrained(text_encoder_type, cache_dir=CACHE_DIR, local_files_only=False)
    if text_encoder_type == "roberta-base":
        # 优先使用本地缓存
        try:
            return RobertaModel.from_pretrained(text_encoder_type, cache_dir=CACHE_DIR, local_files_only=True)
        except (OSError, ValueError):
            # 如果本地缓存不存在，尝试下载
            return RobertaModel.from_pretrained(text_encoder_type, cache_dir=CACHE_DIR, local_files_only=False)

    raise ValueError("Unknown text_encoder_type {}".format(text_encoder_type))
