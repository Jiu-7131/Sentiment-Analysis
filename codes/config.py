from pathlib import Path


# ==================== 路径配置 ====================
class PathConfig:
    # 数据路径
    RAW_DATA_DIR = Path("data/aclImdb")  # 原始数据目录
    PROCESSED_DATA_DIR = Path("data/processed")  # 预处理数据目录
    TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "imdb_supervised_train.csv"  # 训练数据
    TEST_DATA_PATH = PROCESSED_DATA_DIR / "imdb_supervised_test.csv"  # 测试数据

    # 模型保存根路径（新增）
    MODEL_ROOT_DIR = Path("saved_models")  # 模型保存根目录
    TOKENIZER_ROOT_DIR = Path("saved_tokenizers")  # Tokenizer保存根目录
    PLOTS_ROOT_DIR = Path("saved_plots")  # 图像保存根目录

    # 按模型类型动态生成路径的方法
    @classmethod
    def get_model_dir(cls, model_type):
        """获取特定模型的保存目录"""
        return cls.MODEL_ROOT_DIR / model_type

    @classmethod
    def get_tokenizer_dir(cls, model_type):
        """获取特定模型的Tokenizer保存目录"""
        return cls.TOKENIZER_ROOT_DIR / model_type

    @classmethod
    def get_plots_dir(cls, model_type):
        """获取特定模型的图像保存目录"""
        return cls.PLOTS_ROOT_DIR / model_type


# ==================== 数据预处理配置 ====================
class DataConfig:
    TOP_WORDS = 10000  # 词汇表最大大小
    MAX_SEQ = 500  # 文本序列最大长度
    RANDOM_STATE = 42  # 随机种子


# ==================== 训练配置 ====================
class TrainConfig:
    MODEL_TYPE = "RNN"
    NUM_RUN = 3
    BATCH_SIZE = 64  # 批次大小
    EPOCHS = 50  # 最大训练轮数
    VALIDATION_SPLIT = 0.2  # 训练集内验证集比例
    EARLY_STOPPING_PATIENCE = 3  # 早停策略 patience值
    NUM_LABELS = 2  # 分类标签数量
    OPTIMIZER = "adam"
    LOSS = "categorical_crossentropy"


# ==================== 模型超参数配置 ====================
class ModelConfig:
    # 嵌入层维度（所有模型共享）
    EMBED_DIM = 128

    # 不同模型的隐藏层维度配置
    HIDDEN_DIMS = {
        "RNN": [64],
        "MLP": [128],
        "LSTM": [64],
        "GRU": [64],
        "CNN": [128],
        "CNN+LSTM": [64],
        "BiLSTM": [64],
        "TextCNN": [128],
        "Transformer": [128]
    }

    # Transformer专属参数
    TRANSFORMER_DENSE_DIM = 32  # Transformer编码器中 dense 层维度
    TRANSFORMER_NUM_HEADS = 4  # 多头注意力头数

    # CNN专属参数
    CNN_FILTERS = 32  # 卷积核数量
    CNN_KERNEL_SIZES = [3, 4, 5]  # TextCNN的多卷积核尺寸
    CNN_POOL_SIZE = 2  # 池化窗口大小


# ==================== 统一配置入口 ====================
class Config:
    path = PathConfig
    data = DataConfig
    train = TrainConfig
    model = ModelConfig


# 实例化配置对象（供其他模块导入使用）
config = Config()