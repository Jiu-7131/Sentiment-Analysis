# ==================== 环境配置 ====================
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import (
    classification_report, cohen_kappa_score,
    roc_curve, auc
)
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, Model, Sequential
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    Embedding, Dropout, SimpleRNN, Dense, Flatten,
    LSTM, GRU, Conv1D, MaxPooling1D, Bidirectional,
    Input, concatenate, GlobalMaxPooling1D
)
from config import config

matplotlib.use('TkAgg')
plt.rcParams['font.sans-serif'] = ['KaiTi']  # 中文显示
plt.rcParams['axes.unicode_minus'] = False  # 负号显示


# ==================== 自定义层 ====================
class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim, activation="relu"),
             layers.Dense(embed_dim), ]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]
        attention_output = self.attention(inputs, inputs, attention_mask=mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dense_dim": self.dense_dim,
        })
        return config


class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, input_dim, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(
            input_dim=input_dim, output_dim=output_dim)
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=output_dim)
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.output_dim = output_dim

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)

    def get_config(self):
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim,
            "sequence_length": self.sequence_length,
            "input_dim": self.input_dim, })
        return config


# ==================== 数据加载 ====================
def load_data(train_path=config.path.TRAIN_DATA_PATH, test_path=config.path.TEST_DATA_PATH):
    """加载预处理数据"""
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    return df_train, df_test


# ==================== 模型构建 ====================
def build_model(top_words=config.data.TOP_WORDS, max_seq=config.data.MAX_SEQ,
                num_labels=config.train.NUM_LABELS, model_type=config.train.MODEL_TYPE,
                hidden_dim=config.model.HIDDEN_DIMS[config.train.MODEL_TYPE]):
    """构建9种不同的神经网络模型"""
    if model_type == 'RNN':
        model = Sequential()
        model.add(Embedding(top_words, output_dim=config.model.EMBED_DIM, mask_zero=True))
        model.add(Dropout(0.25))
        model.add(SimpleRNN(hidden_dim[0]))
        model.add(Dropout(0.25))
        model.add(Dense(num_labels, activation="softmax"))

    elif model_type == 'MLP':
        model = Sequential()
        model.add(Embedding(top_words, output_dim=config.model.EMBED_DIM))
        model.add(Flatten())
        model.add(Dropout(0.25))
        model.add(Dense(hidden_dim[0]))
        model.add(Dropout(0.25))
        model.add(Dense(num_labels, activation="softmax"))

    elif model_type == 'LSTM':
        model = Sequential()
        model.add(Embedding(top_words, output_dim=config.model.EMBED_DIM))
        model.add(Dropout(0.25))
        model.add(LSTM(hidden_dim[0]))
        model.add(Dropout(0.25))
        model.add(Dense(num_labels, activation="softmax"))

    elif model_type == 'GRU':
        model = Sequential()
        model.add(Embedding(top_words, output_dim=config.model.EMBED_DIM))
        model.add(Dropout(0.25))
        model.add(GRU(hidden_dim[0]))
        model.add(Dropout(0.25))
        model.add(Dense(num_labels, activation="softmax"))

    elif model_type == 'CNN':
        model = Sequential()
        model.add(Embedding(top_words, output_dim=config.model.EMBED_DIM, mask_zero=True))
        model.add(Dropout(0.25))
        model.add(Conv1D(filters=config.model.CNN_FILTERS, kernel_size=3, padding="same", activation="relu"))
        model.add(MaxPooling1D(pool_size=config.model.CNN_FILTERS))
        model.add(Flatten())
        model.add(Dense(hidden_dim[0], activation="relu"))
        model.add(Dropout(0.25))
        model.add(Dense(num_labels, activation="softmax"))

    elif model_type == 'CNN+LSTM':
        model = Sequential()
        model.add(Embedding(top_words, output_dim=config.model.EMBED_DIM))
        model.add(Dropout(0.25))
        model.add(Conv1D(filters=config.model.CNN_FILTERS, kernel_size=3, padding="same", activation="relu"))
        model.add(MaxPooling1D(pool_size=config.model.CNN_POOL_SIZE))
        model.add(LSTM(hidden_dim[0]))
        model.add(Dropout(0.25))
        model.add(Dense(num_labels, activation="softmax"))

    elif model_type == 'BiLSTM':
        model = Sequential()
        model.add(Embedding(top_words, output_dim=config.model.EMBED_DIM))
        model.add(Bidirectional(LSTM(hidden_dim[0])))
        model.add(Dense(hidden_dim[0], activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(num_labels, activation='softmax'))

    elif model_type == 'TextCNN':
        inputs = Input(name='inputs', shape=[max_seq, ], dtype='int32')
        layer = Embedding(top_words, output_dim=config.model.EMBED_DIM)(inputs)
        cnn1 = Conv1D(32, 3, padding='same', strides=1, activation='relu')(layer)
        cnn1 = MaxPooling1D(pool_size=2)(cnn1)
        cnn2 = Conv1D(32, 4, padding='same', strides=1, activation='relu')(layer)
        cnn2 = MaxPooling1D(pool_size=2)(cnn2)
        cnn3 = Conv1D(32, 5, padding='same', strides=1, activation='relu')(layer)
        cnn3 = MaxPooling1D(pool_size=2)(cnn3)
        cnn = concatenate([cnn1, cnn2, cnn3], axis=-1)
        x = Flatten()(cnn)
        x = Dense(hidden_dim[0], activation='relu')(x)
        output = Dense(num_labels, activation='softmax')(x)
        model = Model(inputs=inputs, outputs=output)

    elif model_type == 'Transformer':
        inputs = Input(name='inputs', shape=[max_seq, ], dtype='int32')
        x = Embedding(top_words, output_dim=config.model.EMBED_DIM, mask_zero=True)(inputs)
        x = TransformerEncoder(
            embed_dim=config.model.EMBED_DIM,
            dense_dim=config.model.TRANSFORMER_DENSE_DIM,
            num_heads=config.model.TRANSFORMER_NUM_HEADS
        )(x)
        x = GlobalMaxPooling1D()(x)
        x = Dropout(0.25)(x)
        outputs = Dense(num_labels, activation="softmax")(x)
        model = Model(inputs, outputs)

    model.compile(loss=config.train.LOSS,
                  optimizer=config.train.OPTIMIZER,
                  metrics=["accuracy"])
    return model


# ==================== 可视化工具 ====================
def plot_loss(history, model_type, save_path, run_id=1):
    """绘制并保存训练过程曲线"""
    plt.subplots(1, 2, figsize=(10, 3))
    # 添加上方居中的总标题
    plt.suptitle(f'{model_type}模型第{run_id}次训练过程', fontsize=12, ha='center')

    plt.subplot(121)
    loss = history.history["loss"]
    epochs = range(1, len(loss) + 1)
    val_loss = history.history["val_loss"]
    plt.plot(epochs, loss, "bo", label="Training Loss")
    plt.plot(epochs, val_loss, "r", label="Validation Loss")
    plt.title("训练与验证损失")
    plt.xlabel("轮次")
    plt.ylabel("损失值")
    plt.legend()

    plt.subplot(122)
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    plt.plot(epochs, acc, "b-", label="Training Acc")
    plt.plot(epochs, val_acc, "r--", label="Validation Acc")
    plt.title("训练与验证准确率")
    plt.xlabel("轮次")
    plt.ylabel("准确率")
    plt.legend()
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # 调整布局，避免标题被遮挡

    # 保存图像
    save_file = save_path / f"{model_type}_loss_acc_{run_id}.png"
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"{model_type}模型第{run_id}次训练曲线已保存至: {save_file}")


def plot_confusion_matrix(model, X_test, Y_test_original, model_type, save_path, run_id=1):
    """绘制并保存混淆矩阵"""
    prob = model.predict(X_test)
    pred = np.argmax(prob, axis=1)
    table = pd.crosstab(Y_test_original, pred,
                        rownames=['Actual'],
                        colnames=['Predicted'])

    plt.figure(figsize=(8, 6))
    # 添加上方居中的总标题
    plt.suptitle(f'{model_type}模型第{run_id}次混淆矩阵', fontsize=12, ha='center')

    sns.heatmap(table, cmap='Blues', fmt='.20g', annot=True)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    # 调整布局，避免标题被遮挡
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # 保存图像
    save_file = save_path / f"{model_type}_confusion_matrix_{run_id}.png"
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"{model_type}模型第{run_id}次混淆矩阵已保存至: {save_file}")

    # 打印分类报告
    print(classification_report(Y_test_original, pred, digits=4))
    print('Kappa:', cohen_kappa_score(Y_test_original, pred))


def plot_roc_curve(y_true, y_pred_proba, model_type, save_path, run_id=1):
    """绘制ROC曲线并计算AUC，根据模型名称区分图像"""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])  # 取正面类别的预测概率
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # 随机猜测基准线
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳性率 (FPR)')
    plt.ylabel('真阳性率 (TPR)')
    plt.title(f'{model_type}模型第{run_id}次ROC曲线')  # 标题包含模型名称
    plt.legend(loc="lower right")
    plt.tight_layout()

    # 保存路径包含模型名称，避免文件冲突
    save_file = save_path / f"{model_type}_roc_curve_{run_id}.png"
    plt.savefig(save_file, dpi=300)
    plt.close()
    # 打印信息包含模型名称
    print(f"{model_type}模型第{run_id}次ROC曲线已保存至: {save_file}")


def plot_error_length_distribution(df_test, y_true, y_pred, model_type, save_path, run_id=1):
    """绘制并保存特定模型错误预测样本的文本长度分布"""
    df_test = df_test.copy()
    df_test["true_label"] = y_true
    df_test["pred_label"] = y_pred
    df_test["is_error"] = df_test["true_label"] != df_test["pred_label"]
    df_test["text_length"] = df_test["cleaned_review"].apply(lambda x: len(x.split()))

    error_df = df_test[df_test["is_error"]]

    plt.figure(figsize=(10, 5))
    sns.histplot(error_df["text_length"], bins=30, kde=True, color='red')
    plt.title(f'{model_type}模型第{run_id}次错误预测样本的文本长度分布')  # 标题包含模型名称
    plt.xlabel('单词数量')
    plt.ylabel('错误样本数')
    plt.tight_layout()

    # 保存路径包含模型名称，避免不同模型文件冲突
    save_file = save_path / f"{model_type}_error_text_length_{run_id}.png"
    plt.savefig(save_file, dpi=300)
    plt.close()
    print(f"{model_type}模型第{run_id}次错误样本长度分布图已保存至: {save_file}")


# ==================== 模型训练 ====================
def train_model(df_train, df_test, model_type=config.train.MODEL_TYPE,
                epochs=config.train.EPOCHS, batch_size=config.train.BATCH_SIZE):
    """仅负责模型训练"""
    # 对df_train进行向量化和训练/验证分割
    tok = Tokenizer(num_words=config.data.TOP_WORDS)
    tok.fit_on_texts(df_train['cleaned_review'].to_numpy())
    X_train = tok.texts_to_sequences(df_train['cleaned_review'].to_numpy())
    X_train = sequence.pad_sequences(X_train, maxlen=config.data.MAX_SEQ)
    Y_train = to_categorical(df_train['label'].to_numpy())

    # test集作为最终测试集
    X_test = tok.texts_to_sequences(df_test['cleaned_review'].to_numpy())
    X_test = sequence.pad_sequences(X_test, maxlen=config.data.MAX_SEQ)
    Y_test = to_categorical(df_test['label'].to_numpy())
    Y_test_original = df_test['label'].to_numpy()

    # 构建模型
    model = build_model(model_type=model_type)

    # 训练模型
    es = EarlyStopping(patience=config.train.EARLY_STOPPING_PATIENCE, restore_best_weights=True)
    history = model.fit(X_train, Y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=config.train.VALIDATION_SPLIT,
                        callbacks=[es])

    print(f"\n=== {model_type}模型训练完成，参数摘要 ===")
    model.summary()

    return model, tok, history, X_test, Y_test, Y_test_original


# ==================== 模型评估 ====================
def evaluate_model(model, history, X_test, Y_test, Y_test_original, df_test, model_type, plots_dir, run_id=1):
    """模型评估和图像保存（增强版）"""
    # 1. 绘制并保存训练曲线
    plot_loss(history, model_type, save_path=plots_dir, run_id=run_id)

    # 2. 评估模型性能（loss和accuracy）
    loss, accuracy = model.evaluate(X_test, Y_test)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    # 3. 计算预测概率和预测标签（用于后续评估）
    y_pred_proba = model.predict(X_test)  # 预测概率
    y_pred = np.argmax(y_pred_proba, axis=1)  # 预测标签

    # 4. 绘制并保存混淆矩阵（包含分类报告）
    plot_confusion_matrix(model, X_test, Y_test_original, model_type, save_path=plots_dir, run_id=run_id)

    # 5. 绘制ROC曲线并计算AUC
    plot_roc_curve(Y_test_original, y_pred_proba, model_type, save_path=plots_dir, run_id=run_id)

    # 6. 分析错误样本的文本长度分布
    plot_error_length_distribution(df_test, Y_test_original, y_pred, model_type, save_path=plots_dir, run_id=run_id)


# ==================== 主程序 ====================
if __name__ == "__main__":
    # 获取当前模型类型
    model_type = config.train.MODEL_TYPE

    # 创建模型专属的保存目录（按模型类型分类）
    model_save_dir = config.path.get_model_dir(model_type)
    tokenizer_save_dir = config.path.get_tokenizer_dir(model_type)
    plots_dir = config.path.get_plots_dir(model_type)

    model_save_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_save_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载数据
    df_train, df_test = load_data()

    # 训练次数设置
    num_runs = config.train.NUM_RUN  # 训练次数

    for run_id in range(1, num_runs + 1):
        print(f"\n=== 第{run_id}次训练{config.train.MODEL_TYPE}模型 ===")
        # 训练模型
        model, tok, history, X_test, Y_test, Y_test_original = train_model(
            df_train, df_test, model_type=config.train.MODEL_TYPE, epochs=config.train.EPOCHS)

        # 保存模型和Tokenizer
        model.save(model_save_dir / f'{model_type}_model_{run_id}.h5')
        with open(tokenizer_save_dir / f'{model_type}_tokenizer_{run_id}.pickle', 'wb') as handle:
            pickle.dump(tok, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # 统一评估并保存图像
        print(f"\n=== 第{run_id}次训练{config.train.MODEL_TYPE}模型性能评估 ===")
        evaluate_model(
            model, history, X_test, Y_test, Y_test_original, df_test,
            model_type=model_type, plots_dir=plots_dir, run_id=run_id
        )