from config import config
from train import load_data
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

matplotlib.use('TkAgg')
plt.rcParams['font.sans-serif'] = ['KaiTi']  # 中文显示
plt.rcParams['axes.unicode_minus'] = False  # 负号显示

# 定义核心情感词列表（基于实验报告中情感词分布分析）
POSITIVE_WORDS = ["good", "great", "excellent", "wonderful", "amazing", "love", "like", "best", "fantastic", "superb"]
NEGATIVE_WORDS = ["bad", "terrible", "worse", "awful", "horrible", "hate", "dislike", "worst", "poor", "disappointing"]


def plot_text_length_distribution(df, split="train", save_path=config.path.PLOTS_ROOT_DIR):
    """绘制训练/测试集评论长度分布（含均值、中位数、80%区间标注）"""
    # 创建保存目录（若不存在）
    save_path.mkdir(parents=True, exist_ok=True)

    # 空数据校验
    if df.empty:
        print(f"警告：{split}集数据为空，跳过文本长度分布图绘制")
        return

    # 计算每条评论的长度（按单词数）
    df = df.copy()
    df["text_length"] = df["cleaned_review"].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)

    # 统计关键指标
    mean_len = df["text_length"].mean()
    median_len = df["text_length"].median()
    q1 = df["text_length"].quantile(0.1)  # 10分位数
    q9 = df["text_length"].quantile(0.9)  # 90分位数

    plt.figure(figsize=(12, 6))
    sns.histplot(df["text_length"], bins=50, kde=True, color="#1f77b4", alpha=0.7)

    # 标注关键统计线
    plt.axvline(mean_len, color='r', linestyle='--', linewidth=2, label=f'均值: {mean_len:.1f}')
    plt.axvline(median_len, color='orange', linestyle='--', linewidth=2, label=f'中位数: {median_len:.1f}')
    plt.axvline(q1, color='green', linestyle=':', linewidth=2, label=f'10分位数: {q1:.1f}')
    plt.axvline(q9, color='green', linestyle=':', linewidth=2, label=f'90分位数: {q9:.1f}')
    plt.axvspan(50, 400, alpha=0.1, color='yellow', label='80%样本区间(50-400词)')

    plt.title(f'{split}集评论长度分布', fontsize=14, fontweight='bold')
    plt.xlabel('单词数量', fontsize=12)
    plt.ylabel('样本数', fontsize=12)
    plt.xlim(0, 1000)  # 聚焦有效长度范围（排除极长异常值）
    plt.legend(fontsize=10)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    save_file = save_path / f"{split}_text_length_dist.png"
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"{split}集评论长度分布图已保存至: {save_file}")


def plot_sentiment_balance(df, split="train", save_path=config.path.PLOTS_ROOT_DIR):
    """绘制正负样本平衡分布图（验证1:1比例）"""
    save_path.mkdir(parents=True, exist_ok=True)

    # 空数据校验
    if df.empty:
        print(f"警告：{split}集数据为空，跳过正负样本分布图绘制")
        return

    df = df.copy()
    sentiment_count = df["label"].value_counts().sort_index()
    sentiment_label = {0: "负向情感", 1: "正向情感"}

    plt.figure(figsize=(8, 5))
    bars = sns.countplot(x="label", data=df, palette=["#ff6b6b", "#4ecdc4"], alpha=0.8)

    # 添加数值标签
    total = len(df)
    for bar in bars.patches:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 50,
                 f'{height}\n({height / total * 100:.1f}%)',
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.title(f'{split}集正负样本分布（平衡验证）', fontsize=14, fontweight='bold')
    plt.xlabel('情感标签', fontsize=12)
    plt.ylabel('样本数', fontsize=12)
    plt.xticks([0, 1], [sentiment_label[0], sentiment_label[1]], fontsize=11)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    save_file = save_path / f"{split}_sentiment_balance.png"
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"{split}集正负样本分布图已保存至: {save_file}")


def plot_rating_sentiment_mapping(df, split="train", save_path=config.path.PLOTS_ROOT_DIR):
    """绘制评分与情感标签对应关系图（验证评分≤4=负向，≥7=正向）"""
    save_path.mkdir(parents=True, exist_ok=True)

    # 空数据校验
    if df.empty:
        print(f"警告：{split}集数据为空，跳过评分-情感对应图绘制")
        return

    df = df.copy()
    # 过滤有效评分（1-4分、7-10分）
    df_valid = df[df["rating"].between(1, 10)]

    if df_valid.empty:
        print(f"警告：{split}集无有效评分数据，跳过评分-情感对应图绘制")
        return

    plt.figure(figsize=(10, 6))
    sns.countplot(x="rating", hue="label", data=df_valid,
                  palette=["#ff6b6b", "#4ecdc4"], alpha=0.8)

    # 添加分隔线标注规则
    plt.axvline(3.5, color='red', linestyle='--', linewidth=2, alpha=0.7,
                label='负向(≤4分) / 正向(≥7分)')
    plt.axvline(5.5, color='gray', linestyle='--', linewidth=2, alpha=0.5,
                label='中性(5-6分，已剔除)')

    plt.title(f'{split}集评分与情感标签对应关系', fontsize=14, fontweight='bold')
    plt.xlabel('评分', fontsize=12)
    plt.ylabel('样本数', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    save_file = save_path / f"{split}_rating_sentiment_mapping.png"
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"{split}集评分-情感对应图已保存至: {save_file}")


def plot_sentiment_word_distribution(df, split="train", save_path=config.path.PLOTS_ROOT_DIR):
    """绘制正负情感词分布统计（基于实验报告中情感词频次分析）"""
    save_path.mkdir(parents=True, exist_ok=True)

    # 空数据校验
    if df.empty:
        print(f"警告：{split}集数据为空，跳过情感词分布图绘制")
        return

    df = df.copy()

    # 统计每条评论中的正负情感词数量
    def count_sentiment_words(text, words):
        if pd.isna(text):
            return 0
        return sum(1 for word in words if word.lower() in str(text).lower().split())

    df["positive_word_count"] = df["cleaned_review"].apply(
        lambda x: count_sentiment_words(x, POSITIVE_WORDS)
    )
    df["negative_word_count"] = df["cleaned_review"].apply(
        lambda x: count_sentiment_words(x, NEGATIVE_WORDS)
    )

    # 计算平均频次
    pos_avg = df["positive_word_count"].mean()
    neg_avg = df["negative_word_count"].mean()

    plt.figure(figsize=(10, 6))
    x = ["正向情感词", "负向情感词"]
    y = [pos_avg, neg_avg]
    bars = plt.bar(x, y, color=["#4ecdc4", "#ff6b6b"], alpha=0.8, width=0.5)

    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.05,
                 f'平均每条评论: {height:.1f}个',
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.title(f'{split}集正负情感词分布统计', fontsize=14, fontweight='bold')
    plt.ylabel('平均出现频次（每条评论）', fontsize=12)
    plt.ylim(0, max(y) * 1.2)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    save_file = save_path / f"{split}_sentiment_word_dist.png"
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"{split}集情感词分布图已保存至: {save_file}")


def plot_text_length_vs_sentiment(df, split="train", save_path=config.path.PLOTS_ROOT_DIR):
    """绘制不同情感标签的文本长度分布对比（分析长度与情感的关联性）"""
    save_path.mkdir(parents=True, exist_ok=True)

    # 空数据校验
    if df.empty:
        print(f"警告：{split}集数据为空，跳过文本长度-情感对比图绘制")
        return

    df = df.copy()
    df["text_length"] = df["cleaned_review"].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)

    # 确保有两种情感标签数据
    if df["label"].nunique() < 2:
        print(f"警告：{split}集仅包含单一情感标签，跳过文本长度-情感对比图绘制")
        return

    plt.figure(figsize=(12, 6))
    sns.violinplot(x="label", y="text_length", data=df,
                   palette=["#ff6b6b", "#4ecdc4"], alpha=0.7, inner="quartile")

    # 标注两组的均值
    pos_mean = df[df["label"] == 1]["text_length"].mean()
    neg_mean = df[df["label"] == 0]["text_length"].mean()
    plt.text(0, pos_mean + 50, f'负向均值: {neg_mean:.1f}词',
             ha='center', va='bottom', fontsize=10, color="#ff6b6b", fontweight='bold')
    plt.text(1, neg_mean + 50, f'正向均值: {pos_mean:.1f}词',
             ha='center', va='bottom', fontsize=10, color="#4ecdc4", fontweight='bold')

    plt.title(f'{split}集不同情感标签的文本长度分布对比', fontsize=14, fontweight='bold')
    plt.xlabel('情感标签', fontsize=12)
    plt.ylabel('文本长度（单词数）', fontsize=12)
    plt.xticks([0, 1], ["负向情感", "正向情感"], fontsize=11)
    plt.ylim(0, 800)  # 聚焦主要长度范围
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    save_file = save_path / f"{split}_length_vs_sentiment.png"
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"{split}集文本长度-情感对比图已保存至: {save_file}")


if __name__ == "__main__":
    # 加载数据（适配train.py的load_data返回二元组）
    df_train, df_test = load_data()

    # 为训练集和测试集生成所有分析图表
    plot_configs = [
        ("train", df_train),
        ("test", df_test)
    ]

    for split, df in plot_configs:
        print(f"\n=== 生成{split}集分析图表 ===")
        plot_text_length_distribution(df, split=split)
        plot_sentiment_balance(df, split=split)
        plot_rating_sentiment_mapping(df, split=split)
        plot_sentiment_word_distribution(df, split=split)
        plot_text_length_vs_sentiment(df, split=split)

    print("\n所有数据集分析图表生成完成！")