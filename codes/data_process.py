import pandas as pd
import os
from pathlib import Path
from config import Config

# 原始数据集路径
RAW_DATA_DIR = Path("data/aclImdb")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(exist_ok=True)


def parse_review_file(file_path, split, label):
    """解析单条评论文件（兼容无标签数据）"""
    file_name = file_path.stem
    movie_id, rating = file_name.split("_")
    rating = int(rating)

    with open(file_path, "r", encoding="utf-8") as f:
        review = f.read().strip().replace("<br />", " ").replace("\n", " ")

    return {
        "movie_id": movie_id,
        "rating": rating,
        "review": review,
        "label": label,  # 1表示正面评价，0表示负面评价，-1表示无监督/无标签
        "split": split,
        "data_type": "supervised" if label in [0, 1] else "unsupervised"
    }


# 遍历文件夹
folders_to_process = [
    ("test/neg", 0, "test"),
    ("test/pos", 1, "test"),
    ("train/neg", 0, "train"),
    ("train/pos", 1, "train"),
    ("train/unsup", -1, "train")
]

all_data = []
for folder_rel_path, label, split in folders_to_process:
    folder_full_path = RAW_DATA_DIR / folder_rel_path
    if not folder_full_path.exists():
        print(f"警告：文件夹 {folder_full_path} 不存在，跳过")
        continue
    for review_file in folder_full_path.glob("*.txt"):
        try:
            data = parse_review_file(review_file, split, label)
            all_data.append(data)
        except Exception as e:
            print(f"解析文件 {review_file} 失败：{e}")

df = pd.DataFrame(all_data)
print(f"原始数据总量（含无监督）：{len(df)}")
print(f"数据类型分布：\n{df.groupby(['data_type', 'split', 'label']).size()}")

# ========== 数据筛选 ==========
# pretest. 统一筛选有效评分（≤4或≥7）
df_filtered = df[df["rating"].apply(lambda x: x <= 4 or x >= 7)].copy()
# 2. 统一限制每部电影最多30条评论
df_filtered = df_filtered.groupby("movie_id").head(30).reset_index(drop=True)
# 3. 分离有监督/无监督数据
df_supervised = df_filtered[df_filtered["data_type"] == "supervised"].copy()
df_unsupervised = df_filtered[df_filtered["data_type"] == "unsupervised"].copy()

# ========== 优化：电影ID去重逻辑（避免测试集被清空） ==========
train_movie_ids = set(df_supervised[df_supervised["split"] == "train"]["movie_id"])
test_movie_ids = set(df_supervised[df_supervised["split"] == "test"]["movie_id"])
overlap_movie_ids = train_movie_ids & test_movie_ids
overlap_ratio = len(overlap_movie_ids) / len(test_movie_ids) if len(test_movie_ids) > 0 else 0

print(f"\n=== 电影ID重叠分析 ===")
print(f"训练集电影ID数量：{len(train_movie_ids)}")
print(f"测试集电影ID数量：{len(test_movie_ids)}")
print(f"重叠电影ID数量：{len(overlap_movie_ids)}")
print(f"测试集电影ID重叠比例：{overlap_ratio:.2%}")

# 仅当重叠比例<50%时才剔除重叠数据（避免测试集被清空）
if overlap_ratio < 0.5:
    df_supervised = df_supervised[
        ~((df_supervised["split"] == "test") & (df_supervised["movie_id"].isin(overlap_movie_ids)))]
    print(f"剔除重叠电影ID后，有监督数据量：{len(df_supervised)}")
else:
    print(f"警告：测试集电影ID重叠比例过高（{overlap_ratio:.2%}），跳过剔除操作（避免测试集清空）")


# ========== 修复：动态采样（兼容空DataFrame） ==========
def balance_supervised_data(df_sup, split, target_total=25000):
    """
    平衡有监督数据（保证正负样本数量一致，兼容空DataFrame）
    """
    # 筛选当前划分的数据
    df_split = df_sup[df_sup["split"] == split].copy()
    # 初始化空DataFrame（保留列结构）
    balanced_df = pd.DataFrame(columns=["movie_id", "rating", "review", "label", "split", "data_type"])

    if len(df_split) == 0:
        print(f"警告：{split}集无可用的有监督数据")
        return balanced_df

    # 分别获取正负样本
    pos_df = df_split[df_split["label"] == 1]
    neg_df = df_split[df_split["label"] == 0]

    # 计算每类的最大可用样本数
    pos_count = len(pos_df)
    neg_count = len(neg_df)
    max_per_class = min(pos_count, neg_count, target_total // 2)

    # 采样（不足时取全部）
    pos_sampled = pos_df.sample(n=max_per_class, random_state=42) if pos_count > 0 else pd.DataFrame()
    neg_sampled = neg_df.sample(n=max_per_class, random_state=42) if neg_count > 0 else pd.DataFrame()

    # 合并并返回
    balanced_df = pd.concat([pos_sampled, neg_sampled]).reset_index(drop=True)
    print(f"{split}集平衡后：")
    print(f"  - 正面样本数：{len(pos_sampled)}（原始：{pos_count}）")
    print(f"  - 负面样本数：{len(neg_sampled)}（原始：{neg_count}）")
    print(f"  - 总样本数：{len(balanced_df)}（目标：{target_total}）")
    return balanced_df


# 平衡训练集
train_sup_balanced = balance_supervised_data(df_supervised, "train")
# 平衡测试集
test_sup_balanced = balance_supervised_data(df_supervised, "test")

# 合并最终有监督数据
final_supervised = pd.concat([train_sup_balanced, test_sup_balanced]).reset_index(drop=True)
# 无监督数据：直接保留
final_unsupervised = df_unsupervised.reset_index(drop=True)

print(f"\n最终有监督数据总量：{len(final_supervised)}")
print(f"最终无监督数据总量：{len(final_unsupervised)}")


# ========== 修复：清洗数据（兼容空DataFrame） ==========
def clean_review(text):
    return " ".join(text.split())


# 清洗训练集（仅当非空时）
if len(train_sup_balanced) > 0:
    train_sup_balanced["cleaned_review"] = train_sup_balanced["review"].apply(clean_review)
else:
    train_sup_balanced["cleaned_review"] = []

# 清洗测试集（仅当非空时）
if len(test_sup_balanced) > 0:
    test_sup_balanced["cleaned_review"] = test_sup_balanced["review"].apply(clean_review)
else:
    test_sup_balanced["cleaned_review"] = []

# 清洗总有监督数据
if len(final_supervised) > 0:
    final_supervised["cleaned_review"] = final_supervised["review"].apply(clean_review)

# 清洗无监督数据
if len(final_unsupervised) > 0:
    final_unsupervised["cleaned_review"] = final_unsupervised["review"].apply(clean_review)

# ========== 修复：保存CSV（仅当非空时） ==========
# pretest. 有监督总数据（仅当非空时保存）
if len(final_supervised) > 0:
    final_supervised_output = final_supervised[["movie_id", "rating", "label", "split", "cleaned_review"]]
    final_supervised_output.to_csv(OUTPUT_DIR / "imdb_supervised_full.csv", index=False, encoding="utf-8")
else:
    print("警告：有监督总数据为空，跳过保存")

# 2. 有监督训练集（仅当非空时保存）
if len(train_sup_balanced) > 0:
    train_sup_output = train_sup_balanced[["movie_id", "rating", "label", "cleaned_review"]]
    train_sup_output.to_csv(OUTPUT_DIR / "imdb_supervised_train.csv", index=False, encoding="utf-8")
else:
    print("警告：有监督训练集为空，跳过保存")

# 3. 有监督测试集（仅当非空时保存）
if len(test_sup_balanced) > 0:
    test_sup_output = test_sup_balanced[["movie_id", "rating", "label", "cleaned_review"]]
    test_sup_output.to_csv(OUTPUT_DIR / "imdb_supervised_test.csv", index=False, encoding="utf-8")
else:
    print("警告：有监督测试集为空，跳过保存")

# 4. 无监督数据（仅当非空时保存）
if len(final_unsupervised) > 0:
    final_unsupervised_output = final_unsupervised[["movie_id", "rating", "label", "split", "cleaned_review"]]
    final_unsupervised_output.to_csv(OUTPUT_DIR / "imdb_unsupervised_train.csv", index=False, encoding="utf-8")
else:
    print("警告：无监督数据为空，跳过保存")

print(f"\n所有非空CSV文件已保存至：{OUTPUT_DIR}")