import matplotlib.pyplot as plt

import seaborn as sns

def plot_class_distribution(combined_df):
    plt.figure(figsize=(14,6))
    counts = combined_df['category'].value_counts()
    sns.barplot(x=counts.index, y=counts.values)
    plt.xticks(rotation=90)
    plt.xlabel("Категория", fontsize=12)
    plt.ylabel("Количество записей", fontsize=12)
    plt.title("Распределение классов (дисбаланс)", fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_feature_correlation(features_df):
    plt.figure(figsize=(16,12))
    corr = features_df.drop(columns=['category']).corr()
    sns.heatmap(corr, cmap='viridis')
    plt.title("Корреляционная матрица признаков", fontsize=14)
    plt.show()

def plot_feature_distributions(features_df):
    feat_cols = features_df.drop(columns=['category']).columns[:10]  # первые 10 признаков
    features_df[feat_cols].hist(figsize=(16,10), bins=30)
    plt.suptitle("Распределение признаков", fontsize=14)
    plt.show()

def plot_raw_segment(X_raw, idx=0):
    seg = X_raw[idx]
    axes = ['AccX', 'AccY', 'AccZ']

    plt.figure(figsize=(14,10))

    for i in range(3):
        plt.subplot(3,1,i+1)
        plt.plot(seg[:,i])
        plt.title(f"Сигнал {axes[i]}", fontsize=12)
        plt.xlabel("Индекс точки", fontsize=10)
        plt.ylabel("Амплитуда", fontsize=10)
        plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

