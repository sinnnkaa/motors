import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score
import random
import seaborn as sns
from model import tta_predict
import numpy as np

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


def visualize_results(histories, models, X_raw, X_feat, y_encoded, le):
    print("4. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")

    plt.figure(figsize=(20, 12))
    
    for i, history in enumerate(histories):
        plt.subplot(2, 3, i+1)
        plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        plt.title(f'Fold {i+1} - Точность модели', fontsize=12, fontweight='bold')
        plt.xlabel('Эпоха')
        plt.ylabel('Точность')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, i+4)
        plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
        plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        plt.title(f'Fold {i+1} - Функция потерь', fontsize=12, fontweight='bold')
        plt.xlabel('Эпоха')
        plt.ylabel('Потери')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

    print("ОЦЕНКА С TTA")
    preds = tta_predict(models, X_raw, X_feat, tta_rounds=6)
    y_pred = np.argmax(preds, axis=1)
    
    print("report (ensemble + TTA):")
    print(classification_report(y_encoded, y_pred, target_names=le.classes_))
    print("Accuracy:", accuracy_score(y_encoded, y_pred))