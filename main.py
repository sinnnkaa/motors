import pandas as pd
import numpy as np
import os
import glob
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from scipy.signal import welch
from scipy.stats import kurtosis, skew
import matplotlib.pyplot as plt
import seaborn as sns
import random
from keras.regularizers import l2
from keras.losses import sparse_categorical_crossentropy
from categories import category_map

# ЗАГРУЗКА ДАННЫХ

PATH = "C:/Users/User/Downloads/archive (2)/files"  


def get_category_by_number(file_name):
    """
    ПРИСВОЕНИЕ НОМЕРА КАТЕГОРИИ
    """
    file_number = file_name.split(' - ')[0]   
    return category_map.get(file_number, 'Unknown')

def load_and_prepare_data():
    """
    ОБРАБОТКА КСВ ФАЙЛОВ + СОЗДАНИЕ ДФ
    """
    print("ЗАГРУЗКА ДАННЫХ")
    csv_files = glob.glob(os.path.join(PATH, "*.csv"))
    print(f"Найдено файлов: {len(csv_files)}")

    all_data = []

    for file in csv_files:
        file_name = os.path.basename(file)
        
        try:
            category = get_category_by_number(file_name)
            df = pd.read_csv(file)
            df['category'] = category
            all_data.append(df)
            print(f"Загружен: {file_name}")

        except Exception as e:
            print(f"Ошибка загрузки {file_name}: {e}")

    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = combined_df.dropna()
        
        print("\nОБЗОР ДАННЫХ:")
        print(f"Общий размер: {combined_df.shape}")
        print(f"Общее количество записей: {len(combined_df):,}")
        
        print("\nРАСПРЕДЕЛЕНИЕ ПО КАТЕГОРИЯМ:")
        category_counts = combined_df['category'].value_counts()
        for category, count in category_counts.items():
            percentage = (count / len(combined_df)) * 100
            print(f"  {category}: {count:,} записей ({percentage:.1f}%)")

        return combined_df
    else:
        raise Exception("Не удалось загрузить данные")
    
# ДОБАВЛЕНИЕ ПРИЗНАКОВ

def extract_general_features(group):
    """
    ДОБАВЛЕНИЕ ПРИЗНАКОВ ИСХОДЯ ИЗ ПОКАЗАНИЙ АКСЕЛЕРОМЕТРА
    """
    features = {}
    for axis in ['AccX','AccY','AccZ']:
        data = group[axis].values
        features[f'{axis}_mean'] = np.mean(data)
        features[f'{axis}_std'] = np.std(data)
        features[f'{axis}_rms'] = np.sqrt(np.mean(data**2))
        features[f'{axis}_max'] = np.max(data)
        features[f'{axis}_min'] = np.min(data)
        features[f'{axis}_skew'] = skew(data)
        features[f'{axis}_kurtosis'] = kurtosis(data)

        f, Pxx = welch(data, fs=1000, nperseg=min(256,len(data)))
        features[f'{axis}_spec_centroid'] = np.sum(f*Pxx)/(np.sum(Pxx)+1e-8)

    features['corr_xy'] = np.corrcoef(group['AccX'], group['AccY'])[0,1]
    features['corr_xz'] = np.corrcoef(group['AccX'], group['AccZ'])[0,1]
    features['corr_yz'] = np.corrcoef(group['AccY'], group['AccZ'])[0,1]
    return features


def create_balanced_segments(combined_df, base_segment_size=1000, overlap=0.4, target_segments_per_class=None, random_state=42):
    """
    СОЗДАНИЕ И БАЛАНС СЕГМЕНТОВ
    """
    random.seed(random_state)
    features_list = []
    per_class_available = {}

    for category in combined_df['category'].unique():
        cat_df = combined_df[combined_df['category'] == category].reset_index(drop=True)
        segment_size = base_segment_size
        step_size = int(segment_size * (1 - overlap))
        if step_size <= 0:
            step_size = max(1, segment_size // 2)
        if len(cat_df) < segment_size:
            n_segments = 0
        else:
            n_segments = (len(cat_df) - segment_size) // step_size + 1
        per_class_available[category] = max(0, n_segments)

    available_counts = [v for v in per_class_available.values() if v > 0]
    if len(available_counts) == 0:
        raise ValueError("Нет доступных сегментов")
    if target_segments_per_class is None:
        target_segments_per_class = int(np.median(available_counts))
        target_segments_per_class = max(1, target_segments_per_class)

    print("Доступные сегменты по классам:", per_class_available)
    print(f"Целевое число сегментов на класс: {target_segments_per_class}")

    for category, n_available in per_class_available.items():
        if n_available == 0:
            continue

        cat_df = combined_df[combined_df['category'] == category].reset_index(drop=True)
        segment_size = base_segment_size
        step_size = int(segment_size * (1 - overlap))
        step_size = max(1, step_size)

        start_indices = [i * step_size for i in range(n_available)]
        if len(start_indices) > target_segments_per_class:
            chosen_starts = random.sample(start_indices, target_segments_per_class)
        else:
            chosen_starts = start_indices

        for start in chosen_starts:
            end = start + segment_size
            if end > len(cat_df):
                continue
            segment = cat_df.iloc[start:end]
            try:
                features = extract_general_features(segment)
                features['category'] = category
                features_list.append(features)
            except Exception as e:
                print(f"Ошибка при обработке сегмента {category} start={start}: {e}")
                continue

    features_df = pd.DataFrame(features_list)
    return features_df

def add_electrical_specific_features(features_df):
    """
    ДОБАВЛЕНИЕ ПРИЗНАКОВ ДЛЯ ЭЛЕКТРИЧЕСТВА
    """

    required_energy_cols = ['AccX_energy', 'AccY_energy', 'AccZ_energy']
    for col in required_energy_cols:
        if col not in features_df.columns:
            features_df[col] = 0.0

    for axis in ['AccX', 'AccY', 'AccZ']:
        lf_col = f'{axis}_line_frequency_max'
        dlf_col = f'{axis}_double_line_freq_max'
        tlf_col = f'{axis}_triple_line_freq_max'

        if lf_col in features_df.columns and dlf_col in features_df.columns:
            features_df[f'{axis}_harmonic_ratio_2nd'] = (
                features_df[dlf_col] / (features_df[lf_col] + 1e-8)
            )
        else:
            features_df[f'{axis}_harmonic_ratio_2nd'] = 0

        if lf_col in features_df.columns and tlf_col in features_df.columns:
            features_df[f'{axis}_harmonic_ratio_3rd'] = (
                features_df[tlf_col] / (features_df[lf_col] + 1e-8)
            )
        else:
            features_df[f'{axis}_harmonic_ratio_3rd'] = 0

    features_df['vibration_asymmetry'] = (
        np.abs(features_df['AccX_energy'] - features_df['AccY_energy']) /
        (features_df['AccX_energy'] + features_df['AccY_energy'] + 1e-8)
    )

    for axis in ['AccX_std', 'AccY_std', 'AccZ_std']:
        if axis not in features_df.columns:
            features_df[axis] = 0.0

    features_df['stability_indicator'] = (
        features_df['AccX_std'] * features_df['AccY_std'] * features_df['AccZ_std']
    )

    for col in [
        'AccX_line_frequency_energy',
        'AccX_double_line_freq_energy',
        'AccX_triple_line_freq_energy'
    ]:
        if col not in features_df.columns:
            features_df[col] = 0.0

    features_df['total_harmonic_energy'] = (
        features_df['AccX_line_frequency_energy'] +
        features_df['AccX_double_line_freq_energy'] +
        features_df['AccX_triple_line_freq_energy']
    )

    return features_df


# МОДЕЛЬ НЕЙРОСЕТИ
def create_electrical_focused_model(input_dim, num_classes):

    model = Sequential([
        Dense(256, activation='relu', input_dim=input_dim, kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Dropout(0.35),

        Dense(128, activation='relu', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Dropout(0.25),

        Dense(64, activation='relu'),
        Dropout(0.2),

        Dense(32, activation='relu'),
        Dropout(0.15),

        Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss=sparse_categorical_crossentropy, 
        metrics=['accuracy']
    )
    return model


# ОСНОВНОЙ ПРОЦЕСС ОБУЧЕНИЯ
def train_model(combined_df):
    """
    ОБУЧЕНИЕ МОДЕЛИ
    """
    print("ЗАПУСК МОДЕЛИ")

    print("1. АНАЛИЗ ДАННЫХ:")
    print(f"Размер combined_df: {combined_df.shape}")
    print("Распределение категорий:")
    print(combined_df['category'].value_counts())

    print("\n2. СОЗДАНИЕ СЕГМЕНТОВ")
    features_df = create_balanced_segments(combined_df)
    
    print(f"Итоговый размер датасета: {features_df.shape}")
    print("Распределение по категориям:")
    print(features_df['category'].value_counts())

    print("\n3. ДОБАВЛЕНИЕ СПЕЦИАЛЬНЫХ ПРИЗНАКОВ")
    features_df = add_electrical_specific_features(features_df)
    print(f"Размер после добавления признаков: {features_df.shape}")

    print("\n4. ПОДГОТОВКА ДАННЫХ")
    X = features_df.drop('category', axis=1)
    y = features_df['category']

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"Количество классов: {len(label_encoder.classes_)}")
    print("Соответствие классов:")
    for i, class_name in enumerate(label_encoder.classes_):
        count = (y_encoded == i).sum()
        print(f"  {i}: {class_name} ({count} примеров)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.25, stratify=y_encoded, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\nРазмеры данных:")
    print(f"X_train: {X_train_scaled.shape}, y_train: {y_train.shape}")
    print(f"X_test: {X_test_scaled.shape}, y_test: {y_test.shape}")

    print("\n5. ВЫЧИСЛЕНИЕ ВЕСОВ КЛАССОВ")
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))
    
    electrical_classes = ['Electrical fault', 'Electrical fault with load', 
                         'Electrical fault with noise', 'Electrical fault with load and noise']
    for i, class_name in enumerate(label_encoder.classes_):
        print(f"  {class_name}: {class_weight_dict[i]:.3f}")

    print("\n6. СОЗДАНИЕ МОДЕЛИ")
    input_dim = X_train_scaled.shape[1]
    num_classes = len(label_encoder.classes_)
    
    model = create_electrical_focused_model(input_dim, num_classes)
    
    print(model.summary())

    callbacks = [
        EarlyStopping(
            patience=25, 
            restore_best_weights=True, 
            monitor='val_accuracy',
            mode='max',
            verbose=1
        ),
        ReduceLROnPlateau(
            factor=0.5, 
            patience=15, 
            min_lr=1e-7, 
            monitor='val_accuracy',
            mode='max',
            verbose=1
        ),
        ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
    ]

    print("\n7. ОБУЧЕНИЕ МОДЕЛИ")
    history = model.fit(
        X_train_scaled, y_train,
        batch_size=32,
        epochs=200,
        validation_data=(X_test_scaled, y_test),
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )
    
    print("\n8. ОЦЕНКА МОДЕЛИ")
    test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"ТОЧНОСТЬ НА ТЕСТОВОЙ ВЫБОРКЕ: {test_accuracy:.4f}")
    train_loss, train_accuracy = model.evaluate(X_train_scaled, y_train, verbose=0)
    print(f"ТОЧНОСТЬ НА ОБУЧАЮЩЕЙ ВЫБОРКЕ: {train_accuracy:.4f}")
    print(f"РАЗРЫВ TRAIN/VAL: {abs(train_accuracy - test_accuracy):.4f}")

    y_pred = model.predict(X_test_scaled, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    print("\n9. ДЕТАЛЬНЫЙ ОТЧЕТ:")
    print(classification_report(y_test, y_pred_classes, target_names=label_encoder.classes_))

    print("\n11. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ...")
    plt.figure(figsize=(20, 6))

    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    plt.title('Точность модели', fontsize=14, fontweight='bold')
    plt.xlabel('Эпоха')
    plt.ylabel('Точность')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # График потерь
    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    plt.title('Функция потерь', fontsize=14, fontweight='bold')
    plt.xlabel('Эпоха')
    plt.ylabel('Потери')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    print("\n12. СОХРАНЕНИЕ МОДЕЛИ")
    print("\n13. ТЕСТИРОВАНИЕ НА ОДНОЙ СТРОКЕ:")

    test_indices = random.sample(range(len(X_test_scaled)), 5)

    for i, idx in enumerate(test_indices):
        single_sample = X_test_scaled[idx:idx+1]
        true_label_encoded = y_test[idx]
        true_label = label_encoder.inverse_transform([true_label_encoded])[0]

        prediction_proba = model.predict(single_sample, verbose=0)
        prediction_class = np.argmax(prediction_proba, axis=1)[0]
        predicted_label = label_encoder.inverse_transform([prediction_class])[0]
        confidence = np.max(prediction_proba)
        
        status = "ПРАВИЛЬНО" if predicted_label == true_label else "ОШИБКА"
        
        print(f"\nПРИМЕР {i+1}:")
        print(f"   Предсказанный класс: {predicted_label}")
        print(f"   Истинный класс:    {true_label}")
        print(f"   Уверенность:       {confidence:.4f} ({confidence*100:.2f}%)")
        print(f"   Статус:            {status}")
        
        model.save("model.h5")
        
        import joblib
        joblib.dump(scaler, "scaler.pkl")
        joblib.dump(label_encoder, "label_encoder.pkl")
        
        print("МОДЕЛЬ СОХРАНЕНА!")
        print("   - model.h5")
        print("   - scaler.pkl") 
        print("   - label_encoder.pkl")
        
        return model, scaler, label_encoder, history


if __name__ == "__main__":
    try:
        combined_df = load_and_prepare_data()

        model, scaler, label_encoder, history = train_model(combined_df)

    except Exception as e:
        print(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()