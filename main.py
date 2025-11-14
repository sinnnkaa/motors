import pandas as pd
import numpy as np
import os
import glob
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import random
from scipy.signal import welch
from scipy.stats import kurtosis, skew, iqr

from keras.models import Model
from keras.layers import Input, Conv1D, GlobalAveragePooling1D, concatenate, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import tensorflow as tf
import joblib

from categories import category_map

# ЗАГРУЗКА ДАННЫХ
PATH = "C:/Users/User/Downloads/archive (2)/files"  

def get_category_by_number(file_name):
    file_number = file_name.split(' - ')[0]   
    return category_map.get(file_number, 'Unknown')

def load_and_prepare_data():
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

# ФУНКЦИИ ДЛЯ ИЗВЛЕЧЕНИЯ ПРИЗНАКОВ
def spectral_entropy(Pxx):
    P = Pxx / (np.sum(Pxx) + 1e-12)
    P = P[P > 0]
    return -np.sum(P * np.log(P + 1e-12))

def band_energy(f, Pxx, bands):
    energies = {}
    total = np.sum(Pxx) + 1e-12
    for (low, high) in bands:
        mask = (f >= low) & (f < high)
        energies[f"{low}_{high}_energy"] = np.sum(Pxx[mask]) / total
    return energies

def top_n_fft_peaks(signal, fs=1000, n=3):
    N = len(signal)
    if N < 3:
        return [0.0]*n
    fft = np.abs(np.fft.rfft(signal * np.hanning(N)))
    freqs = np.fft.rfftfreq(N, d=1/fs)
    fft[0] = 0
    peaks_idx = np.argsort(fft)[-n:][::-1]
    peaks = [freqs[idx] for idx in peaks_idx]
    while len(peaks) < n:
        peaks.append(0.0)
    return peaks

def extract_general_features(group, fs=1000):
    features = {}
    bands = [(0,50),(50,150),(150,300),(300,500)]
    for axis in ['AccX','AccY','AccZ']:
        data = group[axis].values.astype(float)
        N = len(data)
        features[f'{axis}_mean'] = np.mean(data)
        features[f'{axis}_std'] = np.std(data)
        features[f'{axis}_median'] = np.median(data)
        features[f'{axis}_iqr'] = iqr(data)
        features[f'{axis}_rms'] = np.sqrt(np.mean(data**2))
        features[f'{axis}_max'] = np.max(data)
        features[f'{axis}_min'] = np.min(data)
        features[f'{axis}_skew'] = skew(data)
        features[f'{axis}_kurtosis'] = kurtosis(data)
        features[f'{axis}_zcr'] = ((data[:-1] * data[1:]) < 0).sum() / max(1, N-1)

        f, Pxx = welch(data, fs=fs, nperseg=min(1024, N))
        features[f'{axis}_spec_centroid'] = np.sum(f*Pxx)/(np.sum(Pxx)+1e-12)
        features[f'{axis}_spec_entropy'] = spectral_entropy(Pxx)
        be = band_energy(f, Pxx, bands)
        for k,v in be.items():
            features[f'{axis}_band_{k}'] = v
        pks = top_n_fft_peaks(data, fs=fs, n=3)
        for i,pk in enumerate(pks):
            features[f'{axis}_fft_peak_{i+1}'] = pk

        peak_energy = np.max(Pxx) if len(Pxx)>0 else 0.0
        total_energy = np.sum(Pxx) + 1e-12
        features[f'{axis}_peak_ratio'] = peak_energy / total_energy

    features['corr_xy'] = np.corrcoef(group['AccX'], group['AccY'])[0,1]
    features['corr_xz'] = np.corrcoef(group['AccX'], group['AccZ'])[0,1]
    features['corr_yz'] = np.corrcoef(group['AccY'], group['AccZ'])[0,1]

    total = np.sqrt(group['AccX'].values**2 + group['AccY'].values**2 + group['AccZ'].values**2)
    features['total_rms'] = np.sqrt(np.mean(total**2))
    features['total_std'] = np.std(total)
    return features

# СОЗДАНИЕ СЕГМЕНТОВ С СЫРЫМИ ДАННЫМИ
def create_balanced_segments_with_raw(combined_df, base_segment_size=1000, overlap=0.4, target_segments_per_class=None, random_state=42, fs=1000):
    random.seed(random_state)
    features_list = []
    raw_segments = []
    labels = []
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
                features = extract_general_features(segment, fs=fs)
                features['category'] = category
                features_list.append(features)

                raw = np.stack([
                    segment['AccX'].values.astype(float),
                    segment['AccY'].values.astype(float),
                    segment['AccZ'].values.astype(float),
                ], axis=1)
                raw_segments.append(raw)
                labels.append(category)
            except Exception as e:
                print(f"Ошибка при обработке сегмента {category} start={start}: {e}")
                continue

    features_df = pd.DataFrame(features_list).reset_index(drop=True)
    X_raw = np.array(raw_segments)
    y_labels = np.array(labels)
    return features_df, X_raw, y_labels

# АУГМЕНТАЦИИ
def jitter(x, sigma=0.005):
    return x + np.random.normal(loc=0.0, scale=sigma, size=x.shape)

def scaling(x, sigma=0.1):
    factor = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[1],))
    return x * factor

def time_shift(x, shift_max=0.1):
    shift = int(np.random.uniform(-shift_max, shift_max) * x.shape[0])
    return np.roll(x, shift, axis=0)

# ГЕНЕРАТОР ДАННЫХ
def data_generator(X_raw, X_feat, y, batch_size=32, augment=True):
    n = X_raw.shape[0]
    idx = np.arange(n)
    while True:
        np.random.shuffle(idx)
        for i in range(0, n, batch_size):
            batch_idx = idx[i:i+batch_size]
            raw_batch = X_raw[batch_idx].copy()
            feat_batch = X_feat[batch_idx].astype(np.float32)
            y_batch = y[batch_idx]
            if augment:
                for j in range(len(raw_batch)):
                    if np.random.rand() < 0.5:
                        raw_batch[j] = jitter(raw_batch[j], sigma=0.01)
                    if np.random.rand() < 0.4:
                        raw_batch[j] = scaling(raw_batch[j], sigma=0.08)
                    if np.random.rand() < 0.3:
                        raw_batch[j] = time_shift(raw_batch[j], shift_max=0.08)
            yield (raw_batch, feat_batch), y_batch


def sparse_focal_loss(gamma=2.0, alpha=None):
    def loss(y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())
        y_true_oh = tf.one_hot(y_true, depth=tf.shape(y_pred)[-1])
        cross_entropy = -tf.reduce_sum(y_true_oh * tf.math.log(y_pred), axis=-1)
        p_t = tf.reduce_sum(y_true_oh * y_pred, axis=-1)
        modulating_factor = tf.pow(1.0 - p_t, gamma)
        if alpha is not None:
            alpha_tensor = tf.constant(alpha, dtype=tf.float32)
            alpha_factor = tf.reduce_sum(y_true_oh * alpha_tensor, axis=-1)
            return alpha_factor * modulating_factor * cross_entropy
        else:
            return modulating_factor * cross_entropy
    return loss

# ДВУХВЕТВЕВАЯ МОДЕЛЬ
def build_two_branch_model(segment_size, n_features, num_classes, lr=1e-3, dropout_rate=0.25):
    # raw branch
    raw_input = Input(shape=(segment_size, 3), name='raw_input')
    x = Conv1D(64, kernel_size=16, strides=2, padding='same', activation='relu')(raw_input)
    x = BatchNormalization()(x)
    x = Conv1D(128, kernel_size=8, strides=2, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv1D(256, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(dropout_rate/2)(x)

    # feature branch
    feat_input = Input(shape=(n_features,), name='feat_input')
    f = Dense(128, activation='relu')(feat_input)
    f = BatchNormalization()(f)
    f = Dropout(dropout_rate)(f)
    f = Dense(64, activation='relu')(f)

    # fusion
    merged = concatenate([x, f])
    m = Dense(128, activation='relu')(merged)
    m = BatchNormalization()(m)
    m = Dropout(dropout_rate)(m)
    m = Dense(64, activation='relu')(m)
    out = Dense(num_classes, activation='softmax')(m)

    model = Model(inputs=[raw_input, feat_input], outputs=out)
    opt = Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss=sparse_focal_loss(gamma=2.0), metrics=['accuracy'])
    return model

# TTA ПРЕДСКАЗАНИЕ
def tta_predict(models, X_raw, X_feat, tta_rounds=5):
    preds_sum = np.zeros((X_feat.shape[0], models[0].output_shape[-1]))
    for t in range(tta_rounds):
        X_raw_aug = X_raw.copy()
        for i in range(X_raw_aug.shape[0]):
            if np.random.rand() < 0.5:
                X_raw_aug[i] = jitter(X_raw_aug[i], sigma=0.008)
            if np.random.rand() < 0.3:
                X_raw_aug[i] = scaling(X_raw_aug[i], sigma=0.06)
        preds_models = np.zeros_like(preds_sum)
        for m in models:
            preds_models += m.predict([X_raw_aug, X_feat], batch_size=128, verbose=0)
        preds_sum += preds_models / len(models)
    preds_avg = preds_sum / tta_rounds
    return preds_avg

# ОСНОВНОЕ ОБУЧЕНИЕ 
def train_keras_ensemble(combined_df,
                         base_segment_size=1000,
                         overlap=0.4,
                         folds=3,
                         epochs=50,
                         batch_size=32,
                         fs=1000,
                         random_state=42):
    
    print("1. СОЗДАНИЕ СЕГМЕНТОВ С СЫРЫМИ ДАННЫМИ")
    features_df, X_raw, y_labels = create_balanced_segments_with_raw(
        combined_df,
        base_segment_size=base_segment_size,
        overlap=overlap,
        target_segments_per_class=None,
        random_state=random_state,
        fs=fs
    )

    print("2. ПОДГОТОВКА ДАННЫХ")
    X_feat = features_df.drop(columns=['category']).fillna(0).values.astype(np.float32)
    scaler = StandardScaler()
    X_feat = scaler.fit_transform(X_feat)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_labels)

    segment_size = X_raw.shape[1]
    n_features = X_feat.shape[1]
    num_classes = len(le.classes_)

    print(f"Размеры данных:")
    print(f"X_raw: {X_raw.shape}")
    print(f"X_feat: {X_feat.shape}")
    print(f"Количество классов: {num_classes}")

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    models = []
    histories = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_feat, y_encoded)):
        print(f"\n--- Fold {fold+1}/{folds} ---")
        X_raw_tr, X_raw_val = X_raw[train_idx], X_raw[val_idx]
        X_feat_tr, X_feat_val = X_feat[train_idx], X_feat[val_idx]
        y_tr, y_val = y_encoded[train_idx], y_encoded[val_idx]

        model = build_two_branch_model(segment_size, n_features, num_classes, lr=1e-3, dropout_rate=0.25)
        if fold == 0: model.summary()
        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True, verbose=1, mode='max'),
            ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=8, min_lr=1e-6, verbose=1, mode='max'),
            ModelCheckpoint(f'best_model_fold{fold+1}.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
        ]

        steps_per_epoch = max(1, int(len(train_idx) / batch_size))
        val_steps = max(1, int(len(val_idx) / batch_size))

        train_gen = data_generator(X_raw_tr, X_feat_tr, y_tr, batch_size=batch_size, augment=True)
        val_gen = data_generator(X_raw_val, X_feat_val, y_val, batch_size=batch_size, augment=False)

        history = model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_gen,
            validation_steps=val_steps,
            epochs=epochs,
            callbacks=callbacks,
            verbose=2
        )
        histories.append(history)
        model.save(f"two_branch_fold{fold+1}.h5")
        models.append(model)

    joblib.dump(scaler, "scaler_two_branch.pkl")
    joblib.dump(le, "label_encoder_two_branch.pkl")

    return models, scaler, le, histories

# ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ
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

    print("5. ФИНАЛЬНАЯ ОЦЕНКА С TTA")
    preds = tta_predict(models, X_raw, X_feat, tta_rounds=6)
    y_pred = np.argmax(preds, axis=1)
    
    print("Final report (ensemble + TTA):")
    print(classification_report(y_encoded, y_pred, target_names=le.classes_))
    print("Accuracy:", accuracy_score(y_encoded, y_pred))

# ОСНОВНАЯ ФУНКЦИЯ ОБУЧЕНИЯ
def train_model(combined_df):
    """
    ОСНОВНОЙ ПРОЦЕСС ОБУЧЕНИЯ
    """
    print("ЗАПУСК МОДЕЛИ")

    models, scaler, le, histories = train_keras_ensemble(
        combined_df,
        base_segment_size=1000,
        overlap=0.4,
        folds=3,
        epochs=50,
        batch_size=32
    )

    features_df, X_raw, y_labels = create_balanced_segments_with_raw(
        combined_df,
        base_segment_size=1000,
        overlap=0.4
    )
    X_feat = features_df.drop(columns=['category']).fillna(0).values.astype(np.float32)
    X_feat = scaler.transform(X_feat)
    y_encoded = le.transform(y_labels)

    visualize_results(histories, models, X_raw, X_feat, y_encoded, le)
    
    print("6. ТЕСТИРОВАНИЕ НА ОДНОЙ СТРОКЕ:")
    test_indices = random.sample(range(len(X_raw)), 3)
    
    for i, idx in enumerate(test_indices):
        single_raw = X_raw[idx:idx+1]
        single_feat = X_feat[idx:idx+1]
        true_label_encoded = y_encoded[idx]
        true_label = le.inverse_transform([true_label_encoded])[0]

        prediction_proba = tta_predict(models, single_raw, single_feat, tta_rounds=3)
        prediction_class = np.argmax(prediction_proba, axis=1)[0]
        predicted_label = le.inverse_transform([prediction_class])[0]
        confidence = np.max(prediction_proba)
        
        status = "ПРАВИЛЬНО" if predicted_label == true_label else "ОШИБКА"
        
        print(f"\nПРИМЕР {i+1}:")
        print(f"   Предсказанный класс: {predicted_label}")
        print(f"   Истинный класс:    {true_label}")
        print(f"   Уверенность:       {confidence:.4f} ({confidence*100:.2f}%)")
        print(f"   Статус:            {status}")
    
    print("\nМОДЕЛЬ СОХРАНЕНА")
    print("   - two_branch_fold{1,2,3}.h5")
    print("   - scaler_two_branch.pkl") 
    print("   - label_encoder_two_branch.pkl")
    
    return models, scaler, le, histories

if __name__ == "__main__":
    try:
        combined_df = load_and_prepare_data()
        models, scaler, le, histories = train_model(combined_df)

    except Exception as e:
        print(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()