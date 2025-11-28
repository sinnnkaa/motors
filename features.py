import pandas as pd
import numpy as np
import random
from scipy.signal import welch
from scipy.stats import kurtosis, skew, iqr
import tensorflow as tf

from plots import plot_feature_distributions, plot_feature_correlation, plot_raw_segment


# ФУНКЦИИ ДЛЯ ИЗВЛЕЧЕНИЯ ПРИЗНАКОВ
def spectral_entropy(Pxx):
    """
    Мера хаотичности спектра сигнала
    Формула: H = -Σ(P_i * log(P_i))
    """
    P = Pxx / (np.sum(Pxx) + 1e-12)
    P = P[P > 0]
    return -np.sum(P * np.log(P + 1e-12))


def band_energy(f, Pxx, bands):
    """
    Доля энергии сигнала в определенных частотных диапазонах
    Формула: E_band = Σ(Pxx[band]) / Σ(Pxx)
    """
    energies = {}
    total = np.sum(Pxx) + 1e-12
    for (low, high) in bands:
        mask = (f >= low) & (f < high)
        energies[f"{low}_{high}_energy"] = np.sum(Pxx[mask]) / total
    return energies


def top_n_fft_peaks(signal, fs=1000, n=3):
    """
    Определение основных частотных составляющих вибрационного сигнала
    1. Вычисление FFT с окном Ханна для уменьшения утечки спектра
    2. Игнорирование постоянной составляющей (freq[0])
    3. Выбор N (3) наибольших амплитуд и их частот
    """
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
    """
    MEAN - Среднее значение - Постоянная составляющая сигнала
    STD - Стандартное отклонение - Средняя амплитуда колебаний
    MEDIAN - Медиана - Центральное значение, устойчивое к выбросам
    IQR - Интерквартильный размах - Разброс центральных 50% данных
    RMS - Среднеквадратичное значение - Эффективная амплитуда сигнала
    MAX/MIN - Максимум/минимум - Пиковые значения акселерометра
    SKEW - Асимметрия распределения - Скошенность гистограммы значений
    KURTOSIS - Эксцесс - Островершинность распределения
    ZCR - Частота пересечения нуля - Количество переходов через ноль в секунду
    SPEC_CENTROID - Спектральный центроид - Средневзвешенная частота спектра
    PEAK_RATIO - Отношение пиковой энергии - индикатор наличия доминирующей частоты
    CORR_XY, CORR_XZ, CORR_YZ - Межосевые корреляции
    TOTAL_RMS - Суммарное cреднеквадратичное значение векторной суммы ускорений
    TOTAL_STD - Суммарное стандартное отклонение
    """
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
    """
    1 Для каждой категории оборудования вычисляется максимально 
    возможное количество сегментов на основе длины временного 
    ряда. 
    2 Целевое количество сегментов на класс устанавливается 
    как медианное значение среди всех доступных классов. 
    3 Для каждого класса производится равномерное извлечение 
    сегментов по всей длине временного ряда. 
    """
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
    plot_feature_correlation(features_df)
    plot_feature_distributions(features_df)
    plot_raw_segment(X_raw)
    return features_df, X_raw, y_labels

# АУГМЕНТАЦИИ
def jitter(x, sigma=0.005):
    """
    Добавление гауссова шума к сигналу
    """
    return x + np.random.normal(loc=0.0, scale=sigma, size=x.shape)

def scaling(x, sigma=0.1):
    """
    Случайное масштабирование
    """
    factor = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[1],))
    return x * factor

def time_shift(x, shift_max=0.1):
    """
    Временной сдвиг
    """
    shift = int(np.random.uniform(-shift_max, shift_max) * x.shape[0])
    return np.roll(x, shift, axis=0)

# ГЕНЕРАТОР ДАННЫХ
def data_generator(X_raw, X_feat, y, batch_size=32, augment=True):
    """
    Генератор применяет аугментации последовательно и вероятностно к каждому примеру в батче
    Каждая аугментация рассматривается независимо
    Могут применяться несколько аугментаций к одному примеру
    """
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
    """
    Взвешенная функция потерь
    Решение проблемы дисбаланса классов на уровне функции потерь
    путем придания большего веса сложным для классификации примерам.
    """
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