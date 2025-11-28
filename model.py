import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold

from keras.models import Model
from keras.layers import Input, Conv1D, GlobalAveragePooling1D, concatenate, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import tensorflow as tf
import joblib

from features import sparse_focal_loss, data_generator, create_balanced_segments_with_raw, jitter, scaling
from plots import visualize_results


def build_two_branch_model(segment_size, n_features, num_classes, lr=1e-3, dropout_rate=0.25):
    # ВЕТВЬ ОБРАБОТКИ СЫРЫХ ДАННЫХ (RAW BRANCH)
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

    # ВЕТВЬ ОБРАБОТКИ ИЗВЛЕЧЕННЫХ ПРИЗНАКОВ (FEATURE BRANCH)
    feat_input = Input(shape=(n_features,), name='feat_input')
    f = Dense(128, activation='relu')(feat_input)
    f = BatchNormalization()(f)
    f = Dropout(dropout_rate)(f)
    f = Dense(64, activation='relu')(f)

    # СЛИЯНИЕ ПРИЗНАКОВ 
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
    
    print("\nМОДЕЛЬ СОХРАНЕНА")
    print("   - two_branch_fold{1,2,3}.keras")
    print("   - scaler_two_branch.pkl") 
    print("   - label_encoder_two_branch.pkl")
    
    return models, scaler, le, histories


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
            ModelCheckpoint(f'best_model_fold{fold+1}.keras', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
        ]

        steps_per_epoch = max(1, int(len(train_idx) / batch_size))
        train_gen = data_generator(X_raw_tr, X_feat_tr, y_tr, batch_size=batch_size, augment=True)

        history = model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            validation_data=([X_raw_val, X_feat_val], y_val), 
            validation_steps= None,
            epochs=epochs,
            callbacks=callbacks,
            verbose=2
        )
        histories.append(history)
        model.save(f"two_branch_fold{fold+1}.keras")
        models.append(model)

    joblib.dump(scaler, "scaler_two_branch.pkl")
    joblib.dump(le, "label_encoder_two_branch.pkl")

    return models, scaler, le, histories


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
