# Система классификации вибрационных сигналов промышленного оборудования

## 1. Аннотация

Разработанная система представляет собой гибридную нейросетевую модель для автоматической диагностики состояния промышленного оборудования на основе анализа вибрационных сигналов. Модель демонстрирует хорошую точность классификации 16 различных состояний оборудования.

## 2. Основные характеристики

- **Точность**: 97% на сбалансированном датасете
- **Архитектура**: Двухветвевая модель CNN + MLP
- **Классы**: 16 категорий неисправностей и нормальных режимов
- **Параметры модели**: 252,752 (987.31 KB)
- **Метод валидации**: Stratified K-Fold (3)

## 3. Архитектура модели

### Ветвь обработки сырых данных
Input(1000, 3) - Conv1D(64) - Conv1D(128) - Conv1D(256) - GlobalAveragePooling - Dense(128)

### Ветвь обработки извлеченных признаков
Concatenate - Dense(128) - Dense(64) - Dense(16) - Softmax

## 4. Результаты классификации

### Метрики по классам

| Категория | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| Electrical fault | 1.00 | 0.97 | 0.99 | 299 |
| Electrical fault with load | 0.91 | 0.90 | 0.90 | 299 |
| Electrical fault with load and noise | 0.90 | 0.91 | 0.91 | 299 |
| Electrical fault with noise | 0.97 | 1.00 | 0.99 | 299 |
| Mechanical and Electrical fault | 0.99 | 0.99 | 0.99 | 299 |
| Mechanical and Electrical fault with load and noise | 1.00 | 1.00 | 1.00 | 299 |
| Mechanical and Electrical fault with noise | 1.00 | 0.99 | 0.99 | 299 |
| Mechanical fault (shaft misalignment) | 1.00 | 1.00 | 1.00 | 299 |
| Mechanical fault (shaft misalignment) with load | 1.00 | 1.00 | 1.00 | 251 |
| Mechanical fault with high noise | 1.00 | 1.00 | 1.00 | 257 |
| Mechanical fault with load and noise | 1.00 | 1.00 | 1.00 | 251 |
| Mechanical fault with noise | 1.00 | 1.00 | 1.00 | 255 |
| Normal operation | 0.96 | 0.88 | 0.91 | 178 |
| Normal operation with load | 0.98 | 0.96 | 0.97 | 170 |
| Normal operation with load and noise | 0.96 | 0.98 | 0.97 | 173 |
| Normal operation with noise | 0.88 | 0.96 | 0.92 | 169 |

### Общие метрики
- **Accuracy**: 0.97
- **Macro Avg**: 0.97 precision, 0.97 recall, 0.97 f1-score
- **Weighted Avg**: 0.97 precision, 0.97 recall, 0.97 f1-score

