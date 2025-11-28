import pandas as pd
import os
import glob
from categories import category_map

# ЗАГРУЗКА ДАННЫХ
# Файл для загрузки и предобработки данных 
PATH = "C:/Users/User/Downloads/archive (2)/files"  

def get_category_by_number(file_name):
    """
    Функция для создания словаря категорий 
    """
    file_number = file_name.split(' - ')[0]   
    return category_map.get(file_number, 'Unknown')

def load_and_prepare_data():
    """
    Функция для составления данных из файлов в датасет + разметка
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