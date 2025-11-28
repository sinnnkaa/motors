import pandas as pd
import numpy as np
from filework import load_and_prepare_data
from plots import plot_class_distribution
from model import train_model


if __name__ == "__main__":
    try:
        combined_df = load_and_prepare_data()
        plot_class_distribution(combined_df)
        models, scaler, le, histories = train_model(combined_df)

    except Exception as e:
        print(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()