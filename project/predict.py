from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd 
import joblib

from features import build_features


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = "../data/model_catboost_full.pkl"

# Загружаем модель
def load_model():
    return joblib.load(MODEL_PATH)

# Загружаем посты из БД
post_text = pd.read_parquet('../data/post_text.parquet')

def predict(user_id: int, prediction_time: datetime, limit: int = 10):
    
    limit = int(limit)
    
    model = load_model()

    df_final, features = build_features(user_id=user_id, prediction_time=prediction_time)
    
    df_final['like_prob'] = model.predict_proba(features)[:, 1]
    
    df_final = df_final.merge(post_text, left_index=True, right_index=True)
    
    # Выбор top-N постов с наибольшей вероятностью
    top_df = df_final.nlargest(limit, 'like_prob')[['post_id', 'text', 'topic']]

    return top_df

def main():
    parser = argparse.ArgumentParser(
        description="Запуск прогноза модели"
    )

    parser.add_argument(
        "--user-id",
        required=True,
        help="ID пользователя",
    )

    parser.add_argument(
        "--prediction-time",
        required=True,
        help="Время в формате YYYY-MM-DD HH:MM:SS",
    )
    
    parser.add_argument(
        "--limit",
        required=True,
        help="Количество",
    )

    args = parser.parse_args()

    result = predict(
        user_id=args.user_id,
        prediction_time=args.prediction_time,
        limit=args.limit
    )

    print(f"Prediction: {result}")


if __name__ == "__main__":
    main()