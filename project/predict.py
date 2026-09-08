from __future__ import annotations
from loguru import logger
import os
from src.db.database import postgres_connection
import argparse
from pathlib import Path
import joblib
from typing import Any, Dict, List
from datetime import datetime
import pandas as pd 
import sys
from .features import build_features


BASE_DIR = Path(__file__).resolve().parent.parent  # Social-Network-Recommender-System
MODEL_PATH = BASE_DIR / "data" / "model_catboost_full.pkl"

# === Вспомогательные функции ===
def load_model(model_path: str = MODEL_PATH):
    """
    Загружает ML-модель из файла.
    Исключения:
        FileNotFoundError — если файл модели не найден.
        RuntimeError — если произошла ошибка при загрузке модели.
    """

    logger.info(f"Загрузка модели из файла {model_path}...")

    try:
        with open(model_path, "rb") as file:
            model = joblib.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"❌ Файл модели не найден: {model_path}")
    except Exception as e:
        raise RuntimeError(f"❌ Ошибка при загрузке модели: {e}") from e

    logger.success("Модель успешно загружена")

    return model

def load_sql(query: str, dtypes: Dict[str, Any] = None) -> pd.DataFrame:
    """
    Выполняет SQL-запрос через соединение postgres_connection и возвращает DataFrame.

    Аргументы:
        query: SQL-запрос
        dtypes: словарь типов колонок для pd.read_sql (по умолчанию None)

    Возвращает:
        pd.DataFrame с результатом запроса

    Исключения:
        RuntimeError, если произошла ошибка при выполнении запроса
    """
    conn = postgres_connection()

    try:
        df = pd.read_sql(query, conn, dtype=dtypes)
    except Exception as e:
        raise RuntimeError(
            f"❌ Ошибка при выполнении SQL-запроса: {e}\nЗапрос: {query}"
        ) from e
    finally:
        conn.close()

    return df

# === Загрузка основных ресурсов ===

logger.info("Инициализация сервиса...")

# Загружаем признаки постов из БД
QUERY_POSTS = """
SELECT *
FROM post_text_df
"""
posts_df = load_sql(QUERY_POSTS) 

# Загружаем timestamp последнего взаимодействия юзера из БД
QUERY_FEEDS = """
SELECT user_id, MAX(timestamp) AS timestamp
FROM feed_data
WHERE ACTION = 'view'
GROUP BY user_id
"""
last_timestamp = load_sql(QUERY_FEEDS) 

# Загружаем количество лайков на каждом посте из БД
QUERY_LIKE_COUNT = """
SELECT post_id, COUNT(*) AS like_count
FROM feed_data
WHERE ACTION = 'like'
GROUP BY post_id
ORDER BY like_count DESC
"""
like_count = load_sql(QUERY_LIKE_COUNT) 
    
# Загружаем модель в память
model = load_model()

# Информация об успешной инициализации сервиса
logger.success("Сервис успешно инициализирован")

# === Получение рекомендаций ===
def predict(user_id: int, prediction_time: datetime, limit: int = 10):
    
    limit = int(limit)
    
    if user_id in last_timestamp['user_id'].values:
        df_final, features = build_features(user_id=user_id, prediction_time=prediction_time)
        
        df_final['like_prob'] = model.predict_proba(features)[:, 1]
        
        df_final = df_final.merge(posts_df['text'], left_index=True, right_index=True)
        
        # Выбор top-N постов с наибольшей вероятностью
        top_df = df_final.nlargest(limit, 'like_prob')[['post_id', 'text', 'topic']]
        
    else:
        # Выбор top-N постов с наибольшим количеством лайков
        popular_posts = like_count.head(limit)[['post_id']]
        top_df = popular_posts.merge(posts_df[['post_id', 'text', 'topic']], on='post_id', how='left')

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