from __future__ import annotations

import pandas as pd
from datetime import datetime
from catboost import Pool


# === Класс для извлечения временных фичей ===
class TimeFeatureExtractor:
    """Класс для добавления временных фичей к DataFrame."""
    
    def __init__(self, df: pd.DataFrame):
        """
        Инициализация extractor с копированием DataFrame.
        
        Параметры:
            df: исходный DataFrame (со столбцами 'timestamp' и 'country')
        """
        self.df = df.copy()
        self._prepared = False
        self._median_sec = 97 # Число получено в результате анализа исходных данных
    
    def _prepare_df(self, timestamp_col: str = 'timestamp'):
        """Приводит timestamp к datetime и извлекает дату (вызывается один раз)."""
        if self._prepared:
            return
        
        self.df[timestamp_col] = pd.to_datetime(self.df[timestamp_col])
        self._prepared = True
    
    def add_holiday_flags(self, df_holiday: pd.DataFrame, timestamp_col: str = 'timestamp') -> pd.DataFrame:
        """
        Добавляет столбцы is_holiday к сохранённому df.

        Параметры:
            df_holiday: DataFrame со столбцами 'date' и 'country' (праздники)

        Возвращает:
            DataFrame с добавленными столбцами 'is_holiday'
        """
        # Подготовка df (timestamp -> datetime, извлечение date)
        self._prepare_df()
        
        df_with_date = self.df.copy()
        df_with_date['date'] = pd.to_datetime(df_with_date[timestamp_col]).dt.date
        
        df_holiday = df_holiday.copy()
        df_holiday['date'] = pd.to_datetime(df_holiday['date']).dt.date

        # Определяем is_holiday = 1
        merged = df_with_date.merge(
            df_holiday[['country', 'date']],
            on=['country', 'date'],
            how='left',
            indicator=True
        )
        
        merged['is_holiday'] = (merged['_merge'] == 'both').astype(int)

        # Убираем служебные колонки
        merged = merged.drop(columns=['_merge', 'date'])

        self.df = merged
        
        return self.df
    
    def add_time_of_day(self, timestamp_col: str = 'timestamp') -> pd.DataFrame:
        """
        Добавим столбец с разбиением на время суток.

        Возвращает:
            DataFrame с добавленными столбцами 'time_of_day'
        """
        self._prepare_df()
            
        bins = [0, 6, 12, 18, 24]
        labels = ['night', 'morning', 'day', 'evening']

        self.df['time_of_day'] = pd.cut(self.df[timestamp_col].dt.hour, bins=bins, labels=labels, right=True, include_lowest=True)

        return self.df
    
    def add_time_features(self, timestamp_col: str = 'timestamp') -> pd.DataFrame:
        """
        Добавляет базовые временные фичи из datetime столбца.

        Параметры:
            date_col: имя datetime столбца (по умолчанию 'timestamp')

        Возвращает:
            DataFrame с добавленными временными фичами
        """
        self._prepare_df()
        
        self.df['is_weekend'] = self.df[timestamp_col].dt.dayofweek.isin([5, 6]).astype(int)
        
        return self.df
    
    def add_time_since_last_interaction(self, timestamp_col='timestamp', user_id_col='user_id'):
        """
        Добавляет время до предыдущего взаимодействия в секундах.

        Возвращает:
            DataFrame с добавленным столбцом 'time_since_last_interaction_sec'
        """
        
        self._prepare_df()
        
        # Сортируем по user_id и timestamp
        self.df = self.df.sort_values([user_id_col, timestamp_col])
        
        # Вычисляем разницу между текущим и предыдущим timestamp для каждого пользователя
        self.df['time_since_last_interaction_sec'] = self.df.groupby(user_id_col)[timestamp_col].diff().dt.total_seconds()

        # Заполняем NaN медианой в обоих
        self.df['time_since_last_interaction_sec'] = self.df['time_since_last_interaction_sec'].fillna(self._median_sec)
        
        return self.df
    
    def get_df(self) -> pd.DataFrame:
        """Возвращает текущий (возможно модифицированный) DataFrame."""
        
        return self.df
    
# === Загрузка основных ресурсов ===

# Загружаем признаки пользователей из БД
user_features = pd.read_parquet('../data/user_features.parquet')

# Загружаем признаки постов из БД
post_features = pd.read_parquet('../data/post_features.parquet')


# Загружаем df с датами праздников по странам из БД
holiday_df = pd.read_parquet('../data/df_holiday.parquet')

# === Вычисление фичей ===

def build_features(user_id: int, prediction_time: datetime) -> pd.DataFrame:
    """
    Возвращает df с фичами, которые затем подаются в модель для прогноза.
    """
    
    user_id = int(user_id)
    
    # Извлечение временных признаков из prediction_time
    df_final = pd.DataFrame({'timestamp': [prediction_time], 'user_id': [user_id]})
    df_final = df_final.merge(user_features[['user_id', 'country']], on='user_id', how='left')

    extractor = TimeFeatureExtractor(df_final)
    extractor.add_time_features()
    extractor.add_holiday_flags(holiday_df)
    extractor.add_time_of_day()
    extractor.add_time_since_last_interaction()

    df_final = extractor.get_df()
    
    df_final = df_final.drop('country', axis=1)
    
    # Объединение временных признаков, признаков пользователя и постов в общий датафрейм
    post_features_rec = post_features.copy()
    post_features_rec['user_id'] = user_id
    
    df_final = df_final.merge(user_features, on='user_id', how='left')
    df_final = df_final.merge(post_features_rec, on='user_id', how='left')
    
    # Получение предсказаний вероятности лайка от модели
    features_columns = [col for col in df_final.columns if col not in ['timestamp', 'user_id', 'post_id', 'target', 'action', 'text']]
    cat_features = ['country', 'city', 'exp_group', 'os', 'source', 'topic', 'time_of_day']
    features = Pool(data=df_final[features_columns], cat_features=cat_features)
    
    return df_final, features