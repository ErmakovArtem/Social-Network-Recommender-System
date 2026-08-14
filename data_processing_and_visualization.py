import pandas as pd
import matplotlib.pyplot as plt


def check_column_for_nan(df, column_name):
    """
    Комплексная проверка столбца на NaN.
    """

    if column_name not in df.columns:
        print(f"Столбца '{column_name}' нет в DataFrame")
        return

    mask = df[column_name].isna()
    nan_count = mask.sum()

    if nan_count > 0:
        total = len(df)
        percentage = nan_count / total * 100
        nan_indices = df.index[mask].tolist()

        print(f"Столбец '{column_name}':")
        print(f"  Количество NaN: {nan_count}")
        print(f"  Процент NaN: {percentage:.2f}%")
        print(f"  Индексы строк с NaN: {nan_indices}")
    else:
        print(f"Столбец '{column_name}' не содержит NaN")
        
        
def plot_histograms(df: pd.DataFrame, columns: list, bins: int = 20, figsize: tuple = (10, 5)):
    """
    Строит гистограммы для числовых столбцов.
    """
    valid_columns = [col for col in columns if col in df.columns]
    
    if not valid_columns:
        print("Нет допустимых столбцов в DataFrame.")
        return

    n_cols = len(valid_columns)
    fig, axes = plt.subplots(nrows=n_cols, ncols=1, figsize=figsize)

    if n_cols == 1:
        axes = [axes]

    for i, col in enumerate(valid_columns):
        ax = axes[i]
        df[col].hist(ax=ax, bins=bins, alpha=0.7, color='steelblue', edgecolor='black')
        ax.set_title(f'Гистограмма столбца "{col}"')
        ax.set_xlabel(col, fontsize=12)
        ax.set_ylabel("Частота", fontsize=12)

    plt.tight_layout(h_pad=2.0)
    plt.show()


def plot_bars(df: pd.DataFrame, columns: list, bins: int = 20, figsize: tuple = (10, 5)):
    """
    Строит bar plots для категориальных/объектных.
    """
    valid_columns = [col for col in columns if col in df.columns]
    if not valid_columns:
        print("Нет допустимых столбцов в DataFrame.")
        return

    n_cols = len(valid_columns)
    fig, axes = plt.subplots(nrows=n_cols, ncols=1, figsize=figsize)

    if n_cols == 1:
        axes = [axes]

    for i, col in enumerate(valid_columns):
        ax = axes[i]

        vc = df[col].value_counts()
        vc.plot(kind="bar", ax=ax, alpha=0.7, color="tab:orange", edgecolor="black")
        ax.set_title(f"Bar plot столбца '{col}' (категориальный)")
        ax.set_xlabel(col, fontsize=12)
        ax.set_ylabel("Количество", fontsize=12)
        ax.tick_params(axis="x", rotation=45)
        max_value = vc.max()
        ax.set_ylim(0, max_value * 1.15)
        # Добавляем подписи значений над столбиками
        ax.bar_label(
        ax.containers[0],
        fontsize=10,
        padding=0,
        label_type="edge"
        )

    plt.tight_layout(h_pad=2.0)
    plt.show()