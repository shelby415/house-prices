"""Общие данные, обработка и сохранение результатов."""
import json
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_data(config, base_dir):
    settings = config['data']
    train = pd.read_csv(base_dir / settings['train_path'])
    test = pd.read_csv(base_dir / settings['test_path'])
    target, identifier = settings['target'], settings['id_column']
    if target not in train or identifier not in train or identifier not in test:
        raise ValueError('Проверьте наличие SalePrice и Id в исходных файлах.')
    y = train[target]
    if y.isna().any() or not np.isfinite(y).all() or (y <= 0).any():
        raise ValueError('Цена должна быть положительным числом без пропусков.')
    X = train.drop(columns=[target, identifier])
    X_test = test.drop(columns=[identifier])
    if set(X.columns) != set(X_test.columns):
        raise ValueError('Столбцы признаков train и test не совпадают.')
    return X, np.log(y), X_test[X.columns], test[identifier]


def make_cv(config):
    return KFold(n_splits=config['validation']['n_splits'], shuffle=True, random_state=config['seed'])


def make_preprocessor():
    numeric = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    categorical = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])
    return ColumnTransformer([
        ('numbers', numeric, make_column_selector(dtype_include=np.number)),
        ('categories', categorical, make_column_selector(dtype_exclude=np.number)),
    ])


def run_directory(config, base_dir):
    path = base_dir / config['output_dir'] / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    path.mkdir(parents=True, exist_ok=False)
    (path / 'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    return path


def save_results(path, summaries, folds, submissions):
    results = pd.DataFrame(summaries).sort_values('rmse_mean')
    results.to_csv(path / 'results.csv', index=False)
    pd.DataFrame(folds).to_csv(path / 'fold_scores.csv', index=False)
    for name, frame in submissions.items():
        frame.to_csv(path / f'submission_{name}.csv', index=False)
    best = results.iloc[0]
    submissions[f"{best['task']}_{best['model']}"].to_csv(path / 'submission.csv', index=False)
    return results


def make_submission(ids, predictions):
    prices = np.exp(np.asarray(predictions).reshape(-1))
    if len(prices) != len(ids) or not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError('Не удалось получить положительные цены для всех домов.')
    return pd.DataFrame({'Id': ids.to_numpy(), 'SalePrice': prices})
