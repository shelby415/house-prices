"""Классические модели: проверка на фолдах и прогноз test."""
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from common import make_cv, make_preprocessor, make_submission


def build_model(name, config):
    params = dict(config['ml']['models'][name])
    if name == 'catboost':
        from catboost import CatBoostRegressor
        return CatBoostRegressor(random_seed=config['seed'], **params)
    constructors = {
        'ridge': Ridge, 'lasso': Lasso, 'elasticnet': ElasticNet, 'svr': SVR,
        'knn': KNeighborsRegressor, 'decision_tree': DecisionTreeRegressor,
        'random_forest': RandomForestRegressor, 'gradient_boosting': GradientBoostingRegressor,
    }
    if name in {'decision_tree', 'random_forest', 'gradient_boosting'}:
        params['random_state'] = config['seed']
    return Pipeline([('preprocessor', make_preprocessor()), ('regressor', constructors[name](**params))])


def cat_data(X):
    X = X.copy()
    columns = list(X.select_dtypes(exclude='number').columns)
    X[columns] = X[columns].fillna('Missing')
    return X, columns


def run_ml(config, X, y, X_test, ids, log):
    summaries, folds, submissions = [], [], {}
    for name in config['ml']['selected_models']:
        train_data, test_data, categories = X, X_test, None
        if name == 'catboost':
            train_data, categories = cat_data(X)
            test_data, _ = cat_data(X_test)
        scores = []
        for fold, (train_idx, valid_idx) in enumerate(make_cv(config).split(X), 1):
            model = build_model(name, config)
            fit_args = {'cat_features': categories} if categories is not None else {}
            model.fit(train_data.iloc[train_idx], y.iloc[train_idx], **fit_args)
            prediction = model.predict(train_data.iloc[valid_idx])
            score = float(np.sqrt(mean_squared_error(y.iloc[valid_idx], prediction)))
            scores.append(score)
            folds.append({'task': 'ml', 'model': name, 'fold': fold, 'rmse': score})
            log.info('%s, фолд %s: RMSE = %.5f', name, fold, score)
        summary = {'task': 'ml', 'model': name, 'rmse_mean': float(np.mean(scores)), 'rmse_std': float(np.std(scores))}
        summaries.append(summary)
        model = build_model(name, config)
        model.fit(train_data, y, **({'cat_features': categories} if categories is not None else {}))
        submissions[f'ml_{name}'] = make_submission(ids, model.predict(test_data))
        log.info('%s: средний RMSE = %.5f', name, summary['rmse_mean'])
    return summaries, folds, submissions
