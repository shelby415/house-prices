"""Небольшая полносвязная сеть PyTorch для регрессии."""
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from common import make_cv, make_preprocessor, make_submission


def train_network(X, y, config, seed):
    settings = config['dl']
    torch.manual_seed(seed)
    features = torch.tensor(np.asarray(X), dtype=torch.float32)
    targets = torch.tensor(np.asarray(y), dtype=torch.float32)
    loader = DataLoader(TensorDataset(features, targets), batch_size=settings['batch_size'], shuffle=True)
    layers, width = [], features.shape[1]
    for size in settings['hidden_sizes']:
        layers.extend([nn.Linear(width, size), nn.ReLU()])
        width = size
    layers.append(nn.Linear(width, 1))
    model = nn.Sequential(*layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings['learning_rate'])
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(settings['epochs']):
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(X_batch).squeeze(1), y_batch)
            loss.backward()
            optimizer.step()
    return model


def predict(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(np.asarray(X), dtype=torch.float32)).squeeze(1).numpy()


def run_dl(config, X, y, X_test, ids, log):
    torch.set_num_threads(config['dl']['num_threads'])
    scores, folds = [], []
    for fold, (train_idx, valid_idx) in enumerate(make_cv(config).split(X), 1):
        processor = make_preprocessor()
        X_train = processor.fit_transform(X.iloc[train_idx])
        X_valid = processor.transform(X.iloc[valid_idx])
        model = train_network(X_train, y.iloc[train_idx], config, config['seed'] + fold)
        error = predict(model, X_valid) - y.iloc[valid_idx].to_numpy()
        score = float(np.sqrt(np.mean(error ** 2)))
        scores.append(score)
        folds.append({'task': 'dl', 'model': 'mlp', 'fold': fold, 'rmse': score})
        log.info('mlp, фолд %s: RMSE = %.5f', fold, score)
    summary = {'task': 'dl', 'model': 'mlp', 'rmse_mean': float(np.mean(scores)), 'rmse_std': float(np.std(scores))}
    processor = make_preprocessor()
    X_train = processor.fit_transform(X)
    X_test_ready = processor.transform(X_test)
    model = train_network(X_train, y, config, config['seed'])
    submission = make_submission(ids, predict(model, X_test_ready))
    log.info('mlp: средний RMSE = %.5f', summary['rmse_mean'])
    return [summary], folds, {'dl_mlp': submission}
