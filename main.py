"""Запуск обучения и сохранение результатов House Prices."""
import argparse
import json
import logging
import sys
from pathlib import Path

from common import load_data, run_directory, save_results


def main():
    parser = argparse.ArgumentParser(description='Обучение моделей House Prices')
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('config.json'))
    parser.add_argument('--task', choices=['ml', 'dl', 'all'], help='Какой пайплайн запустить')
    parser.add_argument('--model', help='Имена ML-моделей через запятую или all')
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if args.task:
        config['task'] = args.task
    if args.model:
        if config['task'] == 'dl':
            parser.error('--model используется для ML; выберите --task ml или all.')
        config['ml']['selected_models'] = list(config['ml']['models']) if args.model == 'all' else [x.strip() for x in args.model.split(',')]
    if config['task'] not in {'ml', 'dl', 'all'}:
        parser.error('В конфиге task должен быть ml, dl или all.')
    names = config['ml']['selected_models']
    if config['task'] in {'ml', 'all'} and (not names or any(name not in config['ml']['models'] for name in names)):
        parser.error('Проверьте названия ML-моделей в конфиге или --model.')
    base_dir = config_path.parent
    path = run_directory(config, base_dir)
    log = logging.getLogger('house_prices')
    log.setLevel(logging.INFO)
    log.handlers.clear()
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(path / 'run.log', encoding='utf-8')):
        handler.setFormatter(formatter)
        log.addHandler(handler)
    try:
        log.info('Запуск: task=%s; папка результатов: %s', config['task'], path)
        X, y, X_test, ids = load_data(config, base_dir)
        log.info('Данные: %s обучающих и %s тестовых домов', len(X), len(X_test))
        summaries, folds, submissions = [], [], {}
        if config['task'] in {'ml', 'all'}:
            from ml_pipeline import run_ml
            rows, fold_rows, files = run_ml(config, X, y, X_test, ids, log)
            summaries.extend(rows)
            folds.extend(fold_rows)
            submissions.update(files)
        if config['task'] in {'dl', 'all'}:
            from dl_pipeline import run_dl
            rows, fold_rows, files = run_dl(config, X, y, X_test, ids, log)
            summaries.extend(rows)
            folds.extend(fold_rows)
            submissions.update(files)
        results = save_results(path, summaries, folds, submissions)
        log.info('Готово. Лучшая модель по CV: %s / %s; RMSE %.5f', results.iloc[0]['task'], results.iloc[0]['model'], results.iloc[0]['rmse_mean'])
        log.info('Файл для Kaggle: %s', path / 'submission.csv')
    except Exception:
        log.exception('Запуск не завершился. Проверьте сообщение об ошибке выше.')
        raise


if __name__ == '__main__':
    main()
