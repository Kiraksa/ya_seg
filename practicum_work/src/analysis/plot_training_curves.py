#!/usr/bin/env python
"""
plot_training_curves.py

Читает mmseg/mmengine *.log.json и строит:
- график train loss;
- графики валидационных метрик (mDice, mIoU, mAcc, aAcc, ...);
- (опционально) график "ошибка = 100 - mDice".

Основано на идее tools/analyze_logs.py из MMSegmentation.:contentReference[oaicite:5]{index=5}

Зависимости:
    pip install matplotlib
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_log_json(log_path: Path):
    """Разбирает log.json в две части: train_logs и val_logs."""
    train_logs = []
    val_logs = []
    logs = []

    with log_path.open('r', encoding='utf-8') as f:
        for line in f:
            logs.append(json.loads(line))

            """mode = log.get('mode', None)
            if mode == 'train':
                train_logs.append(log)
            elif mode in ('val', 'test'):
                val_logs.append(log)"""
    train_logs = logs[:-1]
    val_logs = logs[-1:]

    return train_logs, val_logs


def plot_train_loss(train_logs, out_path: Path):
    """Строим loss по итерациям."""
    if not train_logs:
        print('В train-логах пусто, loss не построен.')
        return

    iters = []
    losses = []

    for i, log in enumerate(train_logs):
        # iter может отсутствовать, тогда используем индекс
        it = log.get('iter', i)
        loss = log.get('loss', None)
        if loss is None:
            continue
        iters.append(it)
        losses.append(loss)

    if not losses:
        print('Не найдено поле "loss" в train-логах.')
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(iters, losses, label='train loss')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.title('Training loss')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'Сохранён график train loss: {out_path}')


def plot_val_metrics(val_logs, out_path: Path, error_out_path: Path | None = None):
    """Строим метрики на валидации (mDice, mIoU, ...)."""
    if not val_logs:
        print('В val-логах пусто, метрики не построены.')
        return

    # Ищем все ключи mXXX / aAcc
    metric_keys = set()
    for log in val_logs:
        for k in log.keys():
            if k.startswith('m') or k == 'aAcc':
                if isinstance(log[k], (int, float)):
                    metric_keys.add(k)

    if not metric_keys:
        print('В val-логах не найдены метрики вида mXXX / aAcc.')
        return

    epochs = [log.get('epoch', idx + 1) for idx, log in enumerate(val_logs)]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1) Все метрики на одном графике
    plt.figure(figsize=(9, 5))
    for key in sorted(metric_keys):
        values = [log.get(key, None) for log in val_logs]
        if any(v is None for v in values):
            continue
        plt.plot(epochs, values, marker='o', label=key)

    plt.xlabel('epoch')
    plt.ylabel('metric (percent)')
    plt.title('Validation metrics')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'Сохранён график валидационных метрик: {out_path}')

    # 2) mDice-ошибка (100 - mDice), если mDice есть
    if error_out_path is not None and 'mDice' in metric_keys:
        mdice = [log.get('mDice', None) for log in val_logs]
        if all(v is not None for v in mdice):
            errors = [100.0 - v for v in mdice]
            plt.figure(figsize=(8, 5))
            plt.plot(epochs, errors, marker='o')
            plt.xlabel('epoch')
            plt.ylabel('error = 100 - mDice')
            plt.title('Validation error (from mDice)')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(error_out_path, dpi=150)
            plt.close()
            print(f'Сохранён график ошибки по mDice: {error_out_path}')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Строит графики обучения (loss + валидационные метрики) из mmseg log.json'
    )
    parser.add_argument(
        'log_json',
        type=str,
        help='Путь к *.log.json (из work_dir mmseg).',
    )
    parser.add_argument(
        '--out-dir',
        type=str,
        default=None,
        help='Куда сохранить PNG-графики (по умолчанию рядом с log.json).',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    log_path = Path(args.log_json)
    out_dir = Path(args.out_dir) if args.out_dir else log_path.parent

    train_logs, val_logs = parse_log_json(log_path)

    plot_train_loss(
        train_logs,
        out_path=out_dir / 'train_loss.png',
    )
    plot_val_metrics(
        val_logs,
        out_path=out_dir / 'val_metrics.png',
        error_out_path=out_dir / 'val_mdice_error.png',
    )

    print('Готово.')


if __name__ == '__main__':
    main()
