#!/usr/bin/env python
"""
infer_unet_fcn.py

Инференс модели MMSegmentation (UNet FCN + др. конфиги) + визуализация.

Функции:
- прогон изображений (одного файла или директории);
- сохранение:
    * визуализаций (оверлей предсказания на исходном изображении);
    * "сырых" масок предсказаний (индексные PNG);
- (опционально) расчёт per-sample mDice при наличии GT масок:
    * сохранение CSV со всеми mDice;
    * сохранение top-K лучших и худших предсказаний (по mDice).

Зависимости:
    pip install mmsegmentation mmengine mmcv matplotlib pillow numpy
"""

import argparse
import os
import csv
import shutil
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import mmcv
from PIL import Image

from mmseg.apis import init_model, inference_model, show_result_pyplot
from mmseg.utils import register_all_modules


IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
MASK_EXTS = IMAGE_EXTS


def list_files_by_stem(root: Path, exts: set[str]) -> Dict[str, Path]:
    mapping = {}
    if not root.exists():
        return mapping
    for p in root.iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            mapping[p.stem] = p
    return mapping


def collect_images(input_path: str) -> List[str]:
    """Собираем список изображений: либо одно, либо все из директории."""
    if os.path.isdir(input_path):
        images = [
            str(p) for p in Path(input_path).iterdir()
            if p.suffix.lower() in IMAGE_EXTS
        ]
        images.sort()
        return images
    else:
        return [input_path]


def compute_mdice_per_sample(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    num_classes: int,
    ignore_index: int = 255,
) -> float:
    """
    Считает mDice для одного изображения по той же логике, что и IoUMetric
    в MMSegmentation (2 * intersect / (pred + label), затем nanmean по классам).:contentReference[oaicite:10]{index=10}
    """
    assert pred_mask.shape == gt_mask.shape, 'pred и gt разных размеров'

    mask_valid = (gt_mask != ignore_index)
    pred = pred_mask[mask_valid].astype(np.int64)
    label = gt_mask[mask_valid].astype(np.int64)

    # пересекающиеся пиксели (TP по каждому классу)
    intersect = pred[pred == label]

    area_intersect = np.bincount(intersect, minlength=num_classes)
    area_pred = np.bincount(pred, minlength=num_classes)
    area_label = np.bincount(label, minlength=num_classes)

    denom = area_pred + area_label
    dice_per_class = np.zeros(num_classes, dtype=np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        dice_per_class = 2.0 * area_intersect / denom
    # классы, где нет ни GT, ни предсказаний -> denom=0 -> считаем их NaN
    zero_union = (denom == 0)
    dice_per_class[zero_union] = np.nan

    # mDice как nanmean по классам
    mdice = float(np.nanmean(dice_per_class))
    return mdice


def infer_and_visualize(
    config_path: str,
    checkpoint_path: str,
    input_path: str,
    out_dir: str = 'outputs/unet_fcn_infer',
    device: str = 'cuda:0',
    opacity: float = 0.6,
    gt_root: str | None = None,
    num_classes: int | None = None,
    ignore_index: int = 255,
    top_k: int = 10,
):
    os.makedirs(out_dir, exist_ok=True)

    # Регистрируем все модули MMSeg (модели, датасеты, визуализаторы и т.п.).:contentReference[oaicite:11]{index=11}
    register_all_modules()

    print('Инициализация модели...')
    model = init_model(config_path, checkpoint_path, device=device)

    # Кол-во классов: если не задано, пробуем взять из мета-информации датасета
    if num_classes is None:
        if hasattr(model, 'dataset_meta') and 'classes' in model.dataset_meta:
            num_classes = len(model.dataset_meta['classes'])
            print(f'num_classes взят из model.dataset_meta: {num_classes}')
        else:
            # fallback: определим позже по макс. значению масок
            num_classes = None

    images = collect_images(input_path)
    print(f'Найдено изображений: {len(images)}')

    # Для mDice по сэмплам
    mdice_records: List[Dict[str, Any]] = []
    gt_map: Dict[str, Path] = {}
    if gt_root is not None:
        gt_map = list_files_by_stem(Path(gt_root), MASK_EXTS)
        if not gt_map:
            print(f'WARNING: в папке GT масок {gt_root} ничего не найдено.')

    for img_path in images:
        img_path = str(img_path)
        print(f'Обработка: {img_path}')
        img = mmcv.imread(img_path)

        result = inference_model(model, img)

        stem = Path(img_path).stem
        vis_out = os.path.join(out_dir, f'{stem}_vis.png')
        mask_out = os.path.join(out_dir, f'{stem}_pred_mask.png')

        # 1) Оверлей (предсказание поверх исходного изображения)
        visualization = show_result_pyplot(
            model,
            img,
            result,
            show=False,
            out_file=vis_out,
            opacity=opacity,
        )

        # 2) "сырая" маска классов (H × W, значения 0..num_classes-1)
        pred_mask = result.pred_sem_seg.data[0].cpu().numpy().astype(np.uint8)
        Image.fromarray(pred_mask).save(mask_out)

        # 3) mDice по этому сэмплу, если есть GT
        if gt_map:
            gt_path = gt_map.get(stem, None)
            if gt_path is None:
                print(f'  [mDice] нет GT маски для {stem}, пропускаем.')
            else:
                gt_mask = np.array(Image.open(gt_path))
                if gt_mask.ndim == 3:
                    gt_mask = gt_mask[:, :, 0]

                if gt_mask.shape != pred_mask.shape:
                    print(
                        f'  [mDice] WARNING: размеры pred ({pred_mask.shape}) '
                        f'и gt ({gt_mask.shape}) не совпадают для {stem}, пропуск.'
                    )
                else:
                    # если num_classes до этого не был определён
                    local_num_classes = num_classes
                    if local_num_classes is None:
                        local_num_classes = int(max(pred_mask.max(), gt_mask.max()) + 1)

                    mdice = compute_mdice_per_sample(
                        pred_mask=pred_mask,
                        gt_mask=gt_mask,
                        num_classes=local_num_classes,
                        ignore_index=ignore_index,
                    )
                    print(f'  [mDice] {mdice:.4f}')
                    mdice_records.append(
                        dict(
                            stem=stem,
                            image_path=img_path,
                            vis_path=vis_out,
                            pred_mask_path=mask_out,
                            gt_mask_path=str(gt_path),
                            mdice=mdice,
                        )
                    )

    # Если считали mDice по сэмплам — сохраним CSV и top-K
    if mdice_records:
        # CSV со всеми сэмплами
        csv_path = os.path.join(out_dir, 'mdice_per_sample.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    'stem',
                    'mdice',
                    'image_path',
                    'vis_path',
                    'pred_mask_path',
                    'gt_mask_path',
                ],
            )
            writer.writeheader()
            for rec in mdice_records:
                writer.writerow(rec)
        print(f'Сохранён CSV с mDice по сэмплам: {csv_path}')

        # top-K лучших и худших
        mdice_sorted = sorted(mdice_records, key=lambda r: r['mdice'])
        worst = mdice_sorted[:top_k]
        best = mdice_sorted[-top_k:]
        best = list(reversed(best))  # от лучшего к худшему

        best_dir = os.path.join(out_dir, f'top{top_k}_best_mdice')
        worst_dir = os.path.join(out_dir, f'top{top_k}_worst_mdice')
        os.makedirs(best_dir, exist_ok=True)
        os.makedirs(worst_dir, exist_ok=True)

        def _copy_vis(records, target_dir):
            for rec in records:
                src = rec['vis_path']
                if os.path.exists(src):
                    dst = os.path.join(
                        target_dir,
                        f"{Path(src).stem}_mdice_{rec['mdice']:.4f}.png"
                    )
                    shutil.copy2(src, dst)

        _copy_vis(best, best_dir)
        _copy_vis(worst, worst_dir)

        print(
            f'Сохранены визуализации top-{top_k} лучших в {best_dir} '
            f'и top-{top_k} худших в {worst_dir}.'
        )

    print(f'Готово. Результаты инференса в: {out_dir}')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Inference & visualization for MMSeg model + per-sample mDice ranking'
    )
    parser.add_argument(
        'config',
        help='Путь к config-файлу, например configs/practicum/unet_fcn-256.py'
    )
    parser.add_argument(
        'checkpoint',
        help='Путь к обученному чекпоинту .pth'
    )
    parser.add_argument(
        'input',
        help='Путь к изображению или директории с изображениями'
    )
    parser.add_argument(
        '--out-dir',
        default='outputs/unet_fcn_infer',
        help='Куда сохранять визуализации и маски'
    )
    parser.add_argument(
        '--device',
        default='cuda:0',
        help='Устройство для инференса, "cuda:0" или "cpu"'
    )
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.6,
        help='Прозрачность сегментации в оверлее [0,1]'
    )
    parser.add_argument(
        '--gt-root',
        type=str,
        default=None,
        help='Папка с GT масками (по basename совпадает с изображениями). '
             'Если указана, считается per-sample mDice и сохраняются top-K.'
    )
    parser.add_argument(
        '--num-classes',
        type=int,
        default=None,
        help='Число классов. Если не указано, берётся из model.dataset_meta["classes"], '
             'а при отсутствии — вычисляется по макс. значению на масках.'
    )
    parser.add_argument(
        '--ignore-index',
        type=int,
        default=255,
        help='Индекс ignore в GT масках (по умолчанию 255, как в IoUMetric).'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=10,
        help='Сколько лучших/худших предсказаний сохранять отдельно.'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    infer_and_visualize(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        input_path=args.input,
        out_dir=args.out_dir,
        device=args.device,
        opacity=args.opacity,
        gt_root=args.gt_root,
        num_classes=args.num_classes,
        ignore_index=args.ignore_index,
        top_k=args.top_k,
    )


if __name__ == '__main__':
    main()
