#!/usr/bin/env python
"""
dataset_stats_segmentation.py

Анализ сегментационного датасета:
- баланс классов (по пикселям);
- размеры изображений;
- (опционально) размеры объектов по классам.

Датасет:
data/
  img/
    train/, val/, test/
  labels/
    train/, val/, test/
"""

import argparse
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
from PIL import Image

try:
    from skimage.measure import label as cc_label, regionprops
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
MASK_EXTS = IMAGE_EXTS


def list_files_by_stem(root: Path, exts: set[str]) -> dict[str, Path]:
    if not root.exists():
        return {}
    files = []
    for p in root.iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    return {p.stem: p for p in files}


def get_pairs_for_split(
    data_root: Path,
    split: str,
    img_dirname: str = 'img',
    mask_dirname: str = 'labels',
):
    img_dir = data_root / img_dirname / split
    mask_dir = data_root / mask_dirname / split

    print(f'Поиск изображений и масок в {img_dir} и {mask_dir}')
    img_map = list_files_by_stem(img_dir, IMAGE_EXTS)
    mask_map = list_files_by_stem(mask_dir, MASK_EXTS)

    common_stems = sorted(set(img_map.keys()) & set(mask_map.keys()))
    pairs = [(img_map[s], mask_map[s]) for s in common_stems]
    return pairs


def analyze_split(
    split: str,
    pairs,
    ignore_label: int | None = 255,
    background_label: int | None = 0,
    max_images: int | None = None,
    objects: bool = False,
):
    print(f'\n=== Статистика по split={split} ===')

    if max_images is not None and len(pairs) > max_images:
        pairs = pairs[:max_images]
        print(f'Ограничиваемся первыми {len(pairs)} изображениями для анализа.')

    if not pairs:
        print('Нет пар изображение-маска, анализ невозможен.')
        return

    class_counts = Counter()
    widths, heights = [], []
    obj_areas = defaultdict(list) if objects and HAS_SKIMAGE else None

    for img_path, mask_path in pairs:
        mask = np.array(Image.open(mask_path))

        # Если маска RGB, возьмём один канал (часто сегментационные маски хранятся в 1 канале). :contentReference[oaicite:2]{index=2}
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        h, w = mask.shape[:2]
        heights.append(h)
        widths.append(w)

        labels, counts = np.unique(mask, return_counts=True)
        for lab, cnt in zip(labels, counts):
            lab = int(lab)
            if ignore_label is not None and lab == ignore_label:
                continue
            class_counts[lab] += int(cnt)

        if objects and HAS_SKIMAGE:
            # для каждого класса (кроме background и ignore) считаем connected components
            for lab in labels:
                lab = int(lab)
                if ignore_label is not None and lab == ignore_label:
                    continue
                if background_label is not None and lab == background_label:
                    continue
                binary = (mask == lab)
                if not np.any(binary):
                    continue
                labeled = cc_label(binary)
                for region in regionprops(labeled):
                    obj_areas[lab].append(region.area)

    # Размеры изображений
    widths = np.array(widths)
    heights = np.array(heights)

    print(f'Всего изображений: {len(pairs)}')
    print(f'Ширина  (px): min={widths.min()}, max={widths.max()}, mean={widths.mean():.1f}')
    print(f'Высота  (px): min={heights.min()}, max={heights.max()}, mean={heights.mean():.1f}')

    # Частоты уникальных разрешений
    from collections import Counter as Cnt
    res_counter = Cnt(zip(widths, heights))
    print('Топ 5 самых частых разрешений (width x height: count):')
    for (w, h), c in res_counter.most_common(5):
        print(f'  {w} x {h}: {c}')

    # Баланс классов
    if not class_counts:
        print('Не найдено ни одного пикселя классов (кроме ignore).')
        return

    total_pixels = sum(class_counts.values())
    print(f'\nКлассов (исключая ignore={ignore_label}): {len(class_counts)}')
    print('ID класса | пикселей      | доля, %')
    print('----------+---------------+--------')
    for lab in sorted(class_counts.keys()):
        cnt = class_counts[lab]
        pct = 100.0 * cnt / total_pixels
        print(f'{lab:9d} | {cnt:13d} | {pct:6.2f}')

    # Статистика по объектам
    if objects:
        if not HAS_SKIMAGE:
            print('\nscikit-image не установлен, пропускаем статистику по объектам.')
        else:
            if not obj_areas:
                print('\nОбъекты не найдены (все маски пустые или только background).')
            else:
                print('\nСтатистика по размерам объектов (в пикселях):')
                for lab in sorted(obj_areas.keys()):
                    arr = np.array(obj_areas[lab])
                    print(f'Класс {lab}: '
                          f'кол-во объектов={len(arr)}, '
                          f'min={arr.min()}, '
                          f'median={np.median(arr):.1f}, '
                          f'mean={arr.mean():.1f}, '
                          f'p90={np.percentile(arr, 90):.1f}, '
                          f'max={arr.max()}')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Статистика сегментационного датасета (классы, размеры, объекты)'
    )
    parser.add_argument(
        '--data-root',
        type=str,
        default='data',
        help='Корень датасета (по умолчанию ./data)',
    )
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['train', 'val', 'test'],
        help='Список сплитов для анализа',
    )
    parser.add_argument(
        '--img-dirname',
        type=str,
        default='img',
        help='Подпапка с изображениями внутри data_root',
    )
    parser.add_argument(
        '--mask-dirname',
        type=str,
        default='labels',
        help='Подпапка с масками внутри data_root',
    )
    parser.add_argument(
        '--ignore-label',
        type=int,
        default=255,
        help='ID ignore-класса, который исключаем из статистики (например 255). '
             'Если не нужно игнорировать, передай --ignore-label -1',
    )
    parser.add_argument(
        '--background-label',
        type=int,
        default=0,
        help='ID фона (background), который можно исключать из объектной статистики.',
    )
    parser.add_argument(
        '--max-images',
        type=int,
        default=None,
        help='Максимальное количество изображений на сплит для анализа (для ускорения).',
    )
    parser.add_argument(
        '--objects',
        action='store_true',
        help='Считать статистику по объектам (connected components, требует scikit-image).',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root)

    for split in args.splits:
        pairs = get_pairs_for_split(
            data_root=data_root,
            split=split,
            img_dirname=args.img_dirname,
            mask_dirname=args.mask_dirname,
        )
        analyze_split(
            split=split,
            pairs=pairs,
            ignore_label=None if args.ignore_label < 0 else args.ignore_label,
            background_label=args.background_label,
            max_images=args.max_images,
            objects=args.objects,
        )

    print('\nАнализ завершён.')


if __name__ == '__main__':
    main()
