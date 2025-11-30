#!/usr/bin/env python
"""
check_dataset_pairs.py

Проверяет:
- совпадает ли количество файлов изображений и масок в train/val/test;
- совпадают ли имена (basename) файлов;
- сохраняет несколько примеров визуализации (картинка, маска, оверлей).

Структура датасета:
data/
  img/
    train/, val/, test/
  labels/
    train/, val/, test/
"""

import argparse
from pathlib import Path
import random

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
MASK_EXTS = IMAGE_EXTS


def list_files_by_stem(root: Path, exts: set[str]) -> dict[str, Path]:
    """Вернуть словарь {stem: Path} для файлов с нужными расширениями."""
    files = []
    if not root.exists():
        return {}
    for p in root.iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    return {p.stem: p for p in files}


def visualize_pairs(pairs, out_dir: Path, num_examples: int = 5, alpha: float = 0.5):
    """
    Сохранить несколько примеров (image, mask, overlay) в out_dir.
    pairs — список кортежей (img_path, mask_path).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if not pairs:
        print('Нет пар для визуализации.')
        return

    random.seed(0)
    examples = random.sample(pairs, min(num_examples, len(pairs)))

    for img_path, mask_path in examples:
        image = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(mask_path))

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        axes[0].imshow(image)
        axes[0].set_title('Image')
        axes[0].axis('off')

        axes[1].imshow(mask, cmap='tab20')
        axes[1].set_title('Mask')
        axes[1].axis('off')

        axes[2].imshow(image)
        axes[2].imshow(mask, cmap='tab20', alpha=alpha)
        axes[2].set_title('Overlay')
        axes[2].axis('off')

        fig.suptitle(img_path.stem, fontsize=12)
        fig.tight_layout()
        out_file = out_dir / f'{img_path.stem}_preview.png'
        fig.savefig(out_file, dpi=150)
        plt.close(fig)

    print(f'Сохранены визуализации {len(examples)} пар в {out_dir}')


def check_split(
    data_root: Path,
    split: str,
    img_dirname: str = 'img',
    mask_dirname: str = 'labels',
    viz_root: Path | None = None,
):
    """Проверка одного сплита (train/val/test)."""
    print(f'\n=== Split: {split} ===')

    img_dir = data_root / img_dirname / split
    mask_dir = data_root / mask_dirname / split

    print(f'Папка изображений: {img_dir}')
    print(f'Папка масок     : {mask_dir}')

    img_map = list_files_by_stem(img_dir, IMAGE_EXTS)
    mask_map = list_files_by_stem(mask_dir, MASK_EXTS)

    print(f'Файлов изображений: {len(img_map)}')
    print(f'Файлов масок      : {len(mask_map)}')

    img_stems = set(img_map.keys())
    mask_stems = set(mask_map.keys())

    common = sorted(img_stems & mask_stems)
    only_imgs = sorted(img_stems - mask_stems)
    only_masks = sorted(mask_stems - img_stems)

    print(f'Пар (image+mask) : {len(common)}')
    print(f'Только изображений (без маски): {len(only_imgs)}')
    print(f'Только масок (без изображения): {len(only_masks)}')

    if only_imgs:
        print('  Примеры лишних изображений:')
        for s in only_imgs[:10]:
            print('   -', img_map[s].name)
    if only_masks:
        print('  Примеры лишних масок:')
        for s in only_masks[:10]:
            print('   -', mask_map[s].name)

    # Соберём список пар для дальнейшего анализа
    pairs = [(img_map[s], mask_map[s]) for s in common]

    # Визуализация
    if viz_root is not None and pairs:
        split_viz_dir = viz_root / split
        visualize_pairs(pairs, split_viz_dir, num_examples=5, alpha=0.6)

    return pairs


def parse_args():
    parser = argparse.ArgumentParser(
        description='Проверка пар (изображение-маска) в сегментационном датасете'
    )
    parser.add_argument(
        '--data-root',
        type=str,
        default='data/practicum_dataset',
        help='Корневая папка датасета (по умолчанию ./data)',
    )
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['train', 'val', 'test'],
        help='Список сплитов для проверки',
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
        '--viz-dir',
        type=str,
        default='practicum_viz',
        help='Куда складывать визуализации примеров',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    viz_root = Path(args.viz_dir)

    all_pairs = {}
    for split in args.splits:
        pairs = check_split(
            data_root=data_root,
            split=split,
            img_dirname=args.img_dirname,
            mask_dirname=args.mask_dirname,
            viz_root=viz_root,
        )
        all_pairs[split] = pairs

    # Можно дальше использовать all_pairs для дополнительных проверок
    print('\nПроверка завершена.')


if __name__ == '__main__':
    main()
