#!/usr/bin/env python
"""
overlay_masks.py

Накладывает маски из папки labels на изображения из img и
сохраняет оверлеи в output-папку.

Ожидаемая структура (по умолчанию):

data/
  img/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/

Пример запуска:
python overlay_masks.py --data-root data --split val --out-dir overlays

В результате появится:
overlays/
  val/
    <имя>_overlay.png
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_files_by_stem(root: Path, exts):
    """Собираем файлы из папки и возвращаем dict {stem: Path}."""
    if not root.exists():
        return {}
    files = []
    for p in root.iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    return {p.stem: p for p in files}


def overlay_image_mask(
    img_path: Path,
    mask_path: Path,
    out_path: Path,
    alpha: float = 0.6,
):
    """
    Рисует оверлей маски поверх изображения и сохраняет в out_path.
    """
    # читаем картинку и маску
    image = np.array(Image.open(img_path).convert("RGB"))
    mask = np.array(Image.open(mask_path))

    # если маска вдруг RGB — берём один канал
    if mask.ndim == 3:
        mask = mask[:, :, 0]

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    ax.imshow(image)
    # cmap можно поменять при желании (tab20, jet, viridis и т.п.)
    ax.imshow(mask, cmap="tab20", alpha=alpha)
    ax.axis("off")
    ax.set_title(img_path.stem)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def process_split(
    data_root: Path,
    split: str,
    img_dirname: str,
    mask_dirname: str,
    out_root: Path,
    alpha: float,
):
    """
    Обрабатывает один split (train/val/test или любой другой).
    """
    img_dir = data_root / img_dirname / split
    mask_dir = data_root / mask_dirname / split

    print(f"\n=== Split: {split} ===")
    print(f"Папка img   : {img_dir}")
    print(f"Папка labels: {mask_dir}")

    img_map = list_files_by_stem(img_dir, IMAGE_EXTS)
    mask_map = list_files_by_stem(mask_dir, IMAGE_EXTS)

    common_stems = sorted(set(img_map.keys()) & set(mask_map.keys()))
    only_imgs = sorted(set(img_map.keys()) - set(mask_map.keys()))
    only_masks = sorted(set(mask_map.keys()) - set(img_map.keys()))

    print(f"Всего изображений: {len(img_map)}")
    print(f"Всего масок      : {len(mask_map)}")
    print(f"Пар (image+mask) : {len(common_stems)}")

    if only_imgs:
        print("Внимание: есть изображения без масок (первые 5):")
        for s in only_imgs[:5]:
            print("  ", img_map[s].name)
    if only_masks:
        print("Внимание: есть маски без изображений (первые 5):")
        for s in only_masks[:5]:
            print("  ", mask_map[s].name)

    out_dir = out_root / split
    out_dir.mkdir(parents=True, exist_ok=True)

    for stem in common_stems:
        img_path = img_map[stem]
        mask_path = mask_map[stem]
        out_path = out_dir / f"{stem}_overlay.png"

        overlay_image_mask(
            img_path=img_path,
            mask_path=mask_path,
            out_path=out_path,
            alpha=alpha,
        )

    print(f"Сохранено оверлеев: {len(common_stems)} в {out_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Наложение масок из labels на изображения из img"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data",
        help="Корневая папка датасета (по умолчанию ./data)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="Список сплитов для обработки (по умолчанию train val test)",
    )
    parser.add_argument(
        "--img-dirname",
        type=str,
        default="img",
        help="Подпапка с изображениями внутри data_root",
    )
    parser.add_argument(
        "--mask-dirname",
        type=str,
        default="labels",
        help="Подпапка с масками внутри data_root",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="overlays",
        help="Куда сохранять оверлеи",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.6,
        help="Прозрачность маски (0..1), 0 — маска невидима, 1 — полностью видна",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    data_root = Path(args.data_root)
    out_root = Path(args.out_dir)

    for split in args.splits:
        process_split(
            data_root=data_root,
            split=split,
            img_dirname=args.img_dirname,
            mask_dirname=args.mask_dirname,
            out_root=out_root,
            alpha=args.alpha,
        )

    print("\nГотово.")


if __name__ == "__main__":
    main()
