#!/usr/bin/env python
"""
mmseg_to_coco.py

Конвертирует датасет сегментации в формате MMSegmentation
(индексные маски в PNG) в COCO JSON с RLE-масками, пригодный
для импорта в CVAT как "COCO 1.0" (Masks).:contentReference[oaicite:3]{index=3}

Пример структуры входных данных:
data/
  img/
    train/, val/, test/
  labels/
    train/, val/, test/

Пример вызова:
python mmseg_to_coco.py --data-root data --split train \
  --classes classes.txt --out-dir coco_from_mmseg
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils  # RLE encode/area/bbox:contentReference[oaicite:4]{index=4}


IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
MASK_EXTS = IMAGE_EXTS


def list_files_by_stem(root: Path, exts) -> dict[str, Path]:
    """Собираем файлы в папке и возвращаем dict {stem: Path}."""
    if not root.exists():
        return {}
    files = []
    for p in root.iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    return {p.stem: p for p in files}


def load_classes(path: Path) -> list[str]:
    """
    Считывает имена классов из txt-файла:
    строка №i = имя класса с label_id = i (как в маске).
    Пустые строки и строки, начинающиеся с #, игнорируются.
    """
    classes = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                classes.append('')
            else:
                classes.append(line)
    return classes


def build_categories(class_names, ignore_label, background_label):
    """
    Строит:
      - mapping label_id -> category_id (1..K)
      - список categories для COCO.
    Background и ignore-label можно исключить.:contentReference[oaicite:5]{index=5}
    """
    label2cat = {}
    categories = []
    cat_id = 1

    for label_id, name in enumerate(class_names):
        if ignore_label is not None and label_id == ignore_label:
            continue
        if background_label is not None and label_id == background_label:
            continue
        if not name:
            continue

        label2cat[label_id] = cat_id
        categories.append(
            {
                "id": cat_id,
                "name": name,
                "supercategory": "object",
            }
        )
        cat_id += 1

    return label2cat, categories


def convert_split(
    data_root: Path,
    split: str,
    class_names: list[str],
    label2cat: dict[int, int],
    ignore_label: int | None,
    background_label: int | None,
):
    """
    Конвертация одного split'а (train/val/test) в COCO-структуру.
    Возвращает (images, annotations).
    """
    img_dir = data_root / "img" / split
    mask_dir = data_root / "labels" / split

    img_map = list_files_by_stem(img_dir, IMAGE_EXTS)
    mask_map = list_files_by_stem(mask_dir, MASK_EXTS)

    common_stems = sorted(set(img_map.keys()) & set(mask_map.keys()))
    print(f"[{split}] images: {len(img_map)}, masks: {len(mask_map)}, pairs: {len(common_stems)}")

    images = []
    annotations = []
    ann_id = 1
    img_id = 1

    for stem in common_stems:
        img_path = img_map[stem]
        mask_path = mask_map[stem]

        # читаем размер картинки
        with Image.open(img_path) as im:
            width, height = im.size

        images.append(
            {
                "id": img_id,
                "file_name": img_path.name,  # CVAT будет матчить по имени файла:contentReference[oaicite:6]{index=6}
                "width": width,
                "height": height,
            }
        )

        # читаем маску
        mask = np.array(Image.open(mask_path))

        # если вдруг RGB-маска – берём один канал (как в VOC/Segmentation Mask).:contentReference[oaicite:7]{index=7}
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        uniq_labels = np.unique(mask)

        for lab in uniq_labels:
            lab = int(lab)

            if ignore_label is not None and lab == ignore_label:
                continue
            if background_label is not None and lab == background_label:
                continue
            if lab not in label2cat:
                continue

            binary = (mask == lab).astype(np.uint8)
            if binary.sum() == 0:
                continue

            # COCO RLE (crowd/stuff-аннотация).:contentReference[oaicite:8]{index=8}
            rle = mask_utils.encode(np.asfortranarray(binary))
            # pycocotools возвращает bytes, а в JSON нужны str
            rle["counts"] = rle["counts"].decode("utf-8")

            area = int(mask_utils.area(rle))
            bbox = mask_utils.toBbox(rle).tolist()  # [x, y, w, h]

            ann = {
                "id": ann_id,
                "image_id": img_id,
                "category_id": label2cat[lab],
                "segmentation": rle,
                "area": area,
                "bbox": bbox,
                "iscrowd": 1,  # RLE-маска (stuff / crowd):contentReference[oaicite:9]{index=9}
            }
            annotations.append(ann)
            ann_id += 1

        img_id += 1

    return images, annotations


def parse_args():
    parser = argparse.ArgumentParser(
        description="Конвертация MMSegmentation (индексные маски) -> COCO (RLE masks) для CVAT"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data",
        help="Корневой каталог с img/ и labels/ (по умолчанию ./data)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Список сплитов для конвертации",
    )
    parser.add_argument(
        "--classes",
        type=str,
        required=True,
        help="TXT-файл с именами классов по label id (строка = ID)",
    )
    parser.add_argument(
        "--ignore-label",
        type=int,
        default=255,
        help="ID ignore-класса в масках (например 255). "
             "Если не нужно игнорировать, укажи --ignore-label -1",
    )
    parser.add_argument(
        "--background-label",
        type=int,
        default=0,
        help="ID фона (background), который НЕ попадёт в COCO categories. "
             "Если хочется включить фон как отдельный класс, укажи --background-label -1",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="coco_from_mmseg",
        help="Куда сохранять instances_*.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    classes_path = Path(args.classes)
    class_names = load_classes(classes_path)

    ignore_label = None if args.ignore_label < 0 else args.ignore_label
    background_label = None if args.background_label < 0 else args.background_label

    label2cat, categories = build_categories(class_names, ignore_label, background_label)

    print("Классы (label_id -> category_id):")
    for lab_id, cat_id in sorted(label2cat.items()):
        print(f"  label {lab_id} -> category {cat_id} ({class_names[lab_id]!r})")

    for split in args.splits:
        images, annotations = convert_split(
            data_root=data_root,
            split=split,
            class_names=class_names,
            label2cat=label2cat,
            ignore_label=ignore_label,
            background_label=background_label,
        )

        coco_dict = {
            "images": images,
            "annotations": annotations,
            "categories": categories,
            # CVAT не требует info/licenses, но можно оставить заглушки.:contentReference[oaicite:10]{index=10}
            "info": {
                "description": f"MMSegmentation -> COCO ({split})",
                "version": "1.0",
            },
            "licenses": [],
        }

        out_json = out_dir / f"instances_{split}.json"
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(coco_dict, f, ensure_ascii=False, indent=2)

        print(f"[{split}] COCO JSON сохранён в {out_json}")


if __name__ == "__main__":
    main()
