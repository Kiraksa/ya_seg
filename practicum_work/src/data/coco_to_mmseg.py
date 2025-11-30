#!/usr/bin/env python
"""
coco_to_mmseg.py

Конвертирует COCO-аннотацию (instances_*.json) с сегментацией
в формат MMSegmentation (индексные маски PNG + структура img/labels).:contentReference[oaicite:13]{index=13}

Пример использования:
python coco_to_mmseg.py \
  --coco-json coco_from_mmseg/instances_train.json \
  --images-dir path/to/images/train \
  --out-root data_mmseg_from_coco \
  --split train
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools.coco import COCO  # класс COCO с annToMask:contentReference[oaicite:14]{index=14}


def write_classes_txt(classes, out_path: Path):
    """Записывает список имён классов в txt-файл (по строке на класс)."""
    with out_path.open("w", encoding="utf-8") as f:
        for name in classes:
            f.write(str(name) + "\n")


def coco_to_mmseg(
    coco_json: Path,
    images_dir: Path,
    out_root: Path,
    split: str = "train",
    background_label: int = 0,
):
    """
    Основная функция конвертации:
    - читает COCO JSON,
    - создаёт маски (PNG) с индексами классов,
    - копирует изображения в out_root/img/split.
    """
    coco = COCO(str(coco_json))

    # Категории в COCO: id -> name
    cat_ids = sorted(coco.cats.keys())
    categories = [coco.cats[cid] for cid in cat_ids]

    # label_id 0 = background, 1..K = классы COCO
    label_names = ["background"] + [cat["name"] for cat in categories]
    catid2label = {cid: idx + 1 for idx, cid in enumerate(cat_ids)}

    out_root.mkdir(parents=True, exist_ok=True)
    classes_txt_path = out_root / "classes.txt"
    write_classes_txt(label_names, classes_txt_path)
    print(f"Записан classes.txt: {classes_txt_path}")

    out_img_dir = out_root / "img" / split
    out_lbl_dir = out_root / "labels" / split
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    img_ids = coco.getImgIds()
    print(f"Всего изображений в COCO: {len(img_ids)}")

    for img_id in img_ids:
        img_info = coco.loadImgs(img_id)[0]
        file_name = img_info["file_name"]
        width = int(img_info["width"])
        height = int(img_info["height"])

        # Итоговая маска: H x W, значения = ID класса (0..N)
        mask = np.full((height, width), fill_value=background_label, dtype=np.uint8)

        ann_ids = coco.getAnnIds(imgIds=[img_id], iscrowd=None)
        anns = coco.loadAnns(ann_ids)

        for ann in anns:
            # пропускаем аннотации без сегментации
            if "segmentation" not in ann or not ann["segmentation"]:
                continue

            cat_id = ann["category_id"]
            if cat_id not in catid2label:
                continue
            class_id = catid2label[cat_id]

            # annToMask корректно обрабатывает и полигоны, и RLE.:contentReference[oaicite:15]{index=15}
            ann_mask = coco.annToMask(ann)  # uint8 {0,1}, shape (H, W)
            if ann_mask.shape != mask.shape:
                # на всякий случай, если размеры не совпали
                print(f"WARNING: размер маски не совпадает с изображением для {file_name}")
                continue

            # При перекрытии объектов последний по порядку перезаписывает пиксели
            mask[ann_mask == 1] = class_id

        # сохраняем маску
        out_mask_path = out_lbl_dir / (Path(file_name).stem + ".png")
        Image.fromarray(mask).save(out_mask_path)

        # копируем изображение
        src_img_path = images_dir / file_name
        if not src_img_path.exists():
            # попробуем вариант с только basename (если в file_name путь)
            src_img_path = images_dir / Path(file_name).name

        if src_img_path.exists():
            dst_img_path = out_img_dir / Path(file_name).name
            shutil.copy2(src_img_path, dst_img_path)
        else:
            print(f"WARNING: не найден файл изображения {src_img_path}, пропускаем копирование.")

    print(f"Готово. Изображения: {out_img_dir}, маски: {out_lbl_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Конвертация COCO (segmentation) -> MMSegmentation (индексные маски PNG)"
    )
    parser.add_argument(
        "--coco-json",
        type=str,
        required=True,
        help="Путь к COCO annotations JSON (например, instances_train.json)",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        required=True,
        help="Папка с изображениями, на которые ссылается COCO (file_name).",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default="data_mmseg_from_coco",
        help="Корень выходного MMSeg-датасета.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Имя сплита для выходной структуры (train/val/test и т.п.).",
    )
    parser.add_argument(
        "--background-label",
        type=int,
        default=0,
        help="ID фона (background) для выходных масок (по умолчанию 0).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    coco_to_mmseg(
        coco_json=Path(args.coco_json),
        images_dir=Path(args.images_dir),
        out_root=Path(args.out_root),
        split=args.split,
        background_label=args.background_label,
    )


if __name__ == "__main__":
    main()
