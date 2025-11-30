# dataset settings

dataset_type = 'PracticumDataset'

img_scale=(256, 256)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', scale=img_scale, keep_ratio=False),
    dict(type='PhotoMetricDistortion'),
    dict(type='RandomRotFlip'),
    
    dict(type='PackSegInputs')
]

train_dataset=dict(
    type=dataset_type,
    data_root='/home/kiriy/code/ya_seg/mmsegmentation/data/practicum_dataset',
    data_prefix=dict(
        img_path='img/train',
        seg_map_path='labels/train'),
    pipeline=train_pipeline,
    img_suffix=".jpg",
    seg_map_suffix=".png"
)
 
train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_dataset
    
)


test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]
val_dataset = dataset=dict(
    type=dataset_type,
    data_root='/home/kiriy/code/ya_seg/mmsegmentation/data/practicum_dataset',
    data_prefix=dict(
        img_path='img/val',
        seg_map_path='labels/val'),
    pipeline=test_pipeline,
    img_suffix=".jpg",
    seg_map_suffix=".png"
)
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=val_dataset
)

test_dataset = dataset=dict(
    type=dataset_type,
    data_root='/home/kiriy/code/ya_seg/mmsegmentation/data/practicum_dataset',
    data_prefix=dict(
        img_path='img/test',
        seg_map_path='labels/test'),
    pipeline=test_pipeline,
    img_suffix=".jpg",
    seg_map_suffix=".png"
)
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=test_dataset
)


# Здесь же в пайплайне данных создаются объекты для подсчёта метрик
# IoUMetric это общий класс для всех метрик, которые работают на уровне регионов 
# конкретные метрики указываются в виде аргумента iou_metrics
# в этом случае мы будет считать только mDice
val_evaluator = dict(type='IoUMetric', iou_metrics=['mDice'])
test_evaluator = val_evaluator