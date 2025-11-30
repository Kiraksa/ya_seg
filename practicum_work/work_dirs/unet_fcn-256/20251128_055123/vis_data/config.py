auxiliary_head = dict(
    loss_decode=[
        dict(
            loss_name='loss_ce',
            loss_weight=1.0,
            type='CrossEntropyLoss',
            use_sigmoid=False),
        dict(loss_name='loss_dice', loss_weight=2.0, type='DiceLoss'),
    ],
    num_classes=3)
data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    pad_val=0,
    seg_pad_val=255,
    size=(
        256,
        256,
    ),
    std=[
        58.395,
        57.12,
        57.375,
    ],
    type='SegDataPreProcessor')
dataset = dict(
    data_prefix=dict(img_path='img/test', seg_map_path='labels/test'),
    data_root='/home/kiriy/code/ya_seg/mmsegmentation/data/practicum_dataset',
    img_suffix='.jpg',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(type='PackSegInputs'),
    ],
    seg_map_suffix='.png',
    type='PracticumDataset')
dataset_type = 'PracticumDataset'
decode_head = dict(
    loss_decode=[
        dict(
            loss_name='loss_ce',
            loss_weight=1.0,
            type='CrossEntropyLoss',
            use_sigmoid=False),
        dict(loss_name='loss_dice', loss_weight=2.0, type='DiceLoss'),
    ],
    num_classes=3)
default_hooks = dict(
    checkpoint=dict(by_epoch=True, interval=10, type='CheckpointHook'),
    logger=dict(interval=1, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(draw=True, interval=10, type='SegVisualizationHook'))
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
img_scale = (
    256,
    256,
)
input_suze = (
    256,
    256,
)
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=False)
model = dict(
    auxiliary_head=dict(
        align_corners=False,
        channels=64,
        concat_input=False,
        dropout_ratio=0.1,
        in_channels=128,
        in_index=3,
        loss_decode=[
            dict(
                loss_name='loss_ce',
                loss_weight=1.0,
                type='CrossEntropyLoss',
                use_sigmoid=False),
            dict(loss_name='loss_dice', loss_weight=2.0, type='DiceLoss'),
        ],
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=3,
        num_convs=1,
        type='FCNHead'),
    backbone=dict(
        act_cfg=dict(type='ReLU'),
        base_channels=64,
        conv_cfg=None,
        dec_dilations=(
            1,
            1,
            1,
            1,
        ),
        dec_num_convs=(
            2,
            2,
            2,
            2,
        ),
        downsamples=(
            True,
            True,
            True,
            True,
        ),
        enc_dilations=(
            1,
            1,
            1,
            1,
            1,
        ),
        enc_num_convs=(
            2,
            2,
            2,
            2,
            2,
        ),
        in_channels=3,
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        norm_eval=False,
        num_stages=5,
        strides=(
            1,
            1,
            1,
            1,
            1,
        ),
        type='UNet',
        upsample_cfg=dict(type='InterpConv'),
        with_cp=False),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_val=0,
        seg_pad_val=255,
        size=(
            256,
            256,
        ),
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='SegDataPreProcessor'),
    decode_head=dict(
        align_corners=False,
        channels=64,
        concat_input=False,
        dropout_ratio=0.1,
        in_channels=64,
        in_index=4,
        loss_decode=[
            dict(
                loss_name='loss_ce',
                loss_weight=1.0,
                type='CrossEntropyLoss',
                use_sigmoid=False),
            dict(loss_name='loss_dice', loss_weight=2.0, type='DiceLoss'),
        ],
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=3,
        num_convs=1,
        type='FCNHead'),
    pretrained=None,
    test_cfg=dict(crop_size=256, mode='whole', stride=170),
    train_cfg=dict(),
    type='EncoderDecoder')
norm_cfg = dict(requires_grad=True, type='SyncBN')
optim_wrapper = dict(
    clip_grad=None,
    optimizer=dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0005),
    type='OptimWrapper')
optimizer = dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0005)
param_scheduler = [
    dict(
        begin=0,
        by_epoch=False,
        end=100,
        eta_min=0.0001,
        power=0.9,
        type='PolyLR'),
]
randomness = dict(seed=0)
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(img_path='img/test', seg_map_path='labels/test'),
        data_root=
        '/home/kiriy/code/ya_seg/mmsegmentation/data/practicum_dataset',
        img_suffix='.jpg',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        seg_map_suffix='.png',
        type='PracticumDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_dataset = dict(
    data_prefix=dict(img_path='img/test', seg_map_path='labels/test'),
    data_root='/home/kiriy/code/ya_seg/mmsegmentation/data/practicum_dataset',
    img_suffix='.jpg',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(type='PackSegInputs'),
    ],
    seg_map_suffix='.png',
    type='PracticumDataset')
test_evaluator = dict(
    iou_metrics=[
        'mDice',
    ], type='IoUMetric')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
train_cfg = dict(max_epochs=100, type='EpochBasedTrainLoop')
train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        data_prefix=dict(img_path='img/train', seg_map_path='labels/train'),
        data_root=
        '/home/kiriy/code/ya_seg/mmsegmentation/data/practicum_dataset',
        img_suffix='.jpg',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(img_scale=(
                256,
                256,
            ), keep_ratio=False, type='Resize'),
            dict(type='PhotoMetricDistortion'),
            dict(type='RandomRotFlip'),
            dict(type='PackSegInputs'),
        ],
        seg_map_suffix='.png',
        type='PracticumDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_dataset = dict(
    data_prefix=dict(img_path='img/train', seg_map_path='labels/train'),
    data_root='/home/kiriy/code/ya_seg/mmsegmentation/data/practicum_dataset',
    img_suffix='.jpg',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(img_scale=(
            256,
            256,
        ), keep_ratio=False, type='Resize'),
        dict(type='PhotoMetricDistortion'),
        dict(type='RandomRotFlip'),
        dict(type='PackSegInputs'),
    ],
    seg_map_suffix='.png',
    type='PracticumDataset')
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(img_scale=(
        256,
        256,
    ), keep_ratio=False, type='Resize'),
    dict(type='PhotoMetricDistortion'),
    dict(type='RandomRotFlip'),
    dict(type='PackSegInputs'),
]
tta_model = dict(type='SegTTAModel')
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(img_path='img/val', seg_map_path='labels/val'),
        data_root=
        '/home/kiriy/code/ya_seg/mmsegmentation/data/practicum_dataset',
        img_suffix='.jpg',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        seg_map_suffix='.png',
        type='PracticumDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_dataset = dict(
    data_prefix=dict(img_path='img/val', seg_map_path='labels/val'),
    data_root='/home/kiriy/code/ya_seg/mmsegmentation/data/practicum_dataset',
    img_suffix='.jpg',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(type='PackSegInputs'),
    ],
    seg_map_suffix='.png',
    type='PracticumDataset')
val_evaluator = dict(
    iou_metrics=[
        'mDice',
    ], type='IoUMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='Visualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(
            init_kwargs=dict(
                auto_connect_arg_parser=True,
                auto_connect_frameworks=True,
                auto_connect_streams=True,
                auto_resource_monitoring=True,
                continue_last_task=False,
                output_uri=None,
                project_name='YaPracticum',
                reuse_last_task_id=False,
                task_name='unet_fcn-256'),
            type='ClearMLVisBackend'),
    ])
work_dir = './work_dirs/unet_fcn-256'
