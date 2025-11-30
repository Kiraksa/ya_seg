
_base_ = [
    '../_base_/models/fcn_unet_s5-d16.py', 
    '../_base_/datasets/practicum_dataset_drop.py',
    '../_base_/default_runtime.py', 
    '../_base_/schedules/practicum_schedule.py'
]

visualizer = dict(
    type='Visualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),      # сохраняем логи локально
        dict(
            type='ClearMLVisBackend',      # дублируем всё в ClearML
            init_kwargs=dict(
                project_name='YaPracticum',
                task_name='unet_fcn-256_v5',
                reuse_last_task_id=False,
                continue_last_task=False,
                output_uri=None,
                auto_connect_arg_parser=True,
                auto_connect_frameworks=True,
                auto_resource_monitoring=True,
                auto_connect_streams=True,
            )
        )     
    ]
)

crop_size = (128, 128)
img_size=(256, 256)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),

    dict(
        type='Resize',
        scale=img_size,  
    ),

    dict(
        type='RandomFlip',
        prob=0.5,
        direction=['horizontal', 'vertical']
    ),

    dict(type='PhotoMetricDistortion'),

    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.8, 1.2),
        saturation_range=(0.8, 1.2),
        hue_delta=10
    ),

    dict(type='PackSegInputs'),
]

input_suze = (256, 256)

data_preprocessor = dict(size=input_suze)

class_weight = [0.024339, 0.44406453, 0.53159648] 

decode_head = dict(
    num_classes=3,
    loss_decode=[
        dict(
            type='CrossEntropyLoss',
            loss_name='loss_ce',
            use_sigmoid=False,
            loss_weight=1.0,
            class_weight=class_weight
        ),
        dict(
            type='DiceLoss',
            loss_name='loss_dice',
            loss_weight=3.0 
        )
    ]
)
auxiliary_head = dict(
    num_classes=3,
    loss_decode=[
        dict(
            type='CrossEntropyLoss',
            loss_name='loss_ce',
            use_sigmoid=False,
            loss_weight=1.0,
            class_weight=class_weight
        ),
        dict(
            type='DiceLoss',
            loss_name='loss_dice',
            loss_weight=3.0
        )
    ]
)

model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=decode_head,
    auxiliary_head=auxiliary_head,
    test_cfg=dict(mode="whole")
)

epoch_num = 300


optimizer = dict(type='AdamW', lr=7e-4, weight_decay=2e-4)

optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)

param_scheduler = [
    dict(
        type='PolyLR',
            eta_min=1e-4,
            power=0.9,
            begin=0,
            end=epoch_num,
            by_epoch=True
    )
]