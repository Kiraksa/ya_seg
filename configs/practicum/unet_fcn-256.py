# Наследуемся от базовых конфигов 
# Датасет и гиперпараметры мы подготовили на прошлых этапах
# Архитектуру используем без изменений 
_base_ = [
    '../_base_/models/fcn_unet_s5-d16.py', 
    '../_base_/datasets/practicum_dataset.py',
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
                task_name='unet_fcn-256',
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


# Определим размер входа 
input_suze = (256, 256)

data_preprocessor = dict(size=input_suze)

decode_head=dict(
    num_classes=3,
    loss_decode=[
        dict(
            type='CrossEntropyLoss',
            loss_name='loss_ce',
            use_sigmoid=False,
            loss_weight=1.0
        ),
        dict(type='DiceLoss', loss_name='loss_dice', loss_weight=2.0)
    ]
)
auxiliary_head=dict(
    num_classes=3,
    loss_decode=[
        dict(
            type='CrossEntropyLoss',
            loss_name='loss_ce',
            use_sigmoid=False,
            loss_weight=1.0
        ),
        dict(type='DiceLoss', loss_name='loss_dice', loss_weight=2.0)
    ]
)

model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=decode_head,
    auxiliary_head=auxiliary_head,
    test_cfg=dict(mode="whole")
)