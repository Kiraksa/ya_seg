epoch_num = 300

# Подготовим оптимайзер, используем дефолтные параметры 
optimizer = dict(type='AdamW', lr=0.001, weight_decay=0.1)

optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)

# Определяем распорядок LR
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


train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=epoch_num)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')


default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=1),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=True, interval=20),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', interval=10, draw=True)
)