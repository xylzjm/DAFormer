_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/models/daformer_sepaspp_mitb5.py',
    '../_base_/datasets/cityscapes_half_640x640.py',
    '../_base_/schedules/adamw.py',
    '../_base_/schedules/poly10warm.py',
]
# Random Seed
seed = 0
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
crop_size = (640, 640)
pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1280, 640)),
    # dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.0),
    # dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    # dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
data = dict(samples_per_gpu=1, workers_per_gpu=4, train=dict(pipeline=pipeline))
# Prototype configuration
proto = dict(
    num_class=19,
    feat_dim=256,
    ignore_index=255,
    momentum=0,
)
optimizer = dict(
    lr=6e-05,
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=10.0),
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0),
        )
    ),
)
n_gpus = 1
# Meta Information for Result Analysis
name = 'cityscapes_proto_daformer_sepaspp_mitb5_s0'
exp = 'basic'
name_dataset = 'cityscapes'
name_architecture = 'daformer_sepaspp_mitb5'
name_encoder = 'mitb5'
name_decoder = 'daformer_sepaspp'
name_uda = None
name_opt = 'adamw_6e-05_pmTrue_poly10warm_1x2_40k'
