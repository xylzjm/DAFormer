# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# DAFormer (with context-aware feature fusion) in Tab. 7

_base_ = ['daformer_conv1_mitb5.py']

norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    decode_head=dict(
        decoder_params=dict(
            fusion_cfg=dict(
                _delete_=True,
                type='aspp',
                sep=True,
                dilations=(1, 6, 12, 18),
                pool=False,
                act_cfg=dict(type='ReLU'),
                norm_cfg=norm_cfg,
            )
        ),
        loss_decode=dict(
            type='LogitConstraintLoss', use_sigmoid=False, loss_weight=1.0,
            class_weight=[0.8373, 0.9180, 0.8660, 1.0345, 1.0166, 0.9969,
                          0.9754, 1.0489, 0.8786, 1.0023, 0.9539, 0.9843,
                          1.1116, 0.9037, 1.0865, 1.0955, 1.0865, 1.1529,
                          1.0507]
        ),
    )
)
