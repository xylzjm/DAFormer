_base_ = ['taskformer_conv1_mitb5.py']

norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    decode_head=dict(
        decoder_params=dict(
            query_cfg=dict(
                type='attn',
                query_dim=512,
                num_heads=8,
                mlp_ratio=4,
                qkv_bias=True,
                qk_scale=None,
                sr_ratio=4,
            ),
            fusion_cfg=dict(
                _delete_=True,
                type='aspp',
                sep=True,
                dilations=(1, 6, 12, 18),
                pool=False,
                act_cfg=dict(type='ReLU'),
                norm_cfg=norm_cfg,
            ),
        ),
        loss_decode=dict(
            type='LogitConstraintLoss', use_sigmoid=False, loss_weight=1.0
        ),
    )
)
