_base_ = ['vecr_ret0.01_lam0.3.py']

uda = dict(
    type='VECR_ProG',
    proto=dict(
        num_class=19,
        feat_dim=256,
        ignore_index=255,
        momentum=0,
    ),
    proto_resume='pretrained/prototype_source.pth',
)
use_ddp_wrapper = True
