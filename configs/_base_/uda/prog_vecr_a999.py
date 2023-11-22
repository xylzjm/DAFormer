_base_ = ['vecr_a999.py']

uda = dict(
    type='Prog_VECR',
    proto=dict(
        num_class=19,
        feat_dim=256,
        ignore_index=255,
        momentum=0,
    ),
    proto_resume='pretrained/prototype_source_initial_value.pth',
)
use_ddp_wrapper = True
