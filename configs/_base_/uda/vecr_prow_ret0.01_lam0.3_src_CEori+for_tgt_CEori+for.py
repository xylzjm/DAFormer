_base_ = ['vecr_prog_ret0.01_lam0.3.py']

uda = dict(
    type='VECR_ProW',
    invariant=dict(
        source=dict(
            ce=['original', 'fourier'],
            consist=None
        ),
        target=dict(
            ce=[('original', 'original'), ('fourier', 'fourier')],
            consist=None,
        ),
        inv_loss=dict(
            src_weight=1.0,
            tgt_weight=1.0
        ),
    ),
)
