_base_ = ['vecr_prog_ret0.01_lam0.3.py']

uda = dict(
    type='VECR_ProW',
    invariant=dict(
        source=dict(
            ce=['original'],
            consist=['original', 'fourier']
        ),
        target=dict(
            ce=[('original', 'original')],
            consist=[('original', 'original'), ('fourier', 'fourier')],
        ),
        inv_loss=dict(
            src_weight=50.0,
            tgt_weight=20.0
        ),
    ),
)
