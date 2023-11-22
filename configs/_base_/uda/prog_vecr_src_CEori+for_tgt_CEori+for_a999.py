_base_ = ['prog_vecr_a999.py']

uda = dict(
    type='Prog_VECR',
    invariant=dict(
        source=dict(
            ce=('original', 'fourier'),
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
