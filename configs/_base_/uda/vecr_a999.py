uda = dict(
    type='VECR',
    alpha=0.999,
    pseudo_threshold=0.968,
    blur=True,
    color_jitter_strength=0.2,
    color_jitter_probability=0.2,
    fourier_lambda=0.3,
    debug_img_interval=1000,
)
use_ddp_wrapper = True
