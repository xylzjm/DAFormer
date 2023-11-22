import torch
import math
import numpy as np
import torch.fft as fft

from .dacs_transforms import denorm_, renorm_


def DarkChannel(im):
    dc, _ = torch.min(im, dim=-1)
    return dc


def AtmLight(im, dark):
    h, w = im.shape[:2]
    imsz = h * w
    numpx = int(max(math.floor(imsz / 1000), 1))
    darkvec = dark.reshape(imsz, 1)
    imvec = im.reshape(imsz, 3)

    indices = darkvec.argsort(0)
    indices = indices[(imsz - numpx) : imsz]

    atmsum = torch.zeros([1, 3]).cuda()
    for ind in range(1, numpx):
        atmsum = atmsum + imvec[indices[ind]]

    A = atmsum / numpx
    return A


def DarkIcA(im, A):
    im3 = torch.empty_like(im)
    for ind in range(0, 3):
        im3[:, :, ind] = im[:, :, ind] / A[0, ind]
    return DarkChannel(im3)


def get_saturation(im):
    saturation = (im.max(-1)[0] - im.min(-1)[0]) / (im.max(-1)[0] + 1e-10)
    return saturation


def process(im, defog_A, IcA, mode='hsv-s-w4'):
    if mode == 'hsv-s-w4':
        img_s = get_saturation(im)
        s = (-img_s.mean() * 4).exp()
        param = torch.ones_like(img_s) * s
    else:
        raise NotImplementedError(f'{mode} not supported yet!')

    param = param[None, :, :, None]
    tx = 1 - param * IcA

    # tx_1 = torch.tile(tx, [1, 1, 1, 3])
    dev = tx.device
    tx = tx.cpu()
    tx_1 = np.tile(tx.numpy(), (1, 1, 1, 3))
    tx_1 = torch.tensor(tx_1).cuda()
    return (im - defog_A[:, None, None, :]) / torch.maximum(
        tx_1, torch.tensor(0.01, device=dev)
    ) + defog_A[:, None, None, :]


def blur_filter(X, mode):
    X = X.permute(1, 2, 0).contiguous()

    dark = DarkChannel(X)
    defog_A = AtmLight(X, dark)
    IcA = DarkIcA(X, defog_A)

    IcA = IcA.unsqueeze(-1)

    return process(X, defog_A, IcA, mode=mode)[0].permute(2, 0, 1).contiguous()


def night_fog_filter(normed_img, means, stds, night_map, mode='hsv-s-w4'):
    img = normed_img * stds + means
    img /= 255.0
    bs = img.shape[0]
    assert bs == len(night_map)
    for i in range(bs):
        if night_map[i]:
            img[i] = 1 - blur_filter(1 - img[i], mode=mode)
        else:
            img[i] = blur_filter(img[i], mode=mode)
    img *= 255.0
    normed_img = (img.float() - means) / stds
    return normed_img


def fourier_transform(data, mean, std, ratio=0.01, lam=0.7):
    denorm_(data, mean, std)
    data = amplitude_mixup(data[0], data[1], ratio, lam)
    renorm_(data, mean, std)
    return data

def fftshift(x: torch.Tensor, dim=None):
    if dim is None:
        dim = tuple(range(x.ndim))
        shift = [d // 2 for d in x.shape]
    elif isinstance(dim, int):
        shift = x.shape[dim] // 2
    else:
        shift = [x.shape[d] // 2 for d in dim]

    return torch.roll(x, shift, dim)


def ifftshift(x: torch.Tensor, dim=None):
    if dim is None:
        dim = tuple(range(x.ndim))
        shift = [-(d // 2) for d in x.shape]
    elif isinstance(dim, int):
        shift = -(x.shape[dim] // 2)
    else:
        shift = [-(x.shape[d] // 2) for d in dim]

    return torch.roll(x, shift, dim)

def get_mix_bbox(amp, ratio):
    _, h, w = amp.shape
    b = np.floor(np.amin((h, w)) * ratio).astype(int)
    c_b = np.floor(b / 2.0).astype(int)
    c_h = np.floor(h / 2.0).astype(int)
    c_w = np.floor(w / 2.0).astype(int)

    h1, h2 = c_h - c_b, c_h + c_b + 1
    w1, w2 = c_w - c_b, c_w + c_b + 1
    # print_log(f'amp shape: {img1_amp.shape}', 'mmseg')
    # print_log(f'h1: {h1}, h2: {h2}, w1: {w1}, w2: {w2}', 'mmseg')
    return h1, h2, w1, w2


def amplitude_mixup(img1, img2, ratio, lam):
    """Input image size: tensor of [C, H, W]"""
    assert img1.shape == img2.shape

    img1_fft = fft.fftn(img1, dim=(-2, -1))
    img2_fft = fft.fftn(img2, dim=(-2, -1))

    img1_amp, img1_pha = torch.abs(img1_fft), torch.angle(img1_fft)
    img2_amp, img2_pha = torch.abs(img2_fft), torch.angle(img2_fft)
    h1, h2, w1, w2 = get_mix_bbox(img1_amp, ratio)

    img1_amp = fftshift(img1_amp, dim=(-2, -1))
    img2_amp = fftshift(img2_amp, dim=(-2, -1))

    img1_amp_, img2_amp_ = torch.clone(img1_amp), torch.clone(img2_amp)
    img1_amp[..., h1:h2, w1:w2] = (
        lam * img2_amp_[..., h1:h2, w1:w2] + (1 - lam) * img1_amp_[..., h1:h2, w1:w2]
    )

    img1_amp = ifftshift(img1_amp, dim=(-2, -1))
    img2_amp = ifftshift(img2_amp, dim=(-2, -1))

    new_img1 = img1_amp * torch.exp(1j * img1_pha)
    new_img1 = torch.real(fft.ifftn(new_img1, dim=(-2, -1)))
    new_img1 = torch.clamp(new_img1, 0, 1)

    return new_img1.unsqueeze(0)