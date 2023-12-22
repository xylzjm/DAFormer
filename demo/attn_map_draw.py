import cv2
import matplotlib.pyplot as plt
import mmcv
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import ArgumentParser
from PIL import Image
from torch import Tensor
from typing import Callable, Dict, Iterable

from mmseg.apis import inference_segmentor, init_segmentor
from mmseg.core.evaluation import get_classes, get_palette
from mmseg.models.utils.visualization import subplotimg
from tools.test import update_legacy_cfg


def visualize_maps_heads(imgs, attn_maps, out_dir=None, save_dir=None):
    assert isinstance(attn_maps, dict)
    rows, cols = len(attn_maps) + 1, (attn_maps[imgs[0]].shape[1] + 1)
    fig, axs = plt.subplots(
        rows,
        cols,
        figsize=(3 * cols, 3 * rows),
        gridspec_kw={
            'hspace': 0.1,
            'wspace': 0,
            'top': 0.95,
            'bottom': 0,
            'right': 1,
            'left': 0,
        },
    )
    for i in range(len(imgs)):
        fimg_name = os.path.split(imgs[i])[-1]
        img_name = os.path.splitext(fimg_name)[0]
        h, w = attn_maps[imgs[i]].shape[2:]
        has_odd = h % 2 != 0 or w % 2 != 0
        size = max(h, w)
        p = attn_maps.pop(imgs[i])
        if h != w:
            p = F.interpolate(
                p, size=(size, size), mode='bicubic', align_corners=has_odd
            )
        for j in range(cols):
            if j == 0:
                img = Image.open(imgs[i], mode='r')
                axs[i][j].imshow(img)
                axs[i][j].set_title(f'{img_name}')
                # pass
            else:
                normed_p = p / p.max()
                normed_p = (normed_p * 255).type(torch.uint8)
                subplotimg(axs[i][j], normed_p[0][j - 1], f'Query {j}')
    for ax in axs.flat:
        ax.axis('off')
    if out_dir is not None and save_dir is not None:
        plt.savefig(os.path.join(out_dir, f'{save_dir}.png'))
        plt.close()
    else:
        plt.show()


def visualize_grid_attention_v2(
    img_path,
    save_path,
    attention_mask,
    ratio=1,
    cmap="jet",
    save_image=False,
    save_original_image=False,
    quality=200,
):
    """
    img_path:   image file path to load
    save_path:  image file path to save
    attention_mask:  2-D attention map with np.array type, e.g, (h, w) or (w, h)
    ratio:  scaling factor to scale the output h and w
    cmap:  attention style, default: "jet"
    quality:  saved image quality
    """
    print("load image from: ", img_path)
    img = Image.open(img_path, mode='r')
    img_h, img_w = img.size[0], img.size[1]
    plt.subplots(nrows=1, ncols=1, figsize=(0.02 * img_h, 0.02 * img_w))

    # scale the image
    img_h, img_w = int(img.size[0] * ratio), int(img.size[1] * ratio)
    img = img.resize((img_h, img_w))
    plt.imshow(img, alpha=1)
    plt.axis('off')

    # normalize the attention map
    mask = cv2.resize(attention_mask, (img_h, img_w))
    normed_mask = mask / mask.max()
    normed_mask = (normed_mask * 255).astype('uint8')
    plt.imshow(normed_mask, alpha=0.5, interpolation='nearest', cmap=cmap)

    if save_image:
        # build save path
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        img_name = img_path.split('/')[-1].split('.')[0] + "_with_attention.jpg"
        img_with_attention_save_path = os.path.join(save_path, img_name)

        # pre-process and save image
        print("save image to: " + save_path + " as " + img_name)
        plt.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(img_with_attention_save_path, dpi=quality)

    if save_original_image:
        # build save path
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        # save original image file
        print("save original image at the same time")
        img_name = img_path.split('/')[-1].split('.')[0] + "_original.jpg"
        original_image_save_path = os.path.join(save_path, img_name)
        img.save(original_image_save_path, quality=quality)


class FeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module, layers: Iterable[str]):
        super().__init__()
        self.model = model
        self.layers = layers
        self._features = {layer: torch.empty(0) for layer in layers}

        for layer_id in layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, __, output):
            self._features[layer_id] = output

        return fn

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        _ = inference_segmentor(self.model, x)
        return self._features


def main():
    parser = ArgumentParser()
    parser.add_argument('-img', nargs='+', help='Image file')
    parser.add_argument('-cfg', help='Config file')
    parser.add_argument('-ckpt', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference'
    )
    parser.add_argument(
        '--palette',
        default='cityscapes',
        help='Color palette used for segmentation map',
    )
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.',
    )
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    cfg = mmcv.Config.fromfile(args.cfg)
    cfg = update_legacy_cfg(cfg)
    model = init_segmentor(
        cfg,
        args.ckpt,
        device=args.device,
        classes=get_classes(args.palette),
        palette=get_palette(args.palette),
        revise_checkpoint=[(r'^module\.', ''), ('model.', '')],
    )
    # test images
    layer = 'decode_head.block.0.attn.attn_drop'
    # backbone_layer = 'backbone.block4.0.attn.attn_drop'
    taskformer_features = FeatureExtractor(model, layers=[layer])
    attn_maps = {}
    for i in range(len(args.img)):
        attn_maps[args.img[i]] = taskformer_features(args.img[i]).pop(layer).cpu()
    print({name: output.shape for name, output in attn_maps.items()})
    # show the results
    out_dir = '/hy-tmp/DAFormer/demo'
    # ffile_name = os.path.split(args.img[0])[-1]
    # save_dir = os.path.splitext(ffile_name)[0]
    # visualize_heads(attn_maps[args.img[0]].cpu(),
    #                 4,
    #                 out_dir=out_dir,
    #                 save_dir=save_dir)
    save_dir = layer
    visualize_maps_heads(args.img, attn_maps, out_dir=out_dir, save_dir=save_dir)
    print(f'Save picture to {out_dir}/{save_dir}.png')


if __name__ == '__main__':
    main()
