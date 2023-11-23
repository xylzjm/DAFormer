import os
import random
import re
from copy import deepcopy

import numpy as np
import torch
from matplotlib import pyplot as plt
from mmseg.core import add_prefix
from mmseg.models import UDA, build_segmentor
from mmseg.models.uda.uda_decorator import UDADecorator, get_module
from mmseg.models.utils.color_transforms import fourier_transform, night_fog_filter
from mmseg.models.utils.dacs_transforms import (
    denorm,
    get_class_masks,
    get_mean_std,
    strong_transform,
)
from mmseg.models.utils.visualization import subplotimg
from timm.models.layers import DropPath
from torch.nn.modules.dropout import _DropoutNd


@UDA.register_module()
class VECR(UDADecorator):
    def __init__(self, **cfg) -> None:
        super(VECR, self).__init__(**cfg)
        self.local_iter = 0
        self.max_iters = cfg['max_iters']
        self.debug_img_interval = cfg['debug_img_interval']
        self.alpha = cfg['alpha']
        self.pseudo_threshold = cfg['pseudo_threshold']
        # dacs transform
        self.blur = cfg['blur']
        self.color_jitter_s = cfg['color_jitter_strength']
        self.color_jitter_p = cfg['color_jitter_probability']
        # color transform
        self.fourier_rat = cfg['fourier_ratio']
        self.fourier_lam = cfg['fourier_lambda']

        ema_cfg = deepcopy(cfg['model'])
        self.ema_model = build_segmentor(ema_cfg)

    def get_ema_model(self):
        return get_module(self.ema_model)

    def _init_ema_weights(self):
        for param in self.get_ema_model().parameters():
            param.detach_()
        mp = list(self.get_model().parameters())
        mcp = list(self.get_ema_model().parameters())
        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()

    def _update_ema(self, iter):
        alpha_teacher = min(1 - 1 / (iter + 1), self.alpha)
        for ema_param, param in zip(
            self.get_ema_model().parameters(), self.get_model().parameters()
        ):
            if not param.data.shape:  # scalar tensor
                ema_param.data = (
                    alpha_teacher * ema_param.data + (1 - alpha_teacher) * param.data
                )
            else:
                ema_param.data[:] = (
                    alpha_teacher * ema_param[:].data[:]
                    + (1 - alpha_teacher) * param[:].data[:]
                )

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """

        optimizer.zero_grad()
        log_vars = self(**data_batch)
        optimizer.step()

        log_vars.pop('loss', None)  # remove the unnecessary 'loss'
        outputs = dict(log_vars=log_vars, num_samples=len(data_batch['img_metas']))
        return outputs

    def forward_train(
        self, img, img_metas, gt_semantic_seg, target_img, target_img_metas
    ):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        log_vars = {}
        batch_size = img.shape[0]
        dev = img.device

        # Init/update ema model
        if self.local_iter == 0:
            self._init_ema_weights()
            # assert _params_equal(self.get_ema_model(), self.get_model())
        if self.local_iter > 0:
            self._update_ema(self.local_iter)
            # assert not _params_equal(self.get_ema_model(), self.get_model())
            # assert self.get_ema_model().training
        for m in self.get_ema_model().modules():
            if isinstance(m, _DropoutNd):
                m.training = False
            if isinstance(m, DropPath):
                m.training = False

        means, stds = get_mean_std(img_metas, dev)
        strong_parameters = {
            'mix': None,
            'color_jitter': random.uniform(0, 1),
            'color_jitter_s': self.color_jitter_s,
            'color_jitter_p': self.color_jitter_p,
            'blur': random.uniform(0, 1) if self.blur else 0,
            'mean': means[0].unsqueeze(0),  # assume same normalization
            'std': stds[0].unsqueeze(0),
        }

        # color style transfer
        night_map = [
            re.search('night | twilight', meta['filename']) is not None
            for meta in target_img_metas
        ]
        tgt_ib_img = night_fog_filter(
            target_img, means, stds, night_map, mode='hsv-s-w4'
        )
        tgt_fr_img = [None] * batch_size
        for i in range(batch_size):
            tgt_fr_img[i] = fourier_transform(
                data=torch.stack((tgt_ib_img[i], img[i])),
                mean=means[0].unsqueeze(0),
                std=stds[0].unsqueeze(0),
                ratio=self.fourier_rat,
                lam=self.fourier_lam,
            )
        tgt_fr_img = torch.cat(tgt_fr_img)
        del tgt_ib_img

        # train student with source
        src_losses = self.get_model().forward_train(
            img, img_metas, gt_semantic_seg, return_feat=False
        )
        src_losses = add_prefix(src_losses, 'src')
        src_loss, src_log = self._parse_losses(src_losses)
        log_vars.update(src_log)
        src_loss.backward()

        # generate pseudo-label
        with torch.no_grad():
            ema_tgt_logits = self.get_ema_model().encode_decode(
                tgt_fr_img, target_img_metas
            )
            ema_tgt_softmax = torch.softmax(ema_tgt_logits, dim=1)
            pseudo_prob, pseudo_lbl = torch.max(ema_tgt_softmax, dim=1)
            # estimate pseudo-weight
            pseudo_msk = pseudo_prob.ge(self.pseudo_threshold).long() == 1
            pseudo_size = np.size(np.array(pseudo_lbl.cpu()))
            pseudo_weight = torch.sum(pseudo_msk).item() / pseudo_size
            pseudo_weight = pseudo_weight * torch.ones(pseudo_lbl.shape, device=dev)
            # get gt pixel-weight
            gt_pixel_weight = torch.ones(pseudo_weight.shape, device=dev)

        # prepare target train data
        mix_msks = get_class_masks(gt_semantic_seg)
        mixed_img, mixed_fr_img, mixed_lbl = (
            [None] * batch_size,
            [None] * batch_size,
            [None] * batch_size,
        )
        for i in range(batch_size):
            strong_parameters['mix'] = mix_msks[i]
            mixed_img[i], mixed_lbl[i] = strong_transform(
                strong_parameters,
                data=torch.stack((img[i], target_img[i])),
                target=torch.stack((gt_semantic_seg[i][0], pseudo_lbl[i])),
            )
            mixed_fr_img[i], pseudo_weight[i] = strong_transform(
                strong_parameters,
                data=torch.stack((img[i], tgt_fr_img[i])),
                target=torch.stack((gt_pixel_weight[i], pseudo_weight[i])),
            )
        mixed_img, mixed_fr_img, mixed_lbl = (
            torch.cat(mixed_img),
            torch.cat(mixed_fr_img),
            torch.cat(mixed_lbl),
        )

        # train student with target
        mixed_losses = self.get_model().forward_train(
            mixed_img, img_metas, mixed_lbl, pseudo_weight, return_feat=False
        )
        mixed_losses = add_prefix(mixed_losses, 'mix')
        mixed_loss, mixed_log = self._parse_losses(mixed_losses)
        log_vars.update(mixed_log)
        mixed_loss.backward()
        mixed_fr_losses = self.get_model().forward_train(
            mixed_fr_img, img_metas, mixed_lbl, pseudo_weight, return_feat=False
        )
        mixed_fr_losses = add_prefix(mixed_fr_losses, 'frmix')
        mixed_fr_loss, mixed_fr_log = self._parse_losses(mixed_fr_losses)
        log_vars.update(mixed_fr_log)
        mixed_fr_loss.backward()

        if self.local_iter % self.debug_img_interval == 0:
            out_dir = os.path.join(self.train_cfg['work_dir'], 'class_mix_debug')
            os.makedirs(out_dir, exist_ok=True)
            vis_img = torch.clamp(denorm(img, means, stds), 0, 1)
            vis_tgt_img = torch.clamp(denorm(target_img, means, stds), 0, 1)
            vis_mixed_img = torch.clamp(denorm(mixed_img, means, stds), 0, 1)
            vis_tgtfr_img = torch.clamp(denorm(tgt_fr_img, means, stds), 0, 1)
            vis_mixfr_img = torch.clamp(denorm(mixed_fr_img, means, stds), 0, 1)
            # vis_ib_img = torch.clamp(denorm(tgt_ib_img, means, stds), 0, 1)
            for j in range(batch_size):
                rows, cols = 3, 4
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
                subplotimg(
                    axs[0][0],
                    vis_img[j],
                    f'{os.path.basename(img_metas[j]["filename"])}',
                )
                subplotimg(
                    axs[1][0],
                    vis_tgt_img[j],
                    f'{os.path.basename(target_img_metas[j]["filename"])}',
                )
                subplotimg(
                    axs[0][1], gt_semantic_seg[j], 'Source Seg GT', cmap='cityscapes'
                )
                subplotimg(
                    axs[1][1],
                    pseudo_lbl[j],
                    'Target Seg Pseudo',
                    cmap='cityscapes',
                )
                subplotimg(axs[0][2], vis_mixed_img[j], 'Mixed Image')
                subplotimg(
                    axs[1][2], mixed_lbl[j], 'Mixed Seg GT', cmap='cityscapes'
                )
                subplotimg(axs[0][3], mix_msks[j][0], 'Domain Mask', cmap='gray')
                subplotimg(axs[1][3], pseudo_weight[j], 'Pseudo W.', vmin=0, vmax=1)
                subplotimg(axs[2][1], vis_tgtfr_img[j], 'Target FR Image')
                subplotimg(axs[2][2], vis_mixfr_img[j], 'Mixed FR Image')
                # subplotimg(axs[2][3], vis_ib_img[j], 'Target IB')
                for ax in axs.flat:
                    ax.axis('off')
                plt.savefig(
                    os.path.join(out_dir, f'{(self.local_iter):06d}_{j}.png')
                )
                plt.close()
        self.local_iter += 1

        return log_vars
