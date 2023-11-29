import os
import random
import re

import matplotlib.pyplot as plt
import mmcv
import numpy as np
import torch
import torch.nn as nn
from mmseg.core import add_prefix
from mmseg.models import UDA
from mmseg.models.uda.vecr import VECR
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
class VECR_ProG(VECR):
    def __init__(self, **cfg) -> None:
        super(VECR_ProG, self).__init__(**cfg)
        self.ignore_index = 255
        self.proto_cfg = cfg['proto']
        self.proto_resume = cfg['proto_resume']

        assert self.num_classes == self.proto_cfg['num_class']
        assert self.ignore_index == self.proto_cfg['ignore_index']
        self.proto_estimator = None

        assert cfg['invariant'].get('source', None)
        assert cfg['invariant'].get('target', None)
        self.inv_cfg = cfg['invariant']
        assert isinstance(self.inv_cfg['source']['ce'], (list, tuple))
        assert isinstance(self.inv_cfg['target']['ce'], (list, tuple))
        self.src_invflag = self.inv_cfg['source']['consist'] is not None
        self.tgt_invflag = self.inv_cfg['target']['consist'] is not None
        mmcv.print_log(
            f'src_invflag={self.src_invflag}, tgt_invflag={self.tgt_invflag}', 'mmcv'
        )
        self.src_invlam = self.inv_cfg['inv_loss']['src_weight']
        self.tgt_invlam = self.inv_cfg['inv_loss']['tgt_weight']
        mmcv.print_log(
            f'src_invlam: {self.src_invlam}, tgt_invlam: {self.tgt_invlam}', 'mmcv'
        )

    def feat_consist_loss(self, f1, f2, weight):
        mse_criterion = nn.MSELoss()
        loss = mse_criterion(f1, f2)
        inv_loss, inv_log = self._parse_losses({'loss_inv_feat': weight * loss})
        inv_log.pop('loss', None)
        return inv_loss, inv_log

    def forward_train(
        self, img, img_metas, gt_semantic_seg, target_img, target_img_metas
    ):
        log_vars = {}
        batch_size = img.shape[0]
        dev = img.device

        # Init/update ema model and prototype
        if self.local_iter == 0:
            self._init_ema_weights()
            # assert _params_equal(self.get_ema_model(), self.get_model())
            """ self.proto_estimator = (
                PrototypeEstimator(self.proto_cfg, resume=self.proto_resume)
                if self.src_invflag or self.tgt_invflag
                else None
            ) """
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
        tgt_fr_img, src_fr_img = [None] * batch_size, [None] * batch_size
        for i in range(batch_size):
            tgt_fr_img[i] = fourier_transform(
                data=torch.stack((tgt_ib_img[i], img[i])),
                mean=means[0].unsqueeze(0),
                std=stds[0].unsqueeze(0),
                ratio=self.fourier_rat,
                lam=self.fourier_lam,
            )
            src_fr_img[i] = fourier_transform(
                data=torch.stack((img[i], target_img[i])),
                mean=means[0].unsqueeze(0),
                std=stds[0].unsqueeze(0),
                ratio=self.fourier_rat,
                lam=self.fourier_lam,
            )
        tgt_fr_img, src_fr_img = torch.cat(tgt_fr_img), torch.cat(src_fr_img)
        del tgt_ib_img

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
            del ema_tgt_logits, ema_tgt_softmax, pseudo_prob

        # prepare target train data
        tgt_semantic_seg = pseudo_lbl.clone()
        tgt_semantic_seg[pseudo_weight == 0.0] = self.ignore_index
        tgt_semantic_seg = tgt_semantic_seg.unsqueeze(1)
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
                data=torch.stack((src_fr_img[i], tgt_fr_img[i])),
                target=torch.stack((gt_pixel_weight[i], pseudo_weight[i])),
            )
        mixed_img, mixed_fr_img, mixed_lbl = (
            torch.cat(mixed_img),
            torch.cat(mixed_fr_img),
            torch.cat(mixed_lbl),
        )

        # update feature statistics
        """ if self.src_invflag or self.tgt_invflag:
            with torch.no_grad():
                src_emafeat = self.get_ema_model().extract_decfeat(img)
                tgt_emafeat = self.get_ema_model().extract_decfeat(target_img)

                assert src_emafeat.shape == tgt_emafeat.shape
                b, a, h, w = src_emafeat.shape
                src_emafeat = (
                    src_emafeat.permute(0, 2, 3, 1).contiguous().view(b * h * w, a)
                )
                tgt_emafeat = (
                    tgt_emafeat.permute(0, 2, 3, 1).contiguous().view(b * h * w, a)
                )
                src_mask = (
                    resize(gt_semantic_seg.float(), size=(h, w), mode='nearest')
                    .long()
                    .contiguous()
                    .view(b * h * w, )
                )
                tgt_mask = (
                    resize(tgt_semantic_seg.float(), size=(h, w), mode='nearest')
                    .long()
                    .contiguous()
                    .view(b * h * w, )
                )
                self.proto_estimator.update(feat=src_emafeat, label=src_mask)
                self.proto_estimator.update(feat=tgt_emafeat, label=tgt_mask)
                del src_emafeat, tgt_emafeat """

        # train student with source
        src_featpool = {}
        for src_args in self.inv_cfg['source']['ce']:
            assert isinstance(src_args, str)
            if src_args == 'original':
                src_losses = self.get_model().forward_train(
                    img, img_metas, gt_semantic_seg, return_decfeat=self.src_invflag
                )
                if (
                    self.src_invflag
                    and src_args in self.inv_cfg['source']['consist']
                ):
                    src_featpool[src_args] = src_losses.pop('dec_feat')
                assert 'dec_feat' not in src_losses
                src_losses = add_prefix(src_losses, 'src_ori')
                src_loss, src_log = self._parse_losses(src_losses)
                log_vars.update(src_log)
                src_loss.backward(retain_graph=self.src_invflag)
            elif src_args == 'fourier':
                src_losses = self.get_model().forward_train(
                    src_fr_img,
                    img_metas,
                    gt_semantic_seg,
                    return_decfeat=self.src_invflag,
                )
                if (
                    self.src_invflag
                    and src_args in self.inv_cfg['source']['consist']
                ):
                    src_featpool[src_args] = src_losses.pop('dec_feat')
                assert 'dec_feat' not in src_losses
                src_losses = add_prefix(src_losses, 'src_for')
                src_loss, src_log = self._parse_losses(src_losses)
                log_vars.update(src_log)
                src_loss.backward(retain_graph=self.src_invflag)
            else:
                raise ValueError(f'{src_args} not allowed in source CE arguments')
        # source domain feature invariance loss
        if self.src_invflag:
            for inv_args in self.inv_cfg['source']['consist']:
                assert isinstance(inv_args, str)
                if inv_args == 'original' and inv_args not in src_featpool:
                    src_featpool[inv_args] = self.get_model().extract_decfeat(img)
                elif inv_args == 'fourier' and inv_args not in src_featpool:
                    src_featpool[inv_args] = self.get_model().extract_decfeat(
                        src_fr_img
                    )
            assert len(src_featpool) == len(self.inv_cfg['source']['consist'])
            src_invloss, src_invlog = self.feat_consist_loss(
                src_featpool[self.inv_cfg['source']['consist'][0]],
                src_featpool[self.inv_cfg['source']['consist'][1]],
                weight=50.0,
            )
            log_vars.update(add_prefix(src_invlog, 'src'))
            src_invloss.backward()
        # Garbage collection
        try:
            del src_loss, src_invloss, src_featpool
        except NameError:
            pass

        # train student with target
        tgt_featpool = {}
        for tgt_args in self.inv_cfg['target']['ce']:
            assert isinstance(tgt_args, (list, tuple))
            if tgt_args == ('original', 'original'):
                mix_losses = self.get_model().forward_train(
                    mixed_img,
                    img_metas,
                    mixed_lbl,
                    pseudo_weight,
                    return_decfeat=self.tgt_invflag,
                )
                if (
                    self.tgt_invflag
                    and tgt_args in self.inv_cfg['target']['consist']
                ):
                    tgt_featpool[tgt_args] = mix_losses.pop('dec_feat')
                assert 'dec_feat' not in mix_losses
                mix_losses = add_prefix(mix_losses, 'mix_ori')
                mix_loss, mix_log = self._parse_losses(mix_losses)
                log_vars.update(mix_log)
                mix_loss.backward(retain_graph=self.tgt_invflag)
            elif tgt_args == ('fourier', 'fourier'):
                mix_losses = self.get_model().forward_train(
                    mixed_fr_img,
                    img_metas,
                    mixed_lbl,
                    pseudo_weight,
                    return_decfeat=self.tgt_invflag,
                )
                if (
                    self.tgt_invflag
                    and tgt_args in self.inv_cfg['target']['consist']
                ):
                    tgt_featpool[tgt_args] = mix_losses.pop('dec_feat')
                assert 'dec_feat' not in mix_losses
                mix_losses = add_prefix(mix_losses, 'mix_for')
                mix_loss, mix_log = self._parse_losses(mix_losses)
                log_vars.update(mix_log)
                mix_loss.backward(retain_graph=self.tgt_invflag)
            else:
                raise ValueError(f'{tgt_args} not allowed in target CE arguments')
        # target domain feature invariance loss
        if self.tgt_invflag:
            for inv_args in self.inv_cfg['target']['consist']:
                assert isinstance(inv_args, (list, tuple))
                if (
                    inv_args == ('original', 'original')
                    and inv_args not in tgt_featpool
                ):
                    tgt_featpool[inv_args] = self.get_model().extract_decfeat(
                        mixed_img
                    )
                elif (
                    inv_args == ('fourier', 'fourier')
                    and inv_args not in tgt_featpool
                ):
                    tgt_featpool[inv_args] = self.get_model().extract_decfeat(
                        mixed_fr_img
                    )
            assert len(tgt_featpool) == len(self.inv_cfg['target']['consist'])
            tgt_invloss, tgt_invlog = self.feat_consist_loss(
                tgt_featpool[self.inv_cfg['target']['consist'][0]],
                tgt_featpool[self.inv_cfg['target']['consist'][1]],
                weight=20.0,
            )
            log_vars.update(add_prefix(tgt_invlog, 'tgt'))
            tgt_invloss.backward()
        # Garbage collection
        try:
            del mix_loss, tgt_invloss, tgt_featpool
        except NameError:
            pass

        if self.local_iter % self.debug_img_interval == 0:
            out_dir = os.path.join(self.train_cfg['work_dir'], 'class_mix_debug')
            os.makedirs(out_dir, exist_ok=True)
            vis_img = torch.clamp(denorm(img, means, stds), 0, 1)
            vis_tgt_img = torch.clamp(denorm(target_img, means, stds), 0, 1)
            vis_mixed_img = torch.clamp(denorm(mixed_img, means, stds), 0, 1)
            vis_fr_img = torch.clamp(denorm(src_fr_img, means, stds), 0, 1)
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
                subplotimg(axs[2][0], vis_fr_img[j], 'Source FR Image')
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
