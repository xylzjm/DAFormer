import os
import random
import re

import matplotlib.pyplot as plt
import mmcv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from mmseg.models.utils.prototype_estimator import PrototypeEstimator
from mmseg.models.utils.visualization import subplotimg
from mmseg.ops import resize
from timm.models.layers import DropPath
from torch.nn.modules.dropout import _DropoutNd


@UDA.register_module()
class Prog_VECR(VECR):
    def __init__(self, **cfg) -> None:
        super(Prog_VECR).__init__(**cfg)
        self.ignore_index = 255
        self.proto_cfg = cfg['proto']
        self.proto_resume = cfg['proto_resume']

        assert self.num_classes == self.proto_cfg['num_class']
        assert self.ignore_index == self.proto_cfg['ignore_index']
        self.proto_estimator = None

        assert cfg['invariant'].get('source', None)
        assert cfg['invariant'].get('target', None)
        assert isinstance(self.inv_cfg['source']['ce'], (list, tuple))
        assert isinstance(self.inv_cfg['target']['ce'], (list, tuple))
        self.inv_cfg = cfg['invariant']
        self.src_invlam = self.inv_cfg['inv_loss']['weight']
        self.tgt_invlam = self.inv_cfg['inv_loss']['weight']
        mmcv.print_log(
            f'src_invlam: {self.src_invlam}, tgt_invlam: {self.tgt_invlam}', 'mmcv'
        )

    def feat_invariance_loss(self, f1, f2, proto, label):
        assert f1.shape == f2.shape
        b, a, h, w = f1.shape
        feat = (f1 + f2).permute(0, 2, 3, 1).contiguous().view(b * h * w, a)
        label = label.contiguous().view(
            b * h * w,
        )

        mask = label != self.ignore_index
        label = label[mask]
        feat = feat[mask]

        feat = F.normalize(feat, p=2, dim=1)
        proto = F.normalize(proto, p=2, dim=1)
        logits = feat @ proto.permute(1, 0).contiguous()
        logits = logits / 50.0

        ce_criterion = nn.CrossEntropyLoss()
        loss = ce_criterion(logits, label)

        return loss

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
            self.proto_estimator = PrototypeEstimator(
                self.proto_cfg, resume=self.proto_resume
            )
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
            )
            src_fr_img[i] = fourier_transform(
                data=torch.stack((img[i], tgt_ib_img[i])),
                mean=means[0].unsqueeze(0),
                std=stds[0].unsqueeze(0),
            )
        tgt_fr_img, src_fr_img = torch.cat(tgt_fr_img), torch.cat(src_fr_img)

        # train student with source
        src_invflag = self.inv_cfg['source']['consist'] is not None
        src_featpool = {}
        for src_args in self.inv_cfg['source']['ce']:
            assert isinstance(src_args, str)
            if src_args == 'original':
                src_losses = self.get_model().forward_train(
                    img, img_metas, gt_semantic_seg, return_decfeat=src_invflag
                )
                if src_invflag and src_args in self.inv_cfg['source']['consist']:
                    src_featpool[src_args] = src_losses.pop('dec_feat')
                assert 'dec_feat' not in src_losses
                src_loss, src_log = self._parse_losses(src_losses)
                log_vars.update(add_prefix(src_log, 'src_ori'))
                src_loss.backward()
            elif src_args == 'fourier':
                src_losses = self.get_model().forward_train(
                    src_fr_img,
                    img_metas,
                    gt_semantic_seg,
                    return_decfeat=src_invflag,
                )
                if src_invflag and src_args in self.inv_cfg['source']['consist']:
                    src_featpool[src_args] = src_losses.pop('dec_feat')
                assert 'dec_feat' not in src_losses
                src_loss, src_log = self._parse_losses(src_losses)
                log_vars.update(add_prefix(src_log, 'src_for'))
                src_loss.backward()
            else:
                raise ValueError(f'{src_args} not allowed in source CE arguments')
        # source domain feature invariance loss
        if src_invflag:
            for inv_args in self.inv_cfg['source']['consist']:
                assert isinstance(inv_args, str)
                if inv_args == 'original' and inv_args not in src_featpool:
                    src_featpool[inv_args] = self.get_model().extract_decfeat(img)
                elif inv_args == 'fourier' and inv_args not in src_featpool:
                    src_featpool[inv_args] = self.get_model().extract_decfeat(
                        src_fr_img
                    )
                else:
                    raise ValueError(
                        f'{inv_args} not allowed in source Consist arguments'
                    )
            assert len(src_featpool) == len(self.inv_cfg['source']['consist'])
            src_invloss, src_invlog = self.feat_invariance_loss(
                src_featpool[self.inv_cfg['source']['consist'][0]],
                src_featpool[self.inv_cfg['source']['consist'][1]],
                proto=self.proto_estimator.Proto.detach(),
                label=gt_semantic_seg,
            )
            log_vars.update(add_prefix(src_invlog, 'src'))
            src_invloss.backward()

        # generate pseudo-label
        with torch.no_grad():
            ema_tgt_logits = self.get_ema_model().encode_decode(
                tgt_ib_img, target_img_metas
            )
            ema_tgt_softmax = torch.softmax(ema_tgt_logits, dim=1)
            pseudo_prob, pseudo_lbl = torch.max(ema_tgt_softmax, dim=1)
            # estimate pseudo-weight
            pseudo_msk = pseudo_prob.ge(self.pseudo_threshold).long() == 1
            pseudo_size = np.size(np.array(pseudo_msk.cpu()))
            pseudo_weight = torch.sum(pseudo_msk.item()) / pseudo_size
            pseudo_weight = pseudo_weight * torch.ones(pseudo_lbl.shape, device=dev)
            # get gt pixel-weight
            gt_pixel_weight = torch.ones(pseudo_weight.shape, device=dev)

        # prepare target train data
        tgt_semantic_seg = pseudo_lbl.clone().unsqueeze(1)
        tgt_semantic_seg[pseudo_weight == 0.0] = self.ignore_index
        mix_msks = get_class_masks(gt_semantic_seg)
        mixed_img, mixed_fr_img, mixed_lbl = (
            [None] * batch_size,
            [None] * batch_size,
            [None] * batch_size,
        )
        for i in range(batch_size):
            strong_parameters['mix'] = mix_msks[i]
            mixed_img[i], mixed_lbl[i] = strong_transform(
                data=torch.stack((img[i], target_img[i])),
                target=torch.stack((gt_semantic_seg[i][0], pseudo_lbl[i])),
            )
            mixed_fr_img[i], pseudo_weight[i] = strong_transform(
                data=torch.stack((img[i], tgt_fr_img[i])),
                target=torch.stack((gt_pixel_weight[i], pseudo_weight[i])),
            )
        mixed_img, mixed_fr_img, mixed_lbl = (
            torch.cat(mixed_img),
            torch.cat(mixed_fr_img),
            torch.cat(mixed_lbl),
        )

        # update feature statistics
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
                .view(
                    b * h * w,
                )
            )
            tgt_mask = (
                resize(tgt_semantic_seg.float(), size=(h, w), mode='nearest')
                .long()
                .contiguous()
                .view(
                    b * h * w,
                )
            )
            self.proto_estimator.update(feat=src_emafeat, label=src_mask)
            self.proto_estimator.update(feat=tgt_emafeat, label=tgt_mask)

        # train student with target
        tgt_invflag = self.inv_cfg['target']['consist'] is not None
        tgt_featpool = {}
        for tgt_args in self.inv_cfg['target']['ce']:
            assert isinstance(tgt_args, (list, tuple))
            if tgt_args == ('original', 'original'):
                mix_losses = self.get_model().forward_train(
                    mixed_img,
                    img_metas,
                    mixed_lbl,
                    pseudo_weight,
                    return_decfeat=src_invflag,
                )
                if tgt_invflag and tgt_args in self.inv_cfg['target']['consist']:
                    tgt_featpool[tgt_args] = mix_losses.pop('dec_feat')
                assert 'dec_feat' not in mix_losses
                mix_loss, mix_log = self._parse_losses(mix_losses)
                log_vars.update(add_prefix(mix_log, 'mix_ori'))
                mix_loss.backward()
            elif tgt_args == ('fourier', 'fourier'):
                mixfr_losses = self.get_model().forward_train(
                    mixed_fr_img,
                    img_metas,
                    mixed_lbl,
                    pseudo_weight,
                    return_decfeat=src_invflag,
                )
                if tgt_invflag and tgt_args in self.inv_cfg['target']['consist']:
                    tgt_featpool[tgt_args] = mixfr_losses.pop('dec_feat')
                assert 'dec_feat' not in mixfr_losses
                mixfr_loss, mixfr_log = self._parse_losses(mixfr_losses)
                log_vars.update(add_prefix(mixfr_log, 'mix_for'))
                mixfr_loss.backward()
            else:
                raise ValueError(f'{tgt_args} not allowed in target CE arguments')
        # target domain feature invariance loss
        if tgt_invflag:
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
                else:
                    raise ValueError(
                        f'{inv_args} not allowed in target Consist arguments'
                    )
                assert len(tgt_featpool) == len(self.inv_cfg['target']['consist'])
                tgt_invloss, tgt_invlog = self.feat_invariance_loss(
                    tgt_featpool[self.inv_cfg['target']['consist'][0]],
                    tgt_featpool[self.inv_cfg['target']['consist'][1]],
                    proto=self.proto_estimator.Proto.detach(),
                    label=tgt_semantic_seg,
                )
                log_vars.update(add_prefix(tgt_invlog, 'tgt'))
                tgt_invloss.backward()
