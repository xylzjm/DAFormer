import os.path as osp

import mmcv
import torch


class PrototypeEstimator:
    def __init__(self, cfg, resume=None) -> None:
        self.num_class = cfg['num_class']
        self.feat_dim = cfg['feat_dim']
        self.ignore_idx = cfg['ignore_index']
        self.momentum = cfg['momentum']
        self.enable_momentum = self.momentum > 0

        if resume is not None:
            mmcv.print_log(f'Loading prototype from {resume}', 'mmseg')
            ckpt = torch.load(resume, map_location='cpu')
            self.Proto = ckpt['Proto'].cuda(non_blocking=True)
            self.Amount = ckpt['Amount'].cuda(non_blocking=True)
            assert self.Proto.shape == (self.num_class, self.feat_dim)
        else:
            mmcv.print_log(f'Initial prototype!!!!!!!!!!!!!!!', 'mmseg')
            self.Proto = torch.zeros(self.num_class, self.feat_dim).cuda(
                non_blocking=True
            )
            self.Amount = torch.zeros(self.num_class).cuda(non_blocking=True)

    def update(self, feat, label):
        mask = label != self.ignore_idx
        label = label[mask]
        feat = feat[mask]

        if not self.enable_momentum:
            N, A = feat.shape
            C = self.num_class
            # refer to SDCA for fast implementation
            feat = feat.view(N, 1, A).expand(N, C, A)
            onehot = torch.zeros(N, C).cuda()
            onehot.scatter_(1, label.view(-1, 1), 1)
            NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)
            feat_by_sort = feat.mul(NxCxA_onehot)
            Amount_CXA = NxCxA_onehot.sum(0)
            Amount_CXA[Amount_CXA == 0] = 1
            mean = feat_by_sort.sum(0) / Amount_CXA
            sum_weight = onehot.sum(0).view(C, 1).expand(C, A)
            weight = sum_weight.div(sum_weight + self.Amount.view(C, 1).expand(C, A))
            weight[sum_weight == 0] = 0
            self.Proto = (self.Proto.mul(1 - weight) + mean.mul(weight)).detach()
            self.Amount = self.Amount + onehot.sum(0)
        else:
            # momentum implementation
            ids_unique = label.unique()
            for i in ids_unique:
                i = i.item()
                mask_i = label == i
                feature = feat[mask_i]
                feature = torch.mean(feature, dim=0)
                self.Amount[i] += len(mask_i)
                self.Proto[i, :] = self.momentum * feature + self.Proto[i, :] * (
                    1 - self.momentum
                )

    def save(self, filename):
        filename = osp.join('pretrained/', filename)
        mmcv.print_log(f'save initialized prototype on: {filename}', 'mmseg')
        torch.save(
            {'Proto': self.Proto.cpu(), 'Amount': self.Amount.cpu()},
            filename,
        )
