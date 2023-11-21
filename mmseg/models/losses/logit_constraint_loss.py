import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .cross_entropy_loss import CrossEntropyLoss
from .utils import get_class_weight, weight_reduce_loss


@LOSSES.register_module()
class LogitConstraintLoss(CrossEntropyLoss):
    def __init__(
        self,
        use_sigmoid=False,
        use_mask=False,
        reduction='mean',
        class_weight=None,
        loss_weight=1,
        eps=1e-7,
    ):
        super(LogitConstraintLoss, self).__init__(
            use_sigmoid, use_mask, reduction, class_weight, loss_weight
        )
        self.eps = eps

    def forward(
        self,
        cls_score,
        label,
        weight=None,
        avg_factor=None,
        reduction_override=None,
        **kwargs
    ):
        norms = torch.norm(cls_score, p=2, dim=1, keepdim=True) + self.eps
        normed_logit = torch.div(cls_score, norms)
        loss_cl = super(LogitConstraintLoss, self).forward(
            normed_logit, label, weight, avg_factor, reduction_override, **kwargs
        )
        return loss_cl
