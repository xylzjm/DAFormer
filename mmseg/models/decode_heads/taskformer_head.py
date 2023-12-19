import mmcv
import torch
import torch.nn as nn
from functools import partial
from mmcv.cnn import ConvModule
from timm.models.layers import DropPath

from mmseg.ops import resize
from ..backbones.mix_transformer import Mlp, OverlapPatchEmbed
from ..builder import HEADS
from .daformer_head import build_layer
from .decode_head import BaseDecodeHead


class AttentionLayer(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        sr_ratio=1,
    ):
        super(AttentionLayer, self).__init__()
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.task_query = nn.Parameter(torch.randn(1, 128, dim))

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        task_q = self.task_query

        if B > 1:
            task_q = task_q.unsqueeze(0).repeat(B, 1, 1, 1)
            task_q = task_q.squeeze(1)

        q = (
            self.q(task_q)
            .reshape(B, task_q.shape[1], self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = (
                self.kv(x_)
                .reshape(B, -1, 2, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
        else:
            kv = (
                self.kv(x)
                .reshape(B, -1, 2, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
        k, v = kv[0], kv[1]
        q = resize(
            q, size=(N, C // self.num_heads), mode='bicubic', align_corners=False
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class WeatherTaskBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.1,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        sr_ratio=1,
        **kwargs,
    ):
        super(WeatherTaskBlock, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = AttentionLayer(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


@HEADS.register_module()
class TaskFormerHead(BaseDecodeHead):
    def __init__(
        self,
        img_size=224,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.1,
        depth=3,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    ):
        super(TaskFormerHead, self).__init__(
            input_transform='multiple_select', **kwargs
        )
        assert not self.align_corners

        decoder_params = kwargs['decoder_params']
        szalign_cfg = decoder_params['szalign_cfg']
        embed_dims = szalign_cfg['embed_dims']
        if isinstance(embed_dims, int):
            embed_dims = [embed_dims] * len(self.in_index)
        embed_cfg = szalign_cfg['embed_cfg']
        embed_neck_cfg = szalign_cfg['embed_neck_cfg']
        fusion_cfg = decoder_params['fusion_cfg']
        query_cfg = decoder_params['query_cfg']
        query_dim = query_cfg['query_dim']
        for cfg in [embed_cfg, embed_neck_cfg, fusion_cfg, query_cfg]:
            if cfg is not None and 'aspp' in cfg['type']:
                cfg['align_corners'] = self.align_corners

        self.embed_layers = {}
        for i, in_channels, embed_dim in zip(
            self.in_index, self.in_channels, embed_dims
        ):
            if i == self.in_index[-1]:
                self.embed_layers[str(i)] = build_layer(
                    in_channels, embed_dim, **embed_neck_cfg
                )
            else:
                self.embed_layers[str(i)] = build_layer(
                    in_channels, embed_dim, **embed_cfg
                )
        self.embed_layers = nn.ModuleDict(self.embed_layers)

        self.patch_embed = OverlapPatchEmbed(
            img_size=img_size // 4,
            patch_size=3,
            stride=2,
            in_chans=sum(embed_dims),
            embed_dim=query_dim,
        )
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.block = nn.ModuleList(
            [
                WeatherTaskBlock(
                    dim=query_dim,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    **query_cfg,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(query_dim)
        self.proj = ConvModule(
            sum(embed_dims) + query_dim,
            sum(embed_dims),
            1,
            1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )

        self.fuse_layer = build_layer(sum(embed_dims), self.channels, **fusion_cfg)

    def fusion_bottle_feat(self, inputs):
        x = inputs
        n, _, h, w = x[-1].shape

        os_size = x[0].size()[2:]
        _c = {}
        for i in self.in_index:
            _c[i] = self.embed_layers[str(i)](x[i])
            if _c[i].dim() == 3:
                _c[i] = (
                    _c[i]
                    .permute(0, 2, 1)
                    .contiguous()
                    .reshape(n, -1, x[i].shape[2], x[i].shape[3])
                )
            if _c[i].size()[2:] != os_size:
                _c[i] = resize(
                    _c[i],
                    size=os_size,
                    mode='bicubic',
                    align_corners=self.align_corners,
                )
        feat = torch.cat(list(_c.values()), dim=1)

        y, H, W = self.patch_embed(feat)
        # mmcv.print_log(f'type of feat: {type(feat)} and size of feat: {feat.shape}', 'mmseg')
        # mmcv.print_log(f'type of y: {type(y)} and size of y: {y.shape}', 'mmseg')
        for j, blk in enumerate(self.block):
            y = blk(y, H, W)
        y = self.norm(y)
        y = y.reshape(n, H, W, -1).permute(0, 3, 1, 2).contiguous()
        if y.size()[2:] != os_size:
            y = resize(
                y, size=os_size, mode='bicubic', align_corners=self.align_corners
            )
        out = self.proj(torch.cat((feat, y), dim=1))

        return self.fuse_layer(out)
    
    def forward(self, inputs, return_decfeat=False):
        x = self.fusion_bottle_feat(inputs)
        
        if return_decfeat:
            out = {}
            out['feat'] = x
            out['out'] = self.cls_seg(x)
            return out
        else:
            return self.cls_seg(x)
