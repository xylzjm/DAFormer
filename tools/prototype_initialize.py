import argparse
import mmcv
import torch
from mmcv.parallel import MMDataParallel, scatter
from mmcv.runner import load_checkpoint

from mmseg.apis import set_random_seed
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models.builder import build_segmentor
from mmseg.models.utils.prototype_estimator import PrototypeEstimator
from mmseg.ops import resize


def proto_init(model, data_loader, device, cfg):
    mmcv.print_log(
        f'---------------- Initialize prototype ----------------', 'mmseg'
    )
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    feat_estimator = PrototypeEstimator(cfg.proto, resume=None)
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            data = scatter(data, device)[0]
            src_img, src_label = data['img'], data['gt_semantic_seg']

            src_feat = model.module.encode_decode(src_img, None)
            B, A, Hs, Ws = src_feat.shape

            # source mask: downsample the ground-truth label
            src_mask = (
                src_label.squeeze(0)
                .long()
                .contiguous()
                .view(
                    B * Hs * Ws,
                )
            )

            src_feat = src_feat.permute(0, 2, 3, 1).contiguous().view(B * Hs * Ws, A)
            feat_estimator.update(feat=src_feat.detach().clone(), label=src_mask)

            for _ in range(B):
                prog_bar.update()
        mmcv.print_log('')
        feat_estimator.save('prototype_source.pth')


def main():
    parser = argparse.ArgumentParser(
        description="Calculate Target Prototype and initialize"
    )
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--gpu-id', type=int, default=0, help='id of gpu to use')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    args = parser.parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # set random seeds
    if args.seed is None and 'seed' in cfg:
        args.seed = cfg['seed']
    if args.seed is not None:
        mmcv.print_log(f'Set random seed to {args.seed}', 'mmseg')
        set_random_seed(args.seed, deterministic=False)
    cfg.seed = args.seed
    # set gup id
    if args.gpu_id is not None:
        cfg.gpu_ids = [args.gpu_id]

    # build the dataloader
    data_cfg = cfg.data.train
    mmcv.print_log(f'data_pipeline: {data_cfg["pipeline"]}', 'mmseg')
    dataset = build_dataset(data_cfg)
    data_loader = build_dataloader(
        dataset,
        cfg.data.samples_per_gpu,
        cfg.data.workers_per_gpu,
        dist=False,
        seed=cfg.seed,
        drop_last=False,
        shuffle=False,
    )

    # build the model and load checkpoint
    cfg.model.pretrained = None
    model = build_segmentor(
        cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg')
    )
    load_checkpoint(
        model,
        args.checkpoint,
        map_location='cpu',
        revise_keys=[(r'^module\.', ''), ('model.', '')],
    )
    model = MMDataParallel(model, device_ids=cfg.gpu_ids)

    proto_init(model, data_loader, cfg.gpu_ids, cfg)


if __name__ == '__main__':
    main()
