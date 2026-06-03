#  CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train.py --config configs/cod-sam-vit-b.yaml --name exp_vitb

"""
单卡单种子：
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/cod-sam-vit-b-pvp.yaml --name exp_vitb --seed 42
多卡单种子：
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train.py --config configs/cod-sam-vit-b.yaml --name exp_vitb --seed 42
多种子（会顺序训练）：
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train.py --config configs/cod-sam-vit-b.yaml --name exp_vitb --seeds "42,123,456"

CUDA_VISIBLE_DEVICES=1 python train.py --config configs/cod-sam-vit-b.yaml --name exp_vitb2 --seeds "42,123,456"

"""
import argparse
import os
import random
import copy

import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

import datasets
import models
import utils
from statistics import mean
import torch
import torch.distributed as dist
import numpy as np

local_rank = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed: int, rank: int = 0):
    """Set random seeds for reproducibility. Adds rank offset for DDP."""
    seed = int(seed) + int(rank)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':16:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_seeds_arg(seeds_arg):
    if seeds_arg is None:
        return None
    if isinstance(seeds_arg, (list, tuple)):
        return [int(x) for x in seeds_arg]
    # allow comma/space separated
    parts = []
    for chunk in str(seeds_arg).replace(',', ' ').split():
        if chunk.strip():
            parts.append(int(chunk))
    return parts if parts else None


def setup_distributed(args):
    global local_rank, device

    if not torch.cuda.is_available():
        raise RuntimeError('RSAM-Seg training requires CUDA. Please run on a GPU-enabled environment.')

    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    os.environ.setdefault('MASTER_PORT', '29500')
    os.environ.setdefault('RANK', '0')
    os.environ.setdefault('WORLD_SIZE', '1')

    env_local_rank = os.environ.get('LOCAL_RANK')
    if env_local_rank is not None:
        local_rank = int(env_local_rank)
    elif args.local_rank >= 0:
        local_rank = args.local_rank
        os.environ['LOCAL_RANK'] = str(local_rank)
    else:
        local_rank = 0
        os.environ['LOCAL_RANK'] = '0'

    if not dist.is_initialized():
        dist.init_process_group(backend='nccl', init_method='env://')

    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)


def make_data_loader(spec, tag=''):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    if local_rank == 0:
        log('{} dataset: size={}'.format(tag, len(dataset)))
        for k, v in dataset[0].items():
            log('  {}: shape={}'.format(k, tuple(v.shape)))

    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        shuffle=False, num_workers=8, pin_memory=True, sampler=sampler)
    return loader


def make_data_loaders():
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    val_loader = make_data_loader(config.get('val_dataset'), tag='val')
    return train_loader, val_loader


def eval_psnr(loader, model, eval_type=None):
    model.eval()

    if eval_type == 'f1':
        metric_fn = utils.calc_f1
        metric1, metric2, metric3, metric4 = 'f1', 'auc', 'none', 'none'
    elif eval_type == 'fmeasure':
        metric_fn = utils.calc_fmeasure
        metric1, metric2, metric3, metric4 = 'f_mea', 'mae', 'none', 'none'
    elif eval_type == 'ber':
        metric_fn = utils.calc_ber
        metric1, metric2, metric3, metric4 = 'shadow', 'non_shadow', 'ber', 'none'
    elif eval_type == 'cod':
        metric_fn = utils.calc_cod
        metric1, metric2, metric3, metric4 = 'sm', 'em', 'wfm', 'mae'
    elif eval_type == 'seg':
        metric_fn = utils.calc_prf_iou
        metric1, metric2, metric3, metric4 = 'precision', 'recall', 'f1', 'iou'

    if local_rank == 0:
        pbar = tqdm(total=len(loader), leave=False, desc='val')
    else:
        pbar = None

    sum1 = 0.0
    sum2 = 0.0
    sum3 = 0.0
    sum4 = 0.0
    count = 0.0
    amp_enabled = bool(config.get('amp', False))

    with torch.no_grad():
        for batch in loader:
            for k, v in batch.items():
                batch[k] = v.cuda()

            inp = batch['inp']
            bs = inp.shape[0]
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                pred = torch.sigmoid(model.infer(inp))

            pred = pred.float().cpu()
            gt = batch['gt'].float().cpu()
            model.features = None
            torch.cuda.empty_cache()
            result1, result2, result3, result4 = metric_fn(pred, gt)
            def _to_number(v):
                return v.item() if hasattr(v, 'item') else float(v)

            sum1 += _to_number(result1) * bs
            sum2 += _to_number(result2) * bs
            sum3 += _to_number(result3) * bs
            sum4 += _to_number(result4) * bs
            count += bs

            if pbar is not None:
                pbar.update(1)

    if pbar is not None:
        pbar.close()

    if dist.is_initialized():
        t = torch.tensor([sum1, sum2, sum3, sum4, count], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        sum1, sum2, sum3, sum4, count = t.tolist()

    if count == 0:
        result1 = result2 = result3 = result4 = 0.0
    else:
        result1 = sum1 / count
        result2 = sum2 / count
        result3 = sum3 / count
        result4 = sum4 / count

    return result1, result2, result3, result4, metric1, metric2, metric3, metric4


def prepare_training():
    if config.get('resume') is not None:
        model = models.make(config['model']).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = config.get('resume') + 1
    else:
        model = models.make(config['model']).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = 1
    max_epoch = config.get('epoch_max')
    lr_scheduler = CosineAnnealingLR(optimizer, max_epoch, eta_min=config.get('lr_min'))
    if local_rank == 0:
        log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
    return model, optimizer, epoch_start, lr_scheduler

def train(train_loader, model):
    model.train()

    if local_rank == 0:
        pbar = tqdm(total=len(train_loader), leave=False, desc='train')
    else:
        pbar = None

    loss_list = []
    for batch in train_loader:
        for k, v in batch.items():
            batch[k] = v.to(device)
        inp = batch['inp']
        gt = batch['gt']
        model.set_input(inp, gt)
        model.optimize_parameters()
        batch_loss = [torch.zeros_like(model.loss_G) for _ in range(dist.get_world_size())]
        dist.all_gather(batch_loss, model.loss_G)
        loss_list.extend(batch_loss)
        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    loss = [i.item() for i in loss_list]
    return mean(loss)


def main(config_, save_path, args):
    global config, log, writer, log_info
    config = config_
    log, writer = utils.set_save_path(save_path, remove=False)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    train_loader, val_loader = make_data_loaders()
    if config.get('data_norm') is None:
        config['data_norm'] = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }

    model, optimizer, epoch_start, lr_scheduler = prepare_training()
    model.optimizer = optimizer
    amp_enabled = bool(config.get('amp', False))
    model.use_amp = amp_enabled
    model.scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    lr_scheduler = CosineAnnealingLR(model.optimizer, config['epoch_max'], eta_min=config.get('lr_min'))
    model = model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
        broadcast_buffers=False
    )
    model = model.module

    sam_checkpoint = torch.load(config['sam_checkpoint'], map_location=device)
    model.load_state_dict(sam_checkpoint, strict=False)
    for name, para in model.named_parameters():
        if "image_encoder" in name and "prompt_generator" not in name:
            para.requires_grad_(False)
    if local_rank == 0:
        model_total_params = sum(p.numel() for p in model.parameters())
        model_grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('model_grad_params:' + str(model_grad_params), '\nmodel_total_params:' + str(model_total_params))

    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    # Track best IoU (or BER if eval_type == 'ber')
    max_val_v = -1e18 if config['eval_type'] != 'ber' else 1e8
    timer = utils.Timer()
    for epoch in range(epoch_start, epoch_max + 1):
        train_loader.sampler.set_epoch(epoch)
        t_epoch_start = timer.t()
        train_loss_G = train(train_loader, model)
        lr_scheduler.step()

        if local_rank == 0:
            log_info = ['epoch {}/{}'.format(epoch, epoch_max)]
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
            log_info.append('train G: loss={:.4f}'.format(train_loss_G))
            writer.add_scalars('loss', {'train G': train_loss_G}, epoch)

            model_spec = config['model']
            model_spec['sd'] = model.state_dict()
            optimizer_spec = config['optimizer']
            optimizer_spec['sd'] = optimizer.state_dict()

            save(config, model, save_path, 'last')

        if (epoch_val is not None) and (epoch % epoch_val == 0):
            result1, result2, result3, result4, metric1, metric2, metric3, metric4 = eval_psnr(val_loader, model,
                eval_type=config.get('eval_type'))

            if local_rank == 0:
                log_info.append('val: {}={:.4f}'.format(metric1, result1))
                writer.add_scalars(metric1, {'val': result1}, epoch)
                log_info.append('val: {}={:.4f}'.format(metric2, result2))
                writer.add_scalars(metric2, {'val': result2}, epoch)
                log_info.append('val: {}={:.4f}'.format(metric3, result3))
                writer.add_scalars(metric3, {'val': result3}, epoch)
                log_info.append('val: {}={:.4f}'.format(metric4, result4))
                writer.add_scalars(metric4, {'val': result4}, epoch)

                if config['eval_type'] != 'ber':
                    # Use IoU (metric4/result4) as the "best" criterion abcdegit 
                    if result4 > max_val_v:
                        max_val_v = result4
                        save(config, model, save_path, 'best')
                else:
                    if result3 < max_val_v:
                        max_val_v = result3
                        save(config, model, save_path, 'best')

                t = timer.t()
                prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
                t_epoch = utils.time_text(t - t_epoch_start)
                t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
                log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

                log(', '.join(log_info))
                writer.flush()


def save(config, model, save_path, name):
    if config['model']['name'] == 'segformer' or config['model']['name'] == 'setr':
        if config['model']['args']['encoder_mode']['name'] == 'evp':
            prompt_generator = model.encoder.backbone.prompt_generator.state_dict()
            decode_head = model.encoder.decode_head.state_dict()
            torch.save({"prompt": prompt_generator, "decode_head": decode_head},
                       os.path.join(save_path, f"prompt_epoch_{name}.pth"))
        else:
            torch.save(model.state_dict(), os.path.join(save_path, f"model_epoch_{name}.pth"))
    else:
        torch.save(model.state_dict(), os.path.join(save_path, f"model_epoch_{name}.pth"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/cod-sam-vit-l.yaml')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility.')
    parser.add_argument('--seeds', type=str, default=None,
                        help='Comma/space separated list of seeds. Overrides --seed.')
    parser.add_argument("--local_rank", type=int, default=-1, help="")
    args = parser.parse_args()

    setup_distributed(args)

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        if local_rank == 0:
            print('config loaded.')

    base_save_name = args.name
    if base_save_name is None:
        base_save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        base_save_name += '_' + args.tag

    seeds_list = parse_seeds_arg(args.seeds)
    if seeds_list is None and args.seed is not None:
        seeds_list = [args.seed]
    if seeds_list is None:
        seeds_list = [None]

    try:
        for seed in seeds_list:
            save_name = base_save_name
            if seed is not None and len(seeds_list) > 1:
                save_name = f'{base_save_name}_seed{seed}'
            if seed is not None:
                set_seed(seed, rank=local_rank)
                if local_rank == 0:
                    print(f'Using seed: {seed} (rank {local_rank} offset applied)')
            save_path = os.path.join('./save', save_name)
            main(copy.deepcopy(config), save_path, args=args)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
