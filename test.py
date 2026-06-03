"""
CUDA_VISIBLE_DEVICES=0 python test.py --config configs/cod-sam-vit-b-pvp.yaml --model save/exp_vitb_pvp/model_epoch_best.pth

python test.py --config configs/cod-sam-vit-b-pvp.yaml --model save/exp_vitb_seed123/model_epoch_best.pth --save-dir save/exp_vitb_seed123/vis_test --max-save 20
"""
import argparse
import csv
import os

import numpy as np
import yaml
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import models
import utils


def batched_predict(model, inp, coord, bsize):
    with torch.no_grad():
        model.gen_feat(inp)
        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = model.query_rgb(coord[:, ql: qr, :])
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)
    return pred, preds


if hasattr(Image, 'Resampling'):
    BILINEAR = Image.Resampling.BILINEAR
    NEAREST = Image.Resampling.NEAREST
else:
    BILINEAR = Image.BILINEAR
    NEAREST = Image.NEAREST


def prepare_save_dirs(save_dir, save_mask, save_overlay, save_panel):
    os.makedirs(save_dir, exist_ok=True)
    save_dirs = {}
    save_dirs['gt'] = os.path.join(save_dir, 'gt')
    os.makedirs(save_dirs['gt'], exist_ok=True)
    if save_mask:
        save_dirs['mask'] = os.path.join(save_dir, 'pred_mask')
        os.makedirs(save_dirs['mask'], exist_ok=True)
    if save_overlay:
        save_dirs['overlay'] = os.path.join(save_dir, 'overlay')
        os.makedirs(save_dirs['overlay'], exist_ok=True)
    if save_panel:
        save_dirs['panel'] = os.path.join(save_dir, 'panel')
        os.makedirs(save_dirs['panel'], exist_ok=True)
    return save_dirs


def open_metrics_writer(save_dir):
    csv_path = os.path.join(save_dir, 'metrics.csv')
    csv_file = open(csv_path, 'w', newline='')
    writer = csv.DictWriter(
        csv_file,
        fieldnames=['name', 'precision', 'recall', 'f1', 'iou', 'img_path', 'gt_path']
    )
    writer.writeheader()
    return csv_file, writer


def load_rgb_image(path):
    with Image.open(path) as img:
        return img.convert('RGB')


def load_binary_mask(path, size=None):
    with Image.open(path) as img:
        mask = img.convert('L')
        if size is not None and mask.size != size:
            mask = mask.resize(size, resample=NEAREST)
        mask = np.asarray(mask, dtype=np.uint8)
    return np.where(mask >= 128, 255, 0).astype(np.uint8)


def resize_prediction(pred_prob, size):
    pred_img = Image.fromarray(np.clip(pred_prob * 255.0, 0, 255).astype(np.uint8), mode='L')
    pred_img = pred_img.resize(size, resample=BILINEAR)
    return np.asarray(pred_img, dtype=np.float32) / 255.0


def save_binary_mask(mask, out_path):
    Image.fromarray(mask, mode='L').save(out_path)


def mask_to_rgb(mask):
    return np.repeat(mask[:, :, None], 3, axis=2)


def make_overlay(image_rgb, pred_mask, color=(255, 0, 0), alpha=0.45):
    overlay = image_rgb.astype(np.float32).copy()
    mask = pred_mask > 0
    if np.any(mask):
        color_arr = np.asarray(color, dtype=np.float32)
        overlay[mask] = overlay[mask] * (1.0 - alpha) + color_arr * alpha
    return np.clip(overlay, 0, 255).astype(np.uint8)


def _text_size(draw, text):
    if hasattr(draw, 'textbbox'):
        left, top, right, bottom = draw.textbbox((0, 0), text)
        return right - left, bottom - top
    return draw.textsize(text)


def make_panel(image_rgb, gt_mask, pred_mask, overlay_rgb, header_height=24):
    tiles = [
        Image.fromarray(image_rgb, mode='RGB'),
        Image.fromarray(mask_to_rgb(gt_mask), mode='RGB'),
        Image.fromarray(mask_to_rgb(pred_mask), mode='RGB'),
        Image.fromarray(overlay_rgb, mode='RGB')
    ]
    labels = ['image', 'gt', 'pred', 'overlay']
    tile_width, tile_height = tiles[0].size
    panel = Image.new('RGB', (tile_width * len(tiles), tile_height + header_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(panel)

    for idx, (tile, label) in enumerate(zip(tiles, labels)):
        offset_x = idx * tile_width
        panel.paste(tile, (offset_x, header_height))
        text_w, text_h = _text_size(draw, label)
        text_x = offset_x + max((tile_width - text_w) // 2, 0)
        text_y = max((header_height - text_h) // 2, 0)
        draw.text((text_x, text_y), label, fill=(0, 0, 0))

    return panel


def export_sample_visualizations(name, img_path, gt_path, pred_prob, save_dirs, threshold, overlay_alpha):
    image = load_rgb_image(img_path)
    image_rgb = np.asarray(image, dtype=np.uint8)
    gt_mask = load_binary_mask(gt_path, size=image.size)
    pred_prob_orig = resize_prediction(pred_prob, image.size)
    pred_mask = np.where(pred_prob_orig >= threshold, 255, 0).astype(np.uint8)

    save_binary_mask(gt_mask, os.path.join(save_dirs['gt'], f'{name}.png'))
    if 'mask' in save_dirs:
        save_binary_mask(pred_mask, os.path.join(save_dirs['mask'], f'{name}.png'))

    overlay_rgb = None
    if 'overlay' in save_dirs or 'panel' in save_dirs:
        overlay_rgb = make_overlay(image_rgb, pred_mask, alpha=overlay_alpha)

    if 'overlay' in save_dirs:
        Image.fromarray(overlay_rgb, mode='RGB').save(os.path.join(save_dirs['overlay'], f'{name}.png'))

    if 'panel' in save_dirs:
        # gt_mask = load_binary_mask(gt_path, size=image.size)
        panel = make_panel(image_rgb, gt_mask, pred_mask, overlay_rgb)
        panel.save(os.path.join(save_dirs['panel'], f'{name}.png'))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval_psnr(loader, model, data_norm=None, eval_type=None, eval_bsize=None,
              verbose=False, save_dir=None, save_mask=False, save_overlay=False,
              save_panel=False, threshold=0.5, max_save=None, overlay_alpha=0.45):
    model.eval()
    if data_norm is None:
        data_norm = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }

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

    val_metric1 = utils.Averager()
    val_metric2 = utils.Averager()
    val_metric3 = utils.Averager()
    val_metric4 = utils.Averager()
    save_requested = save_dir is not None
    saved_count = 0
    save_dirs = {}
    csv_file = None
    csv_writer = None
    if save_requested:
        save_dirs = prepare_save_dirs(save_dir, save_mask, save_overlay, save_panel)
        csv_file, csv_writer = open_metrics_writer(save_dir)

    pbar = tqdm(loader, leave=False, desc='val')
    amp_enabled = bool(config.get('amp', False))

    try:
        for batch in pbar:
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(device)

            inp = batch['inp']

            with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp_enabled):
                pred = torch.sigmoid(model.infer(inp))

            pred = pred.float().cpu()
            gt = batch['gt'].float().cpu()
            model.features = None
            torch.cuda.empty_cache()
            result1, result2, result3, result4 = metric_fn(pred, gt)

            def _to_number(v):
                return v.item() if hasattr(v, 'item') else float(v)

            bs = inp.shape[0]
            val_metric1.add(_to_number(result1), bs)
            val_metric2.add(_to_number(result2), bs)
            val_metric3.add(_to_number(result3), bs)
            val_metric4.add(_to_number(result4), bs)

            if save_requested and (max_save is None or saved_count < max_save):
                for idx in range(bs):
                    if max_save is not None and saved_count >= max_save:
                        break
                    precision, recall, f1, iou = utils.calc_prf_iou(
                        pred[idx: idx + 1], gt[idx: idx + 1], threshold=threshold
                    )
                    name = batch['name'][idx]
                    img_path = batch['img_path'][idx]
                    gt_path = batch['gt_path'][idx]
                    export_sample_visualizations(
                        name=name,
                        img_path=img_path,
                        gt_path=gt_path,
                        pred_prob=pred[idx, 0].numpy(),
                        save_dirs=save_dirs,
                        threshold=threshold,
                        overlay_alpha=overlay_alpha
                    )
                    csv_writer.writerow({
                        'name': name,
                        'precision': float(precision),
                        'recall': float(recall),
                        'f1': float(f1),
                        'iou': float(iou),
                        'img_path': img_path,
                        'gt_path': gt_path
                    })
                    saved_count += 1

            if verbose:
                pbar.set_description('val {} {:.4f}'.format(metric1, val_metric1.item()))
                pbar.set_description('val {} {:.4f}'.format(metric2, val_metric2.item()))
                pbar.set_description('val {} {:.4f}'.format(metric3, val_metric3.item()))
                pbar.set_description('val {} {:.4f}'.format(metric4, val_metric4.item()))
    finally:
        if csv_file is not None:
            csv_file.close()

    return val_metric1.item(), val_metric2.item(), val_metric3.item(), val_metric4.item(), saved_count


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--model')
    parser.add_argument('--prompt', default='none')
    parser.add_argument('--save-dir', default=None)
    parser.add_argument('--save-mask', action='store_true')
    parser.add_argument('--save-overlay', action='store_true')
    parser.add_argument('--save-panel', action='store_true')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--max-save', type=int, default=None)
    parser.add_argument('--overlay-alpha', type=float, default=0.45)
    args = parser.parse_args()

    if any([args.save_mask, args.save_overlay, args.save_panel]) and args.save_dir is None:
        parser.error('--save-dir is required when using visualization export flags.')
    if args.save_dir is not None and not any([args.save_mask, args.save_overlay, args.save_panel]):
        args.save_mask = True
        args.save_overlay = True
        args.save_panel = True
    if not 0.0 <= args.threshold <= 1.0:
        parser.error('--threshold must be between 0 and 1.')
    if args.max_save is not None and args.max_save < 0:
        parser.error('--max-save must be non-negative.')
    if not 0.0 <= args.overlay_alpha <= 1.0:
        parser.error('--overlay-alpha must be between 0 and 1.')

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    spec = config['test_dataset']
    dataset = datasets.make(spec['dataset'])
    wrapper_args = {'dataset': dataset}
    if args.save_dir is not None:
        wrapper_args['return_meta'] = True
    dataset = datasets.make(spec['wrapper'], args=wrapper_args)
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
                        num_workers=0)

    model = models.make(config['model']).to(device)
    sam_checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(sam_checkpoint, strict=True)
    
    metric1, metric2, metric3, metric4, saved_count = eval_psnr(
        loader, model,
        data_norm=config.get('data_norm'),
        eval_type=config.get('eval_type'),
        eval_bsize=config.get('eval_bsize'),
        verbose=True,
        save_dir=args.save_dir,
        save_mask=args.save_mask,
        save_overlay=args.save_overlay,
        save_panel=args.save_panel,
        threshold=args.threshold,
        max_save=args.max_save,
        overlay_alpha=args.overlay_alpha
    )
    print('metric1: {:.4f}'.format(metric1))
    print('metric2: {:.4f}'.format(metric2))
    print('metric3: {:.4f}'.format(metric3))
    print('metric4: {:.4f}'.format(metric4))
    if args.save_dir is not None:
        print('saved_count: {}'.format(saved_count))
        print('saved_dir: {}'.format(args.save_dir))
