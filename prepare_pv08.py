#!/usr/bin/env python3
"""
python3 prepare_pv08.py --source-dir /mnt/data/yanyi2025/zxlzxl/pv-segmentation/data/PV08/ --output-dir ./pv08_prepared --recursive
"""
from __future__ import annotations

import argparse
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


IMAGE_EXTENSIONS = {'.bmp', '.png', '.jpg', '.jpeg', '.tif', '.tiff'}


@dataclass(frozen=True)
class PairItem:
    stem: str
    image_path: Path
    mask_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Prepare PV08-style flat files into RSAM-Seg paired folders.'
    )
    parser.add_argument(
        '--source-dir',
        required=True,
        help='Directory containing files like image1.bmp and image1_label.bmp.',
    )
    parser.add_argument(
        '--output-dir',
        required=True,
        help='Output root directory, e.g. ./data/pv08_prepared.',
    )
    parser.add_argument(
        '--label-suffix',
        default='_label',
        help='Mask filename suffix before extension. Default: _label',
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.2,
        help='Validation ratio in [0, 1). Default: 0.2',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for train/val split. Default: 42',
    )
    parser.add_argument(
        '--recursive',
        action='store_true',
        help='Recursively scan source-dir.',
    )
    parser.add_argument(
        '--copy',
        action='store_true',
        help='Copy files instead of moving them. Default behavior is copy.',
    )
    parser.add_argument(
        '--move',
        action='store_true',
        help='Move files instead of copying them.',
    )
    parser.add_argument(
        '--flat',
        action='store_true',
        help='Do not split train/val; write to output/images and output/masks only.',
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite output directory contents if target files already exist.',
    )
    return parser.parse_args()


def ensure_valid_args(args: argparse.Namespace) -> None:
    if args.copy and args.move:
        raise ValueError('Use only one of --copy or --move.')
    if not 0 <= args.val_ratio < 1:
        raise ValueError('--val-ratio must be in [0, 1).')


def iter_files(source_dir: Path, recursive: bool) -> Iterable[Path]:
    pattern = '**/*' if recursive else '*'
    for path in sorted(source_dir.glob(pattern)):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def collect_pairs(source_dir: Path, label_suffix: str, recursive: bool) -> List[PairItem]:
    pairs: List[PairItem] = []
    for file_path in iter_files(source_dir, recursive):
        suffix = file_path.suffix
        if file_path.stem.endswith(label_suffix):
            continue

        mask_name = f'{file_path.stem}{label_suffix}{suffix}'
        mask_path = file_path.with_name(mask_name)
        if not mask_path.exists():
            continue

        pairs.append(PairItem(stem=file_path.stem, image_path=file_path, mask_path=mask_path))

    pairs.sort(key=lambda item: item.stem)
    return pairs


def split_pairs(pairs: List[PairItem], val_ratio: float, seed: int) -> tuple[List[PairItem], List[PairItem]]:
    if not pairs:
        return [], []

    shuffled = list(pairs)
    random.Random(seed).shuffle(shuffled)

    val_count = int(round(len(shuffled) * val_ratio))
    if val_ratio > 0 and val_count == 0 and len(shuffled) > 1:
        val_count = 1
    if val_count >= len(shuffled):
        val_count = len(shuffled) - 1

    val_pairs = sorted(shuffled[:val_count], key=lambda item: item.stem)
    train_pairs = sorted(shuffled[val_count:], key=lambda item: item.stem)
    return train_pairs, val_pairs


def prepare_dirs(base_dir: Path, overwrite: bool) -> None:
    if base_dir.exists() and any(base_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f'{base_dir} is not empty. Use --overwrite or choose another --output-dir.'
        )
    base_dir.mkdir(parents=True, exist_ok=True)


def write_split(pairs: List[PairItem], split_dir: Path, move_files: bool) -> None:
    images_dir = split_dir / 'images'
    masks_dir = split_dir / 'masks'
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    transfer = shutil.move if move_files else shutil.copy2
    for item in pairs:
        image_dst = images_dir / item.image_path.name
        mask_dst = masks_dir / item.mask_path.name
        transfer(str(item.image_path), str(image_dst))
        transfer(str(item.mask_path), str(mask_dst))


def main() -> None:
    args = parse_args()
    ensure_valid_args(args)

    source_dir = Path(args.source_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not source_dir.exists():
        raise FileNotFoundError(f'source directory does not exist: {source_dir}')
    if not source_dir.is_dir():
        raise NotADirectoryError(f'source directory is not a directory: {source_dir}')

    pairs = collect_pairs(source_dir, args.label_suffix, args.recursive)
    if not pairs:
        raise RuntimeError(
            'No image/mask pairs found. Expected files like image1.bmp and image1_label.bmp.'
        )

    prepare_dirs(output_dir, args.overwrite)
    move_files = args.move and not args.copy

    if args.flat:
        write_split(pairs, output_dir, move_files)
        print(f'Prepared {len(pairs)} pairs into: {output_dir}')
        print(f'Images: {output_dir / "images"}')
        print(f'Masks:  {output_dir / "masks"}')
        return

    train_pairs, val_pairs = split_pairs(pairs, args.val_ratio, args.seed)
    if not train_pairs:
        raise RuntimeError('Training split is empty. Lower --val-ratio or add more samples.')

    write_split(train_pairs, output_dir / 'train', move_files)
    if val_pairs:
        write_split(val_pairs, output_dir / 'val', move_files)

    print(f'Total pairs: {len(pairs)}')
    print(f'Train pairs: {len(train_pairs)} -> {output_dir / "train"}')
    print(f'Val pairs:   {len(val_pairs)} -> {output_dir / "val"}')
    print('Use these config paths in the notebook:')
    print(f'  TRAIN_IMAGE_DIR = {output_dir / "train" / "images"}')
    print(f'  TRAIN_MASK_DIR  = {output_dir / "train" / "masks"}')
    val_images = output_dir / 'val' / 'images'
    val_masks = output_dir / 'val' / 'masks'
    if val_pairs:
        print(f'  VAL_IMAGE_DIR   = {val_images}')
        print(f'  VAL_MASK_DIR    = {val_masks}')
    else:
        print('  VAL_IMAGE_DIR   = <not created; val split is empty>')
        print('  VAL_MASK_DIR    = <not created; val split is empty>')


if __name__ == '__main__':
    main()
