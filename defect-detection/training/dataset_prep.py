"""
Organise raw captured images into labelled train/val split folders.

Usage:
    python training/dataset_prep.py --raw data/raw --out data/annotated --label OK
"""

import argparse
import os
import shutil
import random


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", required=True, help="Folder of raw images")
    p.add_argument("--out", default="data/annotated", help="Output root folder")
    p.add_argument("--label", required=True, choices=["OK", "DEFECT"],
                   help="Class label for these images")
    p.add_argument("--split", type=float, default=0.8, help="Train fraction (default 0.8)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    images = [f for f in os.listdir(args.raw) if os.path.splitext(f)[1].lower() in exts]
    random.shuffle(images)

    split_idx = int(len(images) * args.split)
    splits = {"train": images[:split_idx], "val": images[split_idx:]}

    for subset, files in splits.items():
        dest = os.path.join(args.out, subset, args.label)
        os.makedirs(dest, exist_ok=True)
        for fn in files:
            shutil.copy(os.path.join(args.raw, fn), os.path.join(dest, fn))
        print(f"[DataPrep] {subset}/{args.label}: {len(files)} images → {dest}")


if __name__ == "__main__":
    main()
