#!/usr/bin/env python3
# scripts/preprocess/prepare_flickr8k.py

"""
Prepare the Flickr8k dataset for VisionCapt:

- Reads either the HF-style token file (Flickr8k.token.txt) OR the CSV captions.txt
- Splits images into train/val/test
- Copies all .jpgs into a flat images/ folder
- Emits captions.csv, train/val/test_captions.csv, and README.md
"""

import os
import sys
import argparse
import logging
import re
import csv
import pandas as pd
import numpy as np
import shutil
from tqdm import tqdm

# — logging setup —
def setup_logging(level: str = "INFO"):
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=numeric
    )

# — args —
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--flickr_path",
        type=str,
        required=True,
        help="Root folder of Flickr8k (contains token file or captions.txt and the images dir)"
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="data/flickr8k_processed",
        help="Where to write processed captions + images/"
    )
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--val_ratio",   type=float, default=0.1)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--log_level",   type=str,   default="INFO")
    return p.parse_args()

# — load captions —
def load_flickr8k_captions(root: str) -> pd.DataFrame:
    """
    Returns DataFrame with columns [image_name, caption].
    Supports:
      - token file lines: 1000268201_693b08cb0e.jpg#0 <tab> text...
      - CSV file: captions.txt with header image,caption
    """
    # find token vs csv
    tok = os.path.join(root, "Flickr8k.token.txt")
    alt1 = os.path.join(root, "Flickr8k_text", "Flickr8k.token.txt")
    csvf = os.path.join(root, "captions.txt")

    if os.path.isfile(tok) or os.path.isfile(alt1):
        fpath = tok if os.path.isfile(tok) else alt1
        logging.info(f"Loading token-file captions from: {fpath}")
        pattern = re.compile(r"^(\S+)\#\d+\s+(.*)$")
        rows = []
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                m = pattern.match(line)
                if not m:
                    continue
                img, cap = m.group(1), m.group(2).strip()
                rows.append((img, cap))
        df = pd.DataFrame(rows, columns=["image_name", "caption"])

    elif os.path.isfile(csvf):
        logging.info(f"Loading CSV captions from: {csvf}")
        df = pd.read_csv(csvf, sep=",", quotechar='"', dtype=str)
        # rename column if needed
        if "image" in df.columns:
            df = df.rename(columns={"image": "image_name"})
        # drop any extra columns
        df = df[["image_name", "caption"]]

    else:
        raise FileNotFoundError(
            "No captions file found. Tried:\n"
            f"  {tok}\n  {alt1}\n  {csvf}"
        )

    logging.info(f"Parsed {len(df)} captions for {df['image_name'].nunique()} images")
    return df

# — find images dir —
def find_image_directory(root: str) -> str:
    """
    Look under root for folders containing .jpg files.
    """
    candidates = [
        os.path.join(root, "Images"),
        os.path.join(root, "images"),
        os.path.join(root, "Flickr8k_Dataset"),
        os.path.join(root, "Flickr8k_Dataset", "Flickr8k_Dataset"),
    ]
    for d in candidates:
        if os.path.isdir(d):
            jpgs = [f for f in os.listdir(d) if f.lower().endswith(".jpg")]
            if jpgs:
                logging.info(f"Found {len(jpgs)} images in: {d}")
                return d
    raise FileNotFoundError(f"No folder of .jpg files found under {root!r}")

# — main —
def main():
    args = parse_args()
    setup_logging(args.log_level)
    np.random.seed(args.seed)

    # 1) load captions
    df = load_flickr8k_captions(args.flickr_path)

    # 2) split images
    images = df["image_name"].unique()
    np.random.shuffle(images)
    n = len(images)
    n_train = int(n * args.train_ratio)
    n_val   = int(n * args.val_ratio)

    train_set = set(images[:n_train])
    val_set   = set(images[n_train:n_train + n_val])

    df["split"] = df["image_name"].map(
        lambda im: "train" if im in train_set else ("val" if im in val_set else "test")
    )

    # 3) write CSVs
    os.makedirs(args.output_dir, exist_ok=True)
    df.to_csv(
        os.path.join(args.output_dir, "captions.csv"),
        index=False,
        quoting=csv.QUOTE_MINIMAL
    )
    for split in ["train", "val", "test"]:
        sub = df[df["split"] == split]
        sub.to_csv(
            os.path.join(args.output_dir, f"{split}_captions.csv"),
            index=False,
            quoting=csv.QUOTE_MINIMAL
        )
    logging.info("Wrote captions CSVs")

    # 4) copy images
    src_dir = find_image_directory(args.flickr_path)
    dst_dir = os.path.join(args.output_dir, "images")
    os.makedirs(dst_dir, exist_ok=True)

    logging.info(f"Copying images to {dst_dir}")
    copied = 0
    for im in tqdm(images, desc="Copying"):
        s = os.path.join(src_dir, im)
        d = os.path.join(dst_dir, im)
        if os.path.isfile(s):
            shutil.copy2(s, d)
            copied += 1
    logging.info(f"Copied {copied}/{len(images)} images")

    # 5) README
    with open(os.path.join(args.output_dir, "README.md"), "w") as f:
        f.write("# Flickr8k Processed Dataset\n\n")
        f.write(f"- total images: {len(images)}\n")
        f.write(f"- total captions: {len(df)}\n\n")
        for sp in ["train", "val", "test"]:
            imgs = df[df["split"] == sp]["image_name"].nunique()
            caps = len(df[df["split"] == sp])
            f.write(f"- {sp}: {imgs} images, {caps} captions\n")
        f.write(f"\nProcessed on: {pd.Timestamp.now()}\n")

    logging.info("Finished preparing Flickr8k")

if __name__ == "__main__":
    main()
