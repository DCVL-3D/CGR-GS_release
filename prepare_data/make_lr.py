import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
from tqdm import tqdm

IMAGE_DIRS = ["img", "img_comp", "img_comp_geo", "img_comp_tex"]
NEAREST_DIRS = ["mask", "depth"]

def process_view_dir(root, phase, key, factor):
    n = 0
    for d in IMAGE_DIRS + NEAREST_DIRS:
        src_dir = os.path.join(root, phase, d, key)
        if not os.path.isdir(src_dir):
            continue
        dst_dir = os.path.join(src_dir, "LR")
        os.makedirs(dst_dir, exist_ok=True)
        for name in os.listdir(src_dir):
            if not name.endswith(".png"):
                continue
            dst = os.path.join(dst_dir, name)
            if os.path.exists(dst):
                continue
            img = cv2.imread(os.path.join(src_dir, name), cv2.IMREAD_UNCHANGED)
            h, w = img.shape[:2]
            size = (w // factor, h // factor)
            interp = cv2.INTER_AREA if d in IMAGE_DIRS else cv2.INTER_NEAREST
            cv2.imwrite(dst, cv2.resize(img, size, interpolation=interp), [cv2.IMWRITE_PNG_COMPRESSION, 1])
            n += 1
    src_dir = os.path.join(root, phase, "parm", key)
    if os.path.isdir(src_dir):
        dst_dir = os.path.join(src_dir, "LR")
        os.makedirs(dst_dir, exist_ok=True)
        for name in os.listdir(src_dir):
            if not name.endswith(".npy"):
                continue
            dst = os.path.join(dst_dir, name)
            if os.path.exists(dst):
                continue
            arr = np.load(os.path.join(src_dir, name))
            if name.endswith("_intrinsic.npy"):
                arr = arr.astype(np.float64).copy()
                arr[:2] /= factor
            np.save(dst, arr)
            n += 1
    return n

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="data/thuman2.0_rendered")
    parser.add_argument("--phases", nargs="+", default=["train", "val"])
    parser.add_argument("--factor", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=max(1, os.cpu_count() - 1))
    args = parser.parse_args()

    jobs = []
    for phase in args.phases:
        img_dir = os.path.join(args.root, phase, "img")
        if not os.path.isdir(img_dir):
            continue
        for key in sorted(os.listdir(img_dir)):
            jobs.append((phase, key))

    total = 0
    with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
        futures = [ex.submit(process_view_dir, args.root, phase, key, args.factor) for phase, key in jobs]
        for f in tqdm(as_completed(futures), total=len(futures)):
            total += f.result()
    print(f"wrote {total} files")
