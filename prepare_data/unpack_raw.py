
import os
import struct
from pathlib import Path
import multiprocessing as mp
import numpy as np
import cv2
import mmap
import argparse

HEADER_SIZE = 64
MAGIC = b"THRW"
INDEX_MAGIC = b"INDX"

PHASE_NAME = {0: "train", 1: "val", 2: "test"}
BASE_DIRS = ["img", "img_comp", "img_comp_geo", "img_comp_tex", "depth", "mask", "parm"]

ID_TO_DTYPE = {
    1: np.uint8,
    2: np.uint16,
    3: np.int32,
    4: np.int64,
    5: np.float32,
    6: np.float64,
}

_G_OUT_ROOT = None
_G_PNG_COMP = 1
_G_DELETE_PACKS = False

def _init_worker(out_root: str, png_comp: int, delete_packs: int):
    global _G_OUT_ROOT, _G_PNG_COMP, _G_DELETE_PACKS
    _G_OUT_ROOT = out_root
    _G_PNG_COMP = int(png_comp)
    _G_DELETE_PACKS = bool(int(delete_packs))

def read_index_from_mmap(mm: mmap.mmap):
    header = mm[:HEADER_SIZE]
    magic, version, res_w, res_h, index_offset, record_count = struct.unpack_from("<4sIIIQQ", header, 0)
    if magic != MAGIC:
        raise ValueError(f"Bad magic: {magic}")

    pos = int(index_offset)
    idx_magic, num = struct.unpack_from("<4sI", mm, pos)
    if idx_magic != INDEX_MAGIC:
        raise ValueError(f"Bad index magic: {idx_magic}")
    pos += 8

    offsets = []
    for _ in range(num):
        off, ln = struct.unpack_from("<QI", mm, pos)
        pos += 12
        offsets.append((int(off), int(ln)))
    return offsets

def _ensure_dirs(base_save_path: str, data_id: str, pid: int):
    suffix = f"{data_id}_{pid:03d}"
    dirs = {d: os.path.join(base_save_path, d, suffix) for d in BASE_DIRS}
    for p in dirs.values():
        Path(p).mkdir(parents=True, exist_ok=True)
    return dirs

def _parse_record_from_mmap(mm: mmap.mmap, offset: int, length: int):
    mv = memoryview(mm)[offset: offset + length]
    pos = 0
    (record_len,) = struct.unpack_from("<I", mv, pos)
    pos += 4

    phase_id = mv[pos]
    pos += 1

    (data_len,) = struct.unpack_from("<H", mv, pos)
    pos += 2

    data_id = bytes(mv[pos:pos + data_len]).decode("utf-8")
    pos += data_len

    (pid,) = struct.unpack_from("<H", mv, pos)
    pos += 2

    vid = int(mv[pos])
    pos += 1
    pos += 1

    metas = []
    for _ in range(8):
        dtype_id, ndim, _rsv, s0, s1, s2, nbytes = struct.unpack_from("<BBH4I", mv, pos)
        pos += 20
        metas.append((int(dtype_id), int(ndim), (int(s0), int(s1), int(s2)), int(nbytes)))

    arrays = []
    for dtype_id, ndim, (s0, s1, s2), nbytes in metas:
        dt = ID_TO_DTYPE.get(dtype_id, None)
        if dt is None:
            raise ValueError(f"Unknown dtype_id: {dtype_id}")
        shape = (s0, s1, s2)[:ndim]
        buf = mv[pos:pos + nbytes]
        pos += nbytes
        arr = np.frombuffer(buf, dtype=dt).copy().reshape(shape)
        arrays.append(arr)

    return int(phase_id), data_id, int(pid), int(vid), arrays

def _process_one_packfile(pack_path: str):
    out_root = _G_OUT_ROOT
    png_comp = _G_PNG_COMP
    delete_packs = _G_DELETE_PACKS

    params = [cv2.IMWRITE_PNG_COMPRESSION, int(png_comp)]

    with open(pack_path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            idx = read_index_from_mmap(mm)
            for off, ln in idx:
                phase_id, data_id, pid, vid, arrays = _parse_record_from_mmap(mm, off, ln)
                phase = PHASE_NAME.get(phase_id, "unknown")
                base_save_path = os.path.join(out_root, phase)
                save_dirs = _ensure_dirs(base_save_path, data_id, pid)

                depth_u16, mask_u8, img_bgr, comp_bgr, comp_geo_bgr, comp_tex_bgr, intr, extr = arrays

                cv2.imwrite(os.path.join(save_dirs["depth"], f"{vid}.png"), depth_u16, params)
                cv2.imwrite(os.path.join(save_dirs["mask"], f"{vid}.png"), mask_u8, params)

                cv2.imwrite(os.path.join(save_dirs["img"], f"{vid}.png"), img_bgr, params)
                cv2.imwrite(os.path.join(save_dirs["img_comp"], f"{vid}.png"), comp_bgr, params)
                cv2.imwrite(os.path.join(save_dirs["img_comp_geo"], f"{vid}.png"), comp_geo_bgr, params)
                cv2.imwrite(os.path.join(save_dirs["img_comp_tex"], f"{vid}.png"), comp_tex_bgr, params)

                np.save(os.path.join(save_dirs["parm"], f"{vid}_intrinsic.npy"), intr)
                np.save(os.path.join(save_dirs["parm"], f"{vid}_extrinsic.npy"), extr)
        finally:
            mm.close()

    if delete_packs:
        try:
            os.remove(pack_path)
        except FileNotFoundError:
            pass

    return 1

def list_packfiles(pack_root: str, phases):
    files = []
    for phase in phases:
        phase_dir = os.path.join(pack_root, phase)
        if not os.path.isdir(phase_dir):
            continue
        for data_id in os.listdir(phase_dir):
            subj_dir = os.path.join(phase_dir, data_id)
            if not os.path.isdir(subj_dir):
                continue
            for name in os.listdir(subj_dir):
                if name.endswith(".thrw"):
                    files.append(os.path.join(subj_dir, name))
    files.sort()
    return files

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack_root", type=str, required=True)
    parser.add_argument("--out_root", type=str, required=True)
    parser.add_argument("--phases", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--png_comp", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=max(1, mp.cpu_count() - 1))
    parser.add_argument("--chunksize", type=int, default=1)
    parser.add_argument("--delete_packs", action="store_true")
    args = parser.parse_args()

    pack_files = list_packfiles(args.pack_root, args.phases)
    print(f"[Found] packfiles={len(pack_files)} (phases={args.phases})")
    if len(pack_files) == 0:
        raise SystemExit(0)

    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=args.num_workers,
        initializer=_init_worker,
        initargs=(args.out_root, args.png_comp, 1 if args.delete_packs else 0),
    ) as pool:
        done = 0
        for _ in pool.imap_unordered(_process_one_packfile, pack_files, chunksize=int(args.chunksize)):
            done += 1
            if done % 200 == 0:
                print(f"[Progress] {done}/{len(pack_files)}")

    print(f"[Done] unpacked into: {args.out_root}")
