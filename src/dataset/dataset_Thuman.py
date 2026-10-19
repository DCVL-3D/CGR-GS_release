import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from PIL import Image
import numpy as np
import cv2
from torch.utils.data import Dataset

from .dataset import DatasetCfgCommon

@dataclass
class DatasetThumanCfg(DatasetCfgCommon):
    name: Literal["thuman"]
    roots: list[Path]
    split: str = "val"
    ratio: float = 1.0

class DatasetThuman(Dataset):
    cfg: DatasetThumanCfg
    near: float = 1.0
    far: float = 3.0

    def __init__(self, cfg: DatasetThumanCfg) -> None:
        super().__init__()
        self.cfg = cfg

        self.data_root = os.path.join(str(cfg.roots[0]), cfg.split)
        self.img_path   = os.path.join(self.data_root, "img/%s/LR/%s.png")
        self.mask_path  = os.path.join(self.data_root, "mask/%s/LR/%s.png")
        self.depth_path = os.path.join(self.data_root, "depth/%s/LR/%s.png")
        self.intr_path  = os.path.join(self.data_root, "parm/%s/LR/%s_intrinsic.npy")
        self.extr_path  = os.path.join(self.data_root, "parm/%s/LR/%s_extrinsic.npy")

        self.subject_list = sorted(list(os.listdir(os.path.join(self.data_root, 'img'))))
        self.subject_list = sorted(set(s.split('_')[0] for s in self.subject_list))

    def __len__(self) -> int:
        return len(self.subject_list)

    def __getitem__(self, index):
        subject_idx = self.subject_list[index % len(self.subject_list)]

        ref_views = [f"{i:03d}_0" for i in range(0, 16, 4)]
        tgt_views = [f"{i:03d}_3" for i in range(16)]

        ref_data = self.fill_and_stack(subject_idx, ref_views, self.init_container())
        tgt_data = self.fill_and_stack(subject_idx, tgt_views, self.init_container())

        return {
            'context': ref_data,
            'target': tgt_data,
            'scene': subject_idx
        }

    def get_view_tensor(self, subject_idx, view):
        cam, vid = view.split('_')
        img, mask, depth, intr, extr = self.load_single_view(subject_idx, cam, vid)

        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        if mask.ndim == 3 and mask.shape[2] == 3:
            mask = mask[:, :, [0]]
        mask = torch.from_numpy(mask).permute(2, 0, 1).float() / 255.0

        if len(depth.shape) == 2:
            depth = torch.from_numpy(depth).unsqueeze(0).float()
        else:
            depth = torch.from_numpy(depth).permute(2, 0, 1).float()
        depth = 1 / (depth + 1e-8) * mask

        intr = torch.from_numpy(intr).float()

        R = np.array(extr[:3, :3], np.float32)
        T = np.array(extr[:3, 3], np.float32)
        Rt = np.zeros((4, 4))
        Rt[:3, :3] = R
        Rt[:3, 3] = T
        Rt[3, 3] = 1.0
        C2W = np.linalg.inv(Rt)
        extr = torch.from_numpy(C2W).float()

        near = torch.tensor(self.near, dtype=torch.float32)
        far  = torch.tensor(self.far,  dtype=torch.float32)

        return img, mask, depth, intr, extr, near, far

    def init_container(self):
        return {
            'image': [],
            'mask': [],
            'depth': [],
            'intrinsics': [],
            'extrinsics': [],
            'near': [],
            'far': [],
            'index': [],
        }

    def cam_view_to_index(self, view_str):
        cam_str, view_str2 = view_str.split('_')
        cam_id = int(cam_str)
        view_id = int(view_str2)
        view_id = 0 if view_id == 0 else 1
        return cam_id * 2 + view_id

    def fill_and_stack(self, subject_idx, view_list, container):
        for view in view_list:
            img, mask, depth, intr, extr, near, far = self.get_view_tensor(subject_idx, view)

            container['image'].append(img)
            container['mask'].append(mask)
            container['depth'].append(depth)
            container['intrinsics'].append(intr)
            container['extrinsics'].append(extr)
            container['near'].append(near)
            container['far'].append(far)
            container['index'].append(torch.tensor(self.cam_view_to_index(view), dtype=torch.int64))

        for k in container.keys():
            container[k] = torch.stack(container[k], dim=0)

        return container

    def load_single_view(self, subject_idx, cam, vid):
        key = f"{subject_idx}_{cam}"
        intr, extr = np.load(self.intr_path % (key, vid)), np.load(self.extr_path % (key, vid))

        img = np.array(Image.open(self.img_path % (key, vid)))

        intr[:2] *= self.cfg.ratio
        H, W = int(img.shape[0] * self.cfg.ratio), int(img.shape[1] * self.cfg.ratio)
        intr[0] = intr[0] / W
        intr[1] = intr[1] / H

        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)

        mask = np.array(Image.open(self.mask_path % (key, vid)))
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

        depth = np.array(Image.open(self.depth_path % (key, vid))) / (2.0 ** 15)
        depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_AREA)

        return img, mask, depth, intr, extr
