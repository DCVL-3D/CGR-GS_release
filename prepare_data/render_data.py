import taichi as ti
import taichi_three as t3
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os
import cv2
import pickle
import math
import struct
import argparse
import zlib
import threading
import queue

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

io_queue = queue.Queue(maxsize=30)

def io_worker():
    while True:
        task = io_queue.get()
        if task is None:
            break

        action, writer, kwargs = task
        if action == "write":
            writer.append_record_raw(**kwargs)
        elif action == "close":
            writer.close()

        io_queue.task_done()

def load_tex(p, default_res=1024):
    if not os.path.exists(p):
        t = np.full((default_res, default_res, 3), 128, dtype=np.uint8)
    else:
        t = cv2.imread(p)
        if t is None:
            t = np.full((default_res, default_res, 3), 128, dtype=np.uint8)
    t = cv2.cvtColor(t, cv2.COLOR_BGR2RGB)
    t = np.ascontiguousarray(t)
    return t.swapaxes(0, 1)[:, ::-1, :]

class THRawPackWriter:
    MAGIC = b"THRW"
    VERSION = 1
    HEADER_SIZE = 64
    INDEX_MAGIC = b"INDX"

    DTYPE_TO_ID = {
        np.dtype(np.uint8): 1,
        np.dtype(np.uint16): 2,
        np.dtype(np.int32): 3,
        np.dtype(np.int64): 4,
        np.dtype(np.float32): 5,
        np.dtype(np.float64): 6,
    }

    def __init__(self, out_path: str, res: tuple[int, int]):
        self.out_path = out_path
        self.res = (int(res[0]), int(res[1]))
        Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
        self.f = open(out_path, "wb", buffering=1024 * 1024 * 32)
        self.index = []
        self._write_header_placeholder()

    def _write_header_placeholder(self):
        res_w, res_h = self.res
        header = struct.pack(
            "<4sIIIQQ",
            self.MAGIC,
            self.VERSION,
            res_w,
            res_h,
            0,
            0,
        )
        header += b"\x00" * (self.HEADER_SIZE - len(header))
        self.f.write(header)

    @staticmethod
    def _as_contig(arr: np.ndarray) -> np.ndarray:
        return arr if arr.flags["C_CONTIGUOUS"] else np.ascontiguousarray(arr)

    def _meta20(self, arr: np.ndarray) -> bytes:
        arr = self._as_contig(arr)
        dt = arr.dtype
        if dt not in self.DTYPE_TO_ID:
            raise ValueError(f"Unsupported dtype: {dt}")
        dtype_id = self.DTYPE_TO_ID[dt]
        ndim = arr.ndim
        if ndim > 3:
            raise ValueError(f"ndim>3 not supported (got {ndim})")

        s = list(arr.shape) + [0, 0, 0]
        s0, s1, s2 = int(s[0]), int(s[1]), int(s[2])
        nbytes = int(arr.nbytes)
        return struct.pack("<BBH4I", dtype_id, ndim, 0, s0, s1, s2, nbytes)

    def append_record_raw(
        self,
        phase_id: int,
        data_id: str,
        pid: int,
        vid: int,
        extr: np.ndarray,
        intr: np.ndarray,
        depth: np.ndarray,
        img: np.ndarray,
        mask: np.ndarray,
        img_comp: np.ndarray,
        img_comp_geo: np.ndarray,
        img_comp_tex: np.ndarray,
    ):

        depth_u16 = (depth * (2.0 ** 15)).astype(np.uint16)
        mask_u8 = (np.clip(mask, 0, 1) * 255.0 + 0.5).astype(np.uint8)

        def to_u8_bgr(x: np.ndarray) -> np.ndarray:
            u8 = (np.clip(x, 0, 1) * 255.0 + 0.5).astype(np.uint8)
            return u8[:, :, ::-1]

        img_bgr = to_u8_bgr(img)
        comp_bgr = to_u8_bgr(img_comp)
        comp_geo_bgr = to_u8_bgr(img_comp_geo)
        comp_tex_bgr = to_u8_bgr(img_comp_tex)

        arrays = [
            self._as_contig(depth_u16),
            self._as_contig(mask_u8),
            self._as_contig(img_bgr),
            self._as_contig(comp_bgr),
            self._as_contig(comp_geo_bgr),
            self._as_contig(comp_tex_bgr),
            self._as_contig(intr),
            self._as_contig(extr),
        ]

        data_id_b = data_id.encode("utf-8")
        if len(data_id_b) > 65535:
            raise ValueError("data_id too long")

        fixed_hdr = (
            struct.pack("<B H", int(phase_id) & 0xFF, len(data_id_b))
            + data_id_b
            + struct.pack("<H B B", int(pid) & 0xFFFF, int(vid) & 0xFF, 0)
        )

        meta = b"".join([self._meta20(a) for a in arrays])
        payload_nbytes = sum(int(a.nbytes) for a in arrays)
        record_len = 4 + len(fixed_hdr) + len(meta) + payload_nbytes

        offset = self.f.tell()
        self.f.write(struct.pack("<I", record_len))
        self.f.write(fixed_hdr)
        self.f.write(meta)
        for a in arrays:
            self.f.write(a)

        self.index.append((offset, record_len))

    def close(self):
        index_offset = self.f.tell()
        self.f.write(struct.pack("<4sI", self.INDEX_MAGIC, len(self.index)))
        for off, ln in self.index:
            self.f.write(struct.pack("<QI", int(off), int(ln)))

        self.f.flush()
        self.f.seek(0, os.SEEK_SET)
        res_w, res_h = self.res
        header = struct.pack(
            "<4sIIIQQ",
            self.MAGIC,
            self.VERSION,
            res_w,
            res_h,
            int(index_offset),
            int(len(self.index)),
        )
        header += b"\x00" * (self.HEADER_SIZE - len(header))
        self.f.write(header)
        self.f.flush()
        self.f.close()

class StaticRenderer:
    def __init__(self, src_res):
        ti.init(arch=ti.cuda, device_memory_fraction=float(os.environ.get("TI_MEM_FRAC", "0.15")))
        self.scene = t3.Scene()
        self.N = 150000
        self.src_res = src_res

        self.current_light_angles = np.zeros(6)

    def change_all(self):
        save_obj = [m.init_obj for m in self.scene.models]
        save_tex = [m.init_tex for m in self.scene.models]
        save_tex_comp = [getattr(m, 'init_tex_comp', None) for m in self.scene.models]
        save_tex_comp_geo = [getattr(m, 'init_tex_comp_geo', None) for m in self.scene.models]
        save_tex_comp_tex = [getattr(m, 'init_tex_comp_tex', None) for m in self.scene.models]

        ti.init(arch=ti.cuda, device_memory_fraction=float(os.environ.get("TI_MEM_FRAC", "0.15")))
        self.scene = t3.Scene()
        for i in range(len(save_obj)):
            self.scene.add_model(t3.StaticModel(
                self.N, obj=save_obj[i], tex=save_tex[i],
                tex_comp=save_tex_comp[i], tex_comp_geo=save_tex_comp_geo[i], tex_comp_tex=save_tex_comp_tex[i]
            ))

        self.apply_camera_and_lights()

    def check_update(self, obj):
        temp_n = self.N
        self.N = max(obj["vi"].shape[0], self.N)
        self.N = max(obj["f"].shape[0], self.N)
        if obj["vt"] is not None:
            self.N = max(obj["vt"].shape[0], self.N)

        if self.N > temp_n:
            self.N *= 2
            self.change_all()

    def add_model(self, obj, tex=None, tex_comp=None, tex_comp_geo=None, tex_comp_tex=None):
        self.check_update(obj)
        self.scene.add_model(t3.StaticModel(
            self.N, obj=obj, tex=tex,
            tex_comp=tex_comp, tex_comp_geo=tex_comp_geo, tex_comp_tex=tex_comp_tex
        ))

    def modify_model(self, index, obj, tex=None, tex_comp=None, tex_comp_geo=None, tex_comp_tex=None):
        self.check_update(obj)
        model = self.scene.models[index]
        model.init_obj = obj
        model.init_tex = tex
        model.init_tex_comp = tex_comp
        model.init_tex_comp_geo = tex_comp_geo
        model.init_tex_comp_tex = tex_comp_tex
        model._init()

    def update_subject_lights(self, rng_subj):
        self.current_light_angles = rng_subj.uniform(-30, 30, 6)
        self.apply_camera_and_lights()

    def apply_camera_and_lights(self):
        if len(self.scene.cameras) == 0:
            camera = t3.Camera(res=self.src_res)
            self.scene.add_camera(camera)

        light_dir = np.array([0, 0, 1])
        light_list = []
        for l in range(6):
            rotate = np.matmul(
                rotationX(math.radians(self.current_light_angles[l])),
                rotationY(math.radians(360 // 6 * l)),
            )
            dir = [*np.matmul(rotate, light_dir)]
            light_list.append(t3.Light(dir, color=[1.0, 1.0, 1.0]))

        self.scene.add_lights(t3.Lights(light_list))

def render_subject_pid_to_rawpack(
    renderer,
    writer: THRawPackWriter,
    phase_id: int,
    data_id: str,
    pid: int,
    cam_nums: int,
    res: tuple,
    dis: float,
    angle_base: float,
    seed_subj: int,
):
    rng_pid = np.random.default_rng((seed_subj ^ (pid * 0x9E3779B1)) & 0xFFFFFFFF)

    look_at_center = np.array([0, 0.85, 0])
    base_cam_pitch = -8

    degree_interval = 360 / cam_nums
    angle = angle_base + pid * degree_interval
    angles_to_render = [
        angle,
        (angle + degree_interval) % 360,
        (angle + (rng_pid.uniform() * degree_interval / 2)) % 360,
        (angle + degree_interval / 2) % 360,
        (angle + degree_interval - (rng_pid.uniform() * degree_interval / 2)) % 360,
    ]

    for vid, target_angle in enumerate(angles_to_render):

        ori_vec = np.array([0, 0, dis])
        rotate = np.matmul(rotationY(math.radians(target_angle)), rotationX(math.radians(base_cam_pitch)))
        fwd = np.matmul(rotate, ori_vec)
        cam_pos = look_at_center + fwd

        fx, fy = res[0] * 0.8, res[1] * 0.8
        cx = res[0] * 0.5
        cy = res[1] * 0.5 + 50

        cam = renderer.scene.cameras[0]
        cam.set_intrinsic(fx, fy, cx, cy)
        cam.set(pos=cam_pos, target=look_at_center)
        cam._init()

        renderer.scene.render()

        extr = cam.export_extrinsic()
        intr = cam.export_intrinsic()
        depth = cam.zbuf.to_numpy().swapaxes(0, 1)
        mask = cam.mask.to_numpy().swapaxes(0, 1)
        img = cam.img.to_numpy().swapaxes(0, 1)
        img_comp = getattr(cam, 'img_comp').to_numpy().swapaxes(0, 1)
        img_comp_geo = getattr(cam, 'img_comp_geo').to_numpy().swapaxes(0, 1)
        img_comp_tex = getattr(cam, 'img_comp_tex').to_numpy().swapaxes(0, 1)

        kwargs = dict(
            phase_id=phase_id,
            data_id=data_id,
            pid=pid,
            vid=vid,
            extr=extr,
            intr=intr,
            depth=depth,
            img=img,
            mask=mask,
            img_comp=img_comp,
            img_comp_geo=img_comp_geo,
            img_comp_tex=img_comp_tex,
        )
        io_queue.put(("write", writer, kwargs))

def stable_seed_from_data_id(data_id: str, base_seed: int) -> int:
    crc = zlib.crc32(data_id.encode("utf-8")) & 0xFFFFFFFF
    return (int(base_seed) + int(crc)) & 0xFFFFFFFF

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--thuman_root", type=str, default="data/thuman2.0_raw")
    parser.add_argument("--out_root", type=str, default="data/thuman2.0_rendered")
    parser.add_argument("--pack_subdir", type=str, default="_rawpack_by_pid")

    parser.add_argument("--phase", type=str, required=True, choices=["train", "val", "test"])
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--batch_idx", type=int, default=0)

    parser.add_argument("--cam_nums", type=int, default=16)
    parser.add_argument("--scene_radius", type=float, default=2.0)
    parser.add_argument("--res", type=int, nargs=2, default=[2048, 2048])

    parser.add_argument("--base_seed", type=int, default=1314)
    parser.add_argument("--skip_existing", action="store_true", help="skip subjects whose pack files already exist")

    args = parser.parse_args()

    writer_thread = threading.Thread(target=io_worker, daemon=True)
    writer_thread.start()

    cam_nums = int(args.cam_nums)
    scene_radius = float(args.scene_radius)
    res = (int(args.res[0]), int(args.res[1]))

    thuman_root = args.thuman_root
    base_out_root = args.out_root
    pack_root = os.path.join(base_out_root, args.pack_subdir)

    phase_dir = os.path.join(thuman_root, args.phase)
    if not os.path.isdir(phase_dir):
        raise FileNotFoundError(f"Phase dir not found: {phase_dir}")

    thuman_list = sorted(os.listdir(phase_dir))
    total = len(thuman_list)

    bs = max(1, int(args.batch_size))
    bi = max(0, int(args.batch_idx))
    start = bi * bs
    end = min(total, (bi + 1) * bs)
    batch_list = thuman_list[start:end]

    if len(batch_list) == 0:
        print(f"[Done] No subjects for phase={args.phase}, batch_idx={bi}, batch_size={bs} (total={total})")
        raise SystemExit(0)

    out_phase_dir = os.path.join(pack_root, args.phase)
    Path(out_phase_dir).mkdir(parents=True, exist_ok=True)

    print(f"[Batch] phase={args.phase} batch_idx={bi} batch_size={bs} -> subjects[{start}:{end}] / total={total}")

    renderer = StaticRenderer(src_res=res)

    phase_id = {"train": 0, "val": 1, "test": 2}.get(args.phase, 255)
    if phase_id == 255:
        raise ValueError(f"Unknown phase: {args.phase}")

    for data_id in tqdm(batch_list, desc=f"render {args.phase} b{bi}"):
        seed_subj = stable_seed_from_data_id(data_id, args.base_seed)
        rng_subj = np.random.default_rng(seed_subj)

        renderer.update_subject_lights(rng_subj)

        subj_dir = os.path.join(out_phase_dir, data_id)
        Path(subj_dir).mkdir(parents=True, exist_ok=True)

        obj_path = os.path.join(thuman_root, args.phase, data_id, f"{data_id}.obj")
        img_path = os.path.join(thuman_root, args.phase, data_id, "material0.jpeg")
        comp_path = os.path.join(thuman_root, args.phase, data_id, "material0_complexity.png")
        comp_path_geo = os.path.join(thuman_root, args.phase, data_id, "material0_complexity_geo.png")
        comp_path_tex = os.path.join(thuman_root, args.phase, data_id, "material0_complexity_tex.png")

        texture_main = load_tex(img_path)
        texture_comp = load_tex(comp_path)
        texture_comp_geo = load_tex(comp_path_geo)
        texture_comp_tex = load_tex(comp_path_tex)

        try:
            obj = t3.readobj(obj_path, scale=1)
        except Exception as e:
            print(f"[Error] Failed to load OBJ {data_id}: {e}")
            continue

        vy_max = np.max(obj["vi"][:, 1])
        vy_min = np.min(obj["vi"][:, 1])
        human_height = 1.80 + float(rng_subj.uniform(-0.05, 0.05))
        obj["vi"][:, :3] = obj["vi"][:, :3] / (vy_max - vy_min) * human_height
        obj["vi"][:, 1] -= np.min(obj["vi"][:, 1])

        degree_interval = 360 / cam_nums
        angle_list1 = list(range(360 - int(degree_interval // 2), 360))
        angle_list2 = list(range(0, 0 + int(degree_interval // 2)))
        angle_list = angle_list1 + angle_list2
        angle_base = int(rng_subj.choice(angle_list))

        smpl_path = os.path.join(thuman_root, "THuman2.0_Smpl_X_Paras", data_id, "smplx_param.pkl")
        if os.path.exists(smpl_path):
            with open(smpl_path, "rb") as f:
                smpl_para = pickle.load(f)
                if "global_orient" in smpl_para:
                    y_orient = smpl_para["global_orient"][0][1]
                    angle_base += (y_orient * 180.0 / np.pi)

        if len(renderer.scene.models) >= 1:
            renderer.modify_model(0, obj, texture_main, texture_comp, texture_comp_geo, texture_comp_tex)
        else:
            renderer.add_model(obj, texture_main, texture_comp, texture_comp_geo, texture_comp_tex)

        for pid in range(cam_nums):
            out_file = os.path.join(subj_dir, f"pid_{pid:02d}.thrw")
            if args.skip_existing and os.path.exists(out_file):
                continue

            writer = THRawPackWriter(out_path=out_file, res=res)

            render_subject_pid_to_rawpack(
                renderer, writer,
                phase_id=phase_id,
                data_id=data_id,
                pid=pid,
                cam_nums=cam_nums,
                res=res,
                dis=scene_radius,
                angle_base=angle_base,
                seed_subj=seed_subj
            )

            io_queue.put(("close", writer, {}))

    print("[Info] Waiting for background disk IO to finish...")
    io_queue.join()

    io_queue.put(None)
    writer_thread.join()
    print("[Done] All completed successfully.")
