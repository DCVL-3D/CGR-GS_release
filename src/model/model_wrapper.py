from dataclasses import dataclass
from pathlib import Path
import time

import torch
from einops import rearrange
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities import rank_zero_only
from torch import Tensor, nn

from ..dataset.data_module import get_data_shim
from ..dataset.types import BatchedExample
from ..evaluation.metrics import compute_lpips, compute_psnr, compute_ssim
from ..misc.image_io import save_image
from ..visualization.annotation import add_label
from ..visualization.layout import add_border, hcat
from .decoder.decoder import Decoder
from .encoder import Encoder

@dataclass
class TestCfg:
    output_path: Path
    save_images: bool = True

class ModelWrapper(LightningModule):
    encoder: nn.Module
    decoder: Decoder
    test_cfg: TestCfg

    def __init__(
        self,
        test_cfg: TestCfg,
        encoder: Encoder,
        decoder: Decoder,
    ) -> None:
        super().__init__()
        self.test_cfg = test_cfg

        self.encoder = encoder
        self.decoder = decoder
        self.data_shim = get_data_shim(self.encoder)

        self._scene_metrics: list[dict] = []
        self._image_metrics: list[dict] = []

        self._enc_times: list[float] = []
        self._dec_times: list[float] = []
        self._total_infer_times: list[float] = []

    def on_load_checkpoint(self, checkpoint: dict) -> None:

        sd = checkpoint.get("state_dict", {})
        key = "encoder.offset_range"
        if key not in sd and hasattr(self.encoder, "offset_range"):
            sd[key] = self.encoder.offset_range.detach().clone().cpu()

    @rank_zero_only
    def on_test_start(self) -> None:
        self.test_cfg.output_path.mkdir(parents=True, exist_ok=True)

    def test_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)

        if self.global_rank == 0:
            print(
                f"test step {self.global_step}; "
                f"scene = {batch['scene']}; "
                f"context = {batch['context']['index'].tolist()}"
            )

        b, v, _, h, w = batch["target"]["image"].shape
        assert b == 1, "test mode supports batch_size=1 only."

        should_measure = batch_idx >= 5

        if should_measure:

            start_event = torch.cuda.Event(enable_timing=True)
            enc_end_event = torch.cuda.Event(enable_timing=True)
            dec_end_event = torch.cuda.Event(enable_timing=True)

            torch.cuda.reset_peak_memory_stats()
            start_event.record()

        gaussians = self.encoder(batch["context"])

        if should_measure:
            enc_end_event.record()

        output: DecoderOutput = self.decoder.forward(
            gaussians,
            batch["target"]["extrinsics"],
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            (h, w),
        )

        if should_measure:
            dec_end_event.record()
            torch.cuda.synchronize()

            enc_time_s = start_event.elapsed_time(enc_end_event) / 1000.0
            dec_time_s = enc_end_event.elapsed_time(dec_end_event) / 1000.0
            total_time_s = start_event.elapsed_time(dec_end_event) / 1000.0

            self._enc_times.append(enc_time_s)
            self._dec_times.append(dec_time_s)
            self._total_infer_times.append(total_time_s)

            peak_mb_i = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
            self._peak_mems_mb.append(peak_mb_i)

            ps = self.encoder.patch_size
            mask_patches = rearrange(
                batch["context"]["mask"].float(),
                "b v c (hh ph) (ww pw) -> b (v hh ww) (c ph pw)",
                ph=ps, pw=ps,
            )
            n_valid = int((mask_patches.sum(-1) > 0.5).sum().item())
            self._token_counts.append(n_valid)
            scene = batch["scene"]
            scene_name_t = (
                str(int(scene[0])) if isinstance(scene, torch.Tensor)
                else (str(scene[0]) if isinstance(scene, (list, tuple)) else str(scene))
            )
            self._scene_names_t.append(scene_name_t)

        render_images = output.color[0]
        gt_images     = batch["target"]["image"][0]

        render_depths = output.depth[0]
        gt_depths     = batch["target"]["depth"][0]

        psnr_per_view  = compute_psnr(gt_images, render_images)
        lpips_per_view = compute_lpips(gt_images, render_images)
        ssim_per_view  = compute_ssim(gt_images, render_images)

        psnr_scene  = psnr_per_view.mean()
        lpips_scene = lpips_per_view.mean()
        ssim_scene  = ssim_per_view.mean()

        self.log("test/psnr",  psnr_scene, prog_bar=True)
        self.log("test/lpips", lpips_scene)
        self.log("test/ssim",  ssim_scene)

        scene = batch["scene"]
        if isinstance(scene, torch.Tensor):
            scene_name = str(int(scene[0]))
        else:
            scene_name = str(scene[0]) if isinstance(scene, (list, tuple)) else str(scene)

        self._scene_metrics.append(
            {
                "scene": scene_name,
                "psnr":  psnr_scene.item(),
                "lpips": lpips_scene.item(),
                "ssim":  ssim_scene.item(),
            }
        )

        for i in range(v):
            self._image_metrics.append(
                {
                    "scene": scene_name,
                    "view":  int(i),
                    "psnr":  psnr_per_view[i].item(),
                    "lpips": lpips_per_view[i].item(),
                    "ssim":  ssim_per_view[i].item(),
                }
            )

        if not self.test_cfg.save_images:
            return {
                "psnr":  psnr_scene.detach(),
                "lpips": lpips_scene.detach(),
                "ssim":  ssim_scene.detach(),
            }

        scene_dir = self.test_cfg.output_path / scene_name
        scene_dir.mkdir(parents=True, exist_ok=True)

        gt_dir   = scene_dir / "gt"
        pred_dir = scene_dir / "pred"
        comp_dir = scene_dir / "comparison"

        gt_dir.mkdir(parents=True, exist_ok=True)
        pred_dir.mkdir(parents=True, exist_ok=True)
        comp_dir.mkdir(parents=True, exist_ok=True)

        gt_depth_dir   = scene_dir / "gt_depth"
        pred_depth_dir = scene_dir / "pred_depth"
        comp_depth_dir = scene_dir / "comparison_depth"

        for i in range(v):
            gt_img   = gt_images[i].detach().cpu().float()
            pred_img = render_images[i].detach().cpu().float()

            gt_path   = gt_dir   / f"{batch_idx:05d}_view{i:02d}.png"
            pred_path = pred_dir / f"{batch_idx:05d}_view{i:02d}.png"
            save_image(gt_img,   gt_path)
            save_image(pred_img, pred_path)

            comparison = hcat(
                add_label(gt_images[i],     "GT"),
                add_label(render_images[i], "Pred"),
            )
            comp_img  = add_border(comparison).detach().cpu().float()
            comp_path = comp_dir / f"{batch_idx:05d}_view{i:02d}.png"
            save_image(comp_img, comp_path)

            gt_depth_raw   = gt_depths[i].detach().cpu().float()
            pred_depth_raw = render_depths[i].detach().cpu().float()

            gt_depth_color   = self.apply_jet_colormap(gt_depth_raw, vmin=0.5, vmax=3.5)
            pred_depth_color = self.apply_jet_colormap(pred_depth_raw, vmin=0.5, vmax=3.5)

            gt_depth_path   = gt_depth_dir   / f"{batch_idx:05d}_view{i:02d}.png"
            pred_depth_path = pred_depth_dir / f"{batch_idx:05d}_view{i:02d}.png"
            save_image(gt_depth_color,   gt_depth_path)
            save_image(pred_depth_color, pred_depth_path)

            eps = 1e-5

            invalid_gt = (torch.abs(gt_depth_raw - 1.2) < eps) | (torch.abs(gt_depth_raw - 3.0) < eps)

            invalid_pred = (torch.abs(pred_depth_raw - 1.2) < eps) | (torch.abs(pred_depth_raw - 3.0) < eps)

            valid_mask = ~(invalid_gt | invalid_pred)

            diff_raw = torch.abs(gt_depth_raw - pred_depth_raw)

            diff_raw = diff_raw * valid_mask.float()

            diff_color = self.apply_jet_colormap(diff_raw, vmin=0.0, vmax=1.0)

            diff_color = diff_color * valid_mask.float()

            comparison_depth = hcat(
                add_label(gt_depth_color,   "GT Depth"),
                add_label(pred_depth_color, "Pred Depth"),
                add_label(diff_color,       "Diff (Masked)")
            )
            comp_depth_img  = add_border(comparison_depth).detach().cpu().float()
            comp_depth_path = comp_depth_dir / f"{batch_idx:05d}_view{i:02d}.png"
            save_image(comp_depth_img, comp_depth_path)

        return {
            "psnr":  psnr_scene.detach(),
            "lpips": lpips_scene.detach(),
            "ssim":  ssim_scene.detach(),
        }

    def apply_jet_colormap(self, depth_tensor: torch.Tensor, vmin: float = 0.0, vmax: float = 4.0) -> torch.Tensor:
        import matplotlib.cm as cm

        depth_norm = torch.clamp((depth_tensor - vmin) / (vmax - vmin), 0.0, 1.0)

        depth_np = depth_norm.squeeze().cpu().numpy()

        depth_color_np = cm.jet(depth_np)[..., :3]

        depth_color_tensor = torch.from_numpy(depth_color_np).permute(2, 0, 1).float()

        return depth_color_tensor.to(depth_tensor.device)

    @rank_zero_only
    def on_test_epoch_start(self) -> None:
        self.test_cfg.output_path.mkdir(parents=True, exist_ok=True)

        self._n_params_enc = sum(p.numel() for p in self.encoder.parameters())
        self._n_params_dec = sum(p.numel() for p in self.decoder.parameters())
        print(f"[FPS bench] encoder params: {self._n_params_enc:,}  decoder params: {self._n_params_dec:,}")

        self._enc_times = []
        self._dec_times = []
        self._total_infer_times = []
        self._peak_mems_mb = []
        self._token_counts = []
        self._scene_names_t = []
        self._scene_metrics = []
        self._image_metrics = []

    def on_test_epoch_end(self) -> None:
        metrics = {
            k: v.item()
            for k, v in self.trainer.callback_metrics.items()
            if k.startswith("test/")
        }

        def _fps_stats(times):
            fps = [1.0 / t for t in times]
            return {
                "best":  max(fps),
                "mean":  sum(fps) / len(fps),
                "worst": min(fps),
            }

        if self._total_infer_times:
            avg_enc_s = sum(self._enc_times) / len(self._enc_times)
            avg_dec_s = sum(self._dec_times) / len(self._dec_times)
            avg_tot_s = sum(self._total_infer_times) / len(self._total_infer_times)

            metrics["perf/avg_enc_time_ms"] = avg_enc_s * 1000
            metrics["perf/avg_dec_time_ms"] = avg_dec_s * 1000
            metrics["perf/avg_total_time_ms"] = avg_tot_s * 1000

            metrics["perf/n_params_encoder"]   = self._n_params_enc
            metrics["perf/n_params_decoder"]   = self._n_params_dec
            metrics["perf/n_params_total"]     = self._n_params_enc + self._n_params_dec

            if self._peak_mems_mb:
                metrics["perf/peak_mem_mb_best"]  = round(min(self._peak_mems_mb), 1)
                metrics["perf/peak_mem_mb_mean"]  = round(sum(self._peak_mems_mb) / len(self._peak_mems_mb), 1)
                metrics["perf/peak_mem_mb_worst"] = round(max(self._peak_mems_mb), 1)

            for stage, ts in (("encoder", self._enc_times),
                              ("decoder", self._dec_times),
                              ("total",   self._total_infer_times)):
                s = _fps_stats(ts)
                metrics[f"perf/fps_{stage}_best"]  = s["best"]
                metrics[f"perf/fps_{stage}_mean"]  = s["mean"]
                metrics[f"perf/fps_{stage}_worst"] = s["worst"]

        root = self.test_cfg.output_path
        metrics_path = root / "metrics.txt"
        with open(metrics_path, "w") as f:
            for k, v in metrics.items():
                if "perf" in k or "fps" in k:
                    f.write(f"{k}: {v:.2f}\n")
                else:
                    f.write(f"{k}: {v}\n")

            if "perf/fps_total_mean" in metrics:
                n = len(self._total_infer_times)
                f.write("\n--- Performance Summary (1-view, 4-view input, mask tokens only) ---\n")
                f.write(f"N samples (warmup-excluded): {n}\n")
                for stage in ("encoder", "decoder", "total"):
                    f.write(
                        f"{stage:<7s} FPS  best/mean/worst : "
                        f"{metrics[f'perf/fps_{stage}_best']:.2f} / "
                        f"{metrics[f'perf/fps_{stage}_mean']:.2f} / "
                        f"{metrics[f'perf/fps_{stage}_worst']:.2f}\n"
                    )
                f.write(f"Encoder time (ms) avg : {metrics['perf/avg_enc_time_ms']:.2f}\n")
                f.write(f"Decoder time (ms) avg : {metrics['perf/avg_dec_time_ms']:.2f}\n")
                f.write(f"Total   time (ms) avg : {metrics['perf/avg_total_time_ms']:.2f}\n")
                f.write(f"Encoder params        : {metrics['perf/n_params_encoder']:>12,}\n")
                f.write(f"Decoder params        : {metrics['perf/n_params_decoder']:>12,}\n")
                f.write(f"Total   params        : {metrics['perf/n_params_total']:>12,}\n")
                if "perf/peak_mem_mb_mean" in metrics:
                    f.write(
                        f"Peak GPU mem (MB) best/mean/worst : "
                        f"{metrics['perf/peak_mem_mb_best']:.1f} / "
                        f"{metrics['perf/peak_mem_mb_mean']:.1f} / "
                        f"{metrics['perf/peak_mem_mb_worst']:.1f}\n"
                    )

                if self._token_counts:
                    fps_total = [1.0 / t for t in self._total_infer_times]
                    i_best  = max(range(n), key=lambda i: fps_total[i])
                    i_worst = min(range(n), key=lambda i: fps_total[i])
                    mean_fps = metrics["perf/fps_total_mean"]
                    i_mean  = min(range(n), key=lambda i: abs(fps_total[i] - mean_fps))
                    avg_tok = sum(self._token_counts) / len(self._token_counts)
                    f.write("\n--- Valid mask token counts ---\n")
                    f.write(f"best  FPS sample : scene={self._scene_names_t[i_best]:>6s}  "
                            f"n_tokens={self._token_counts[i_best]:>5d}  fps={fps_total[i_best]:.2f}\n")
                    f.write(f"mean  FPS sample : scene={self._scene_names_t[i_mean]:>6s}  "
                            f"n_tokens={self._token_counts[i_mean]:>5d}  fps={fps_total[i_mean]:.2f}\n")
                    f.write(f"worst FPS sample : scene={self._scene_names_t[i_worst]:>6s}  "
                            f"n_tokens={self._token_counts[i_worst]:>5d}  fps={fps_total[i_worst]:.2f}\n")
                    f.write(f"n_tokens min/avg/max : "
                            f"{min(self._token_counts)} / {avg_tok:.1f} / {max(self._token_counts)}\n")

        if self._total_infer_times:
            per_sample_path = root / "per_sample_perf.csv"
            with open(per_sample_path, "w") as f_ps:
                f_ps.write("scene,n_tokens,enc_ms,dec_ms,total_ms,fps_total,peak_mem_mb\n")
                for i in range(len(self._total_infer_times)):
                    pmem = self._peak_mems_mb[i] if i < len(self._peak_mems_mb) else 0.0
                    f_ps.write(
                        f"{self._scene_names_t[i]},"
                        f"{self._token_counts[i]},"
                        f"{self._enc_times[i]*1000:.3f},"
                        f"{self._dec_times[i]*1000:.3f},"
                        f"{self._total_infer_times[i]*1000:.3f},"
                        f"{1.0/self._total_infer_times[i]:.3f},"
                        f"{pmem:.1f}\n"
                    )

        per_scene_path = root / "per_scene_metrics.txt"
        with open(per_scene_path, "w") as f_scene:
            f_scene.write("scene psnr lpips ssim\n")
            for m in self._scene_metrics:
                f_scene.write(
                    f"{m['scene']} "
                    f"{m['psnr']:.6f} "
                    f"{m['lpips']:.6f} "
                    f"{m['ssim']:.6f}\n"
                )

        per_image_path = root / "per_image_metrics.txt"
        with open(per_image_path, "w") as f_img:
            f_img.write("scene view psnr lpips ssim\n")
            for m in self._image_metrics:
                f_img.write(
                    f"{m['scene']} "
                    f"{m['view']:02d} "
                    f"{m['psnr']:.6f} "
                    f"{m['lpips']:.6f} "
                    f"{m['ssim']:.6f}\n"
                )
