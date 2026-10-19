from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from einops import rearrange
from typing import Tuple
from torch import Tensor, nn

from .transformer_processor.processor import Processor

from ...dataset.shims.bounds_shim import apply_bounds_shim
from ...dataset.shims.patch_shim import apply_patch_shim_to_views
from ...dataset.types import BatchedExample, DataShim
from ..types import Gaussians

from .encoder import Encoder

EPS = 1e-8

def _init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def quaternion_to_matrix(
    quaternions,
    eps = 1e-8,
):

    i, j, k, r = torch.unbind(quaternions, dim=-1)
    two_s = 2 / ((quaternions * quaternions).sum(dim=-1) + eps)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return rearrange(o, "... (i j) -> ... i j", i=3, j=3)

def build_covariance(
    scale,
    rotation_xyzw,
):
    scale = scale.diag_embed()
    rotation = quaternion_to_matrix(rotation_xyzw)
    return (
        rotation
        @ scale
        @ rearrange(scale, "... i j -> ... j i")
        @ rearrange(rotation, "... i j -> ... j i")
    )

@dataclass
class TransformerCfg:
    head_dim: int
    num_layers: int

@dataclass
class GaussianCfg:
    sh_degree: int
    scale_bias: float
    scale_max: float
    opacity_bias: float
    near_plane: float
    far_plane: float

@dataclass
class EncoderLRMCfg:
    name: Literal["lrm"]
    patch_size: int
    attn_dim: int
    transformer: TransformerCfg
    gaussians_params: GaussianCfg
    apply_bounds_shim: bool
    near_disparity: float

    offset_range: float = 8.0

    offset_z_range: float = 1.5

class OffsetHead(nn.Module):
    def __init__(self, dim: int, attr_dim: int, K: int, hidden: int = 512) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, bias=False)
        self.feat = nn.Linear(dim, hidden)
        self.attr = nn.Linear(attr_dim, hidden, bias=False)
        self.out_w = nn.Parameter(torch.zeros(K, hidden, 3))
        self.out_b = nn.Parameter(torch.zeros(K, 3))
        self.scale = nn.Linear(attr_dim, 1)

    def forward(self, x: Tensor, attrs: Tensor) -> Tensor:
        f = self.feat(self.norm(x))
        h = F.gelu(f[:, :, None, :] + self.attr(attrs))
        out = torch.einsum("blkh,khc->blkc", h, self.out_w) + self.out_b
        return out * torch.sigmoid(self.scale(attrs))

class EncoderLRM(Encoder[EncoderLRMCfg]):
    tokenizer: nn.Sequential
    attn_processor: Processor
    token_decoder: nn.Sequential

    def __init__(self, cfg: EncoderLRMCfg) -> None:
        super().__init__(cfg)
        input_dim = 9
        self.patch_size = cfg.patch_size
        self.attn_dim = cfg.attn_dim

        self.K = 64

        self.tokenizer = nn.Sequential(
            nn.Linear(input_dim * self.patch_size ** 2, self.attn_dim, bias=False),
        )
        self.tokenizer.apply(_init_weights)

        self.attn_processor = Processor(cfg)

        self.sh_dim = (cfg.gaussians_params.sh_degree + 1) ** 2 * 3

        self.offset_dim = 2
        self.d_per_gaussian = 1 + self.offset_dim + self.sh_dim + 3 + 4 + 1

        out_dim = self.K * self.d_per_gaussian

        self.token_decoder = nn.Sequential(
            nn.LayerNorm(self.attn_dim, bias=False),
            nn.Linear(self.attn_dim, out_dim, bias=False)
        )
        self.token_decoder.apply(_init_weights)

        self.offset_attn = nn.TransformerEncoderLayer(
            d_model=self.attn_dim, nhead=8, dim_feedforward=2048, dropout=0.0,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.offset_head = OffsetHead(self.attn_dim, self.d_per_gaussian, self.K)

        self.register_buffer("offset_range", torch.tensor(float(cfg.offset_range)))

    def get_camera_ray_dir(
        self,
        context: dict,
    ) -> Tuple[Tensor, Tensor]:

        dtype = context["image"].dtype
        device = context["image"].device
        B, V, _, H, W = context["image"].shape
        input_c2ws, input_intr_raw = context["extrinsics"], context["intrinsics"]

        fx = input_intr_raw[..., 0, 0].unsqueeze(-1) * W
        fy = input_intr_raw[..., 1, 1].unsqueeze(-1) * H
        cx = input_intr_raw[..., 0, 2].unsqueeze(-1) * W
        cy = input_intr_raw[..., 1, 2].unsqueeze(-1) * H

        input_intr = torch.cat([fx, fy, cx, cy], dim=-1)

        ray_o = input_c2ws[:, :, :3, 3].unsqueeze(2).expand(-1, -1, H * W, -1).float()
        x, y = torch.meshgrid(torch.arange(W), torch.arange(H), indexing="xy")
        x = (x.to(dtype) + 0.5).view(1, 1, -1).expand(B, V, -1).to(device).contiguous()
        y = (y.to(dtype) + 0.5).view(1, 1, -1).expand(B, V, -1).to(device).contiguous()

        x = (x - input_intr[:, :, 2:3]) / input_intr[:, :, 0:1]
        y = (y - input_intr[:, :, 3:4]) / input_intr[:, :, 1:2]
        ray_d = torch.stack([x, y, torch.ones_like(x)], dim=-1).float()
        ray_d = F.normalize(ray_d, p=2, dim=-1)
        ray_d = ray_d @ input_c2ws[:, :, :3, :3].transpose(-1, -2).contiguous()
        return ray_o, ray_d

    def feat2gaussian(
        self,
        gaussian_params: dict,
    ) -> Gaussians:
        means, scales, rotations, sh_feature, opacities = gaussian_params['xyz'], gaussian_params['scale'], gaussian_params['rotation'], gaussian_params['sh_feature'], gaussian_params['opacity']
        covariances = build_covariance(scales, rotations)

        return Gaussians(
            means=means.float(),
            covariances=covariances.float(),
            harmonics=sh_feature.permute(0, 1, 3, 2).contiguous().float(),
            opacities=opacities.squeeze(-1).float(),
        )

    def forward(
        self,
        context: dict,
    ) -> Gaussians:
        device = context["image"].device
        b, v, _, h, w = context["image"].shape
        patch_size = self.patch_size

        ray_o, ray_d = self.get_camera_ray_dir(context)

        input_image_cam = torch.cat([
            context["image"].view(b, v, 3, -1).permute(0, 1, 3, 2).contiguous() * 2 - 1,
            torch.cross(ray_o, ray_d, dim=-1),
            ray_d
        ], dim=-1)

        hh = h // patch_size
        ww = w // patch_size

        input_tokens = rearrange(input_image_cam,
                                "b v (hh ph ww pw) d -> b (v hh ww) (ph pw d)",
                                hh=hh, ww=ww, ph=patch_size, pw=patch_size)

        mask = context["mask"].float()
        mask_patches = rearrange(mask,
                                "b v c (hh ph) (ww pw) -> b (v hh ww) (c ph pw)",
                                hh=hh, ww=ww, ph=patch_size, pw=patch_size)

        valid_mask = (mask_patches.sum(dim=-1) > 0.5).long()

        sorted_indices = torch.argsort(valid_mask, dim=1, descending=True)

        batch_valid_counts = valid_mask.sum(dim=1)
        max_valid_len = batch_valid_counts.max().item()

        d_dim = input_tokens.shape[-1]
        gather_indices = sorted_indices.unsqueeze(-1).expand(-1, -1, d_dim)

        packed_tokens = torch.gather(input_tokens, 1, gather_indices)
        active_tokens = packed_tokens[:, :max_valid_len, :]

        attn_mask = torch.arange(max_valid_len, device=device)[None, :] < batch_valid_counts[:, None]

        image_tokens = self.tokenizer(active_tokens)

        with torch.amp.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
            image_tokens = self.attn_processor(image_tokens, mask=attn_mask, use_checkpoint=True)

        gaussians = self.token_decoder(image_tokens)
        feats = image_tokens.float()
        with torch.amp.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
            feats = self.offset_attn(feats, src_key_padding_mask=~attn_mask)
        attrs = gaussians.view(b, max_valid_len, self.K, self.d_per_gaussian).float()
        offset_raw = self.offset_head(feats.float(), attrs)

        ray_o_patches = rearrange(ray_o, "b v (hh ph ww pw) c -> b (v hh ww) (ph pw) c", hh=hh, ww=ww, ph=patch_size, pw=patch_size)
        ray_d_patches = rearrange(ray_d, "b v (hh ph ww pw) c -> b (v hh ww) (ph pw) c", hh=hh, ww=ww, ph=patch_size, pw=patch_size)

        center_idx = (patch_size // 2) * patch_size + (patch_size // 2)
        ray_o_center = ray_o_patches[:, :, center_idx, :]
        ray_d_center = ray_d_patches[:, :, center_idx, :]

        input_c2ws = context["extrinsics"]
        c2w_rot = input_c2ws[:, :, :3, :3].transpose(-1, -2)
        cam_x = c2w_rot[:, :, :, 0]
        cam_y = c2w_rot[:, :, :, 1]

        input_intr_raw = context["intrinsics"]
        fx = input_intr_raw[..., 0, 0] * w
        fy = input_intr_raw[..., 1, 1] * h

        cam_x_tokens = cam_x.view(b, v, 1, 1, 3).expand(b, v, hh, ww, 3).reshape(b, -1, 3)
        cam_y_tokens = cam_y.view(b, v, 1, 1, 3).expand(b, v, hh, ww, 3).reshape(b, -1, 3)
        fx_tokens = fx.view(b, v, 1, 1, 1).expand(b, v, hh, ww, 1).reshape(b, -1, 1)
        fy_tokens = fy.view(b, v, 1, 1, 1).expand(b, v, hh, ww, 1).reshape(b, -1, 1)

        gather_indices_3d = sorted_indices.unsqueeze(-1).expand(-1, -1, 3)
        gather_indices_1d = sorted_indices.unsqueeze(-1)

        active_ray_o = torch.gather(ray_o_center, 1, gather_indices_3d)[:, :max_valid_len]
        active_ray_d = torch.gather(ray_d_center, 1, gather_indices_3d)[:, :max_valid_len]
        active_cam_x = torch.gather(cam_x_tokens, 1, gather_indices_3d)[:, :max_valid_len]
        active_cam_y = torch.gather(cam_y_tokens, 1, gather_indices_3d)[:, :max_valid_len]
        active_fx = torch.gather(fx_tokens, 1, gather_indices_1d)[:, :max_valid_len]
        active_fy = torch.gather(fy_tokens, 1, gather_indices_1d)[:, :max_valid_len]

        gaussians = gaussians.view(b, max_valid_len, self.K, self.d_per_gaussian)
        gaussians = gaussians.reshape(b, -1, self.d_per_gaussian)

        depth_w, xy_offset, feature, scale, rotation, opacity = torch.split(
            gaussians, [1, self.offset_dim, self.sh_dim, 3, 4, 1], dim=-1
        )

        active_len_k = max_valid_len * self.K

        feature = feature.view(b, active_len_k, (self.cfg.gaussians_params.sh_degree + 1) ** 2, 3).contiguous()
        scale = scale.view(b, active_len_k, 3)
        rotation = rotation.view(b, active_len_k, 4)
        opacity = opacity.view(b, active_len_k, 1)

        def expand_to_K(tensor):
            return tensor.unsqueeze(2).expand(-1, -1, self.K, -1).reshape(b, active_len_k, -1)

        ray_o_k = expand_to_K(active_ray_o)
        ray_d_k = expand_to_K(active_ray_d)
        cam_x_k = expand_to_K(active_cam_x)
        cam_y_k = expand_to_K(active_cam_y)
        fx_k = expand_to_K(active_fx)
        fy_k = expand_to_K(active_fy)

        near_map = context['near'].view(b, v, 1, 1).expand(b, v, hh, ww)
        far_map = context['far'].view(b, v, 1, 1).expand(b, v, hh, ww)
        near_tokens = rearrange(near_map, "b v hh ww -> b (v hh ww)")
        far_tokens = rearrange(far_map, "b v hh ww -> b (v hh ww)")
        near_active = torch.gather(near_tokens, 1, sorted_indices)[:, :max_valid_len].unsqueeze(-1)
        far_active = torch.gather(far_tokens, 1, sorted_indices)[:, :max_valid_len].unsqueeze(-1)

        near_k = expand_to_K(near_active)
        far_k  = expand_to_K(far_active)

        w = depth_w.sigmoid()
        depth = near_k * (1 - w) + far_k * w

        base_xyz = ray_o_k + ray_d_k * depth

        base_px = torch.tensor(0.5 * patch_size, device=depth.device, dtype=depth.dtype)
        wide_px = (self.offset_range.to(depth.dtype) - 0.5).clamp(min=0.0) * patch_size
        max_offset_x = (depth * base_px) / fx_k
        max_offset_y = (depth * base_px) / fy_k

        t_x = xy_offset[..., 0:1].tanh()
        t_y = xy_offset[..., 1:2].tanh()
        offset_val_x = t_x * max_offset_x
        offset_val_y = t_y * max_offset_y

        zxy = offset_raw.reshape(b, active_len_k, 3)
        t_z = zxy[..., 0:1].tanh()
        t_xw = zxy[..., 1:2].tanh()
        t_yw = zxy[..., 2:3].tanh()
        max_offset_z = torch.full_like(depth, self.cfg.offset_z_range)
        offset_val_x = offset_val_x + t_xw * (depth * wide_px) / fx_k
        offset_val_y = offset_val_y + t_yw * (depth * wide_px) / fy_k
        offset_z = t_z * max_offset_z * ray_d_k

        offset_xyz = offset_val_x * cam_x_k + offset_val_y * cam_y_k + offset_z

        xyz = base_xyz + offset_xyz

        scale = torch.exp(scale + self.cfg.gaussians_params.scale_bias).clamp(max=self.cfg.gaussians_params.scale_max)
        opacity = (opacity + self.cfg.gaussians_params.opacity_bias).sigmoid()
        rotation = rotation / (rotation.norm(dim=-1,keepdim=True) + EPS)

        k_mask = attn_mask.unsqueeze(-1).expand(-1, -1, self.K).reshape(b, -1, 1)
        opacity = opacity * k_mask.float()

        gaussian_params = dict(xyz=xyz, scale=scale, opacity=opacity, rotation=rotation, sh_feature=feature)
        gaussians = self.feat2gaussian(gaussian_params)
        gaussian_params.update({"gaussians": gaussians})

        return self.feat2gaussian(gaussian_params)

    def get_data_shim(self) -> DataShim:
        def data_shim(batch: BatchedExample) -> BatchedExample:
            batch = {
                **batch,
                "context": apply_patch_shim_to_views(batch["context"], patch_size=16),
            }

            if self.cfg.apply_bounds_shim:
                _, _, _, h, w = batch["context"]["image"].shape
                near_disparity = self.cfg.near_disparity * min(h, w)
                batch = apply_bounds_shim(batch, near_disparity, 0.5)

            return batch

        return data_shim

    @property
    def sampler(self):

        return self.epipolar_transformer.epipolar_sampler
