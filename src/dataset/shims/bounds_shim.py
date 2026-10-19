import torch
from einops import einsum, reduce, repeat
from jaxtyping import Float
from torch import Tensor

from ..types import BatchedExample

def compute_depth_for_disparity(
    extrinsics: Float[Tensor, "batch view 4 4"],
    intrinsics: Float[Tensor, "batch view 3 3"],
    image_shape: tuple[int, int],
    disparity: float,
    delta_min: float = 1e-6,
) -> Float[Tensor, " batch"]:

    origins = extrinsics[:, :, :3, 3]
    deltas = (origins[:, None, :, :] - origins[:, :, None, :]).norm(dim=-1)
    deltas = deltas.clip(min=delta_min)
    baselines = reduce(deltas, "b v ov -> b", "max")

    h, w = image_shape
    pixel_size = 1 / torch.tensor((w, h), dtype=torch.float32, device=extrinsics.device)
    pixel_size = einsum(
        intrinsics[..., :2, :2].inverse(), pixel_size, "... i j, j -> ... i"
    )

    mean_pixel_size = reduce(pixel_size, "b v xy -> b", "mean")

    return baselines / (disparity * mean_pixel_size)

def apply_bounds_shim(
    batch: BatchedExample,
    near_disparity: float,
    far_disparity: float,
) -> BatchedExample:

    context = batch["context"]
    _, cv, _, h, w = context["image"].shape

    near = compute_depth_for_disparity(
        context["extrinsics"],
        context["intrinsics"],
        (h, w),
        near_disparity,
    )
    far = compute_depth_for_disparity(
        context["extrinsics"],
        context["intrinsics"],
        (h, w),
        far_disparity,
    )

    target = batch["target"]
    _, tv, _, _, _ = target["image"].shape
    return {
        **batch,
        "context": {
            **context,
            "near": repeat(near, "b -> b v", v=cv),
            "far": repeat(far, "b -> b v", v=cv),
        },
        "target": {
            **target,
            "near": repeat(near, "b -> b v", v=tv),
            "far": repeat(far, "b -> b v", v=tv),
        },
    }
