from ..types import BatchedExample, BatchedViews

def apply_patch_shim_to_views(views: BatchedViews, patch_size: int) -> BatchedViews:
    _, _, _, h, w = views["image"].shape

    assert h % 2 == 0 and w % 2 == 0

    h_new = (h // patch_size) * patch_size
    row = (h - h_new) // 2
    w_new = (w // patch_size) * patch_size
    col = (w - w_new) // 2

    image = views["image"][:, :, :, row : row + h_new, col : col + w_new]

    intrinsics = views["intrinsics"].clone()
    intrinsics[:, :, 0, 0] *= w / w_new
    intrinsics[:, :, 1, 1] *= h / h_new

    return {
        **views,
        "image": image,
        "intrinsics": intrinsics,
    }

def apply_patch_shim(batch: BatchedExample, patch_size: int) -> BatchedExample:
    return {
        **batch,
        "context": apply_patch_shim_to_views(batch["context"], patch_size),
        "target": apply_patch_shim_to_views(batch["target"], patch_size),
    }
