from typing import Callable, TypedDict

from jaxtyping import Float, Int64
from torch import Tensor

class BatchedViews(TypedDict, total=False):
    extrinsics: Float[Tensor, "batch _ 4 4"]
    intrinsics: Float[Tensor, "batch _ 3 3"]
    image: Float[Tensor, "batch _ _ _ _"]
    near: Float[Tensor, "batch _"]
    far: Float[Tensor, "batch _"]
    index: Int64[Tensor, "batch _"]

class BatchedExample(TypedDict, total=False):
    target: BatchedViews
    context: BatchedViews
    scene: list[str]

DataShim = Callable[[BatchedExample], BatchedExample]
