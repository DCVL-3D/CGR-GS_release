from dataclasses import dataclass

@dataclass
class DatasetCfgCommon:
    image_shape: list[int]
    background_color: list[float]
