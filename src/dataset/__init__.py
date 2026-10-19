from torch.utils.data import Dataset

from .dataset_Thuman import DatasetThuman, DatasetThumanCfg

DATASETS: dict[str, type[Dataset]] = {
    "thuman": DatasetThuman,
}

DatasetCfg = DatasetThumanCfg

def get_dataset(cfg: DatasetCfg) -> Dataset:
    return DATASETS[cfg.name](cfg)
