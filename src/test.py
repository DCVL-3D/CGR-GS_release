from pathlib import Path

import hydra
import torch
from colorama import Fore
from jaxtyping import install_import_hook
from lightning.pytorch import Trainer
from omegaconf import DictConfig

import warnings
warnings.filterwarnings("ignore")

with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.config import load_typed_root_config
    from src.dataset.data_module import DataModule
    from src.misc.LocalLogger import LocalLogger
    from src.model.decoder import get_decoder
    from src.model.encoder import get_encoder
    from src.model.model_wrapper import ModelWrapper

def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"

@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="main",
)
def test(cfg_dict: DictConfig):
    cfg = load_typed_root_config(cfg_dict)

    output_dir = Path(
        hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
    )
    print(cyan(f"Running TEST, outputs to {output_dir}."))

    checkpoint_path = cfg.checkpointing.load
    if checkpoint_path is None or not Path(checkpoint_path).exists():
        raise ValueError("checkpointing.load must point to a checkpoint.")

    trainer = Trainer(
        accelerator="gpu",
        devices=[0],
        logger=LocalLogger(),
        enable_progress_bar=False,
    )
    torch.manual_seed(cfg_dict.seed + trainer.global_rank)

    model_wrapper = ModelWrapper(
        cfg.test,
        get_encoder(cfg.model.encoder),
        get_decoder(cfg.model.decoder, cfg.dataset),
    )
    data_module = DataModule(
        cfg.dataset,
        cfg.data_loader,
        global_rank=trainer.global_rank,
    )

    trainer.test(model_wrapper, datamodule=data_module, ckpt_path=checkpoint_path)

if __name__ == "__main__":
    test()
