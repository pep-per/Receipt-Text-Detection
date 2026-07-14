import os
import sys
from pathlib import Path

import lightning.pytorch as pl
import hydra
import torch
from lightning.pytorch.callbacks import (  # noqa
    LearningRateMonitor,
    ModelCheckpoint,
)

sys.path.append(os.getcwd())
from ocr.lightning_modules import get_pl_modules_by_cfg  # noqa: E402

CONFIG_DIR = os.environ.get('OP_CONFIG_DIR') or '../configs'


def load_encoder_initialization(model_module, checkpoint_path):
    """Load an image-only pretrained timm encoder before detector training."""
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Encoder checkpoint does not exist: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("encoder_state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError("Encoder checkpoint must contain a state dictionary")

    prefix = "backbone."
    if state_dict and all(key.startswith(prefix) for key in state_dict):
        state_dict = {key[len(prefix):]: value for key, value in state_dict.items()}

    encoder = model_module.model.encoder.model
    incompatible = encoder.load_state_dict(state_dict, strict=True)
    print(
        f"Loaded encoder initialization from {checkpoint_path} "
        f"({len(state_dict)} tensors, missing={len(incompatible.missing_keys)}, "
        f"unexpected={len(incompatible.unexpected_keys)})"
    )


@hydra.main(config_path=CONFIG_DIR, config_name='train', version_base='1.2')
def train(config):
    """
    Train a OCR model using the provided configuration.

    Args:
        `config` (dict): A dictionary containing configuration settings for training.
    """
    pl.seed_everything(config.get("seed", 42), workers=True)

    model_module, data_module = get_pl_modules_by_cfg(config)

    encoder_init_path = config.get("encoder_init_path")
    if encoder_init_path:
        load_encoder_initialization(model_module, encoder_init_path)

    if config.get("wandb"):
        from lightning.pytorch.loggers import WandbLogger as Logger  # noqa: E402
        logger = Logger(config.exp_name,
                        project=config.project_name,
                        config=dict(config),
                        )
    else:
        from lightning.pytorch.loggers.tensorboard import TensorBoardLogger  # noqa: E402
        logger = TensorBoardLogger(
            save_dir=config.log_dir,
            name=config.exp_name,
            version=config.exp_version,
            default_hp_metric=False,
        )

    checkpoint_path = config.checkpoint_dir

    callbacks = [
        LearningRateMonitor(logging_interval='step'),
        ModelCheckpoint(dirpath=checkpoint_path,
                        save_top_k=3,
                        save_last=True,
                        monitor='val/hmean',
                        mode='max'),
    ]

    trainer = pl.Trainer(
        **config.trainer,
        logger=logger,
        callbacks=callbacks
    )

    trainer.fit(
        model_module,
        data_module,
        ckpt_path=config.get("resume", None),
    )
    trainer.test(
        model_module,
        data_module,
    )


if __name__ == "__main__":
    train()
