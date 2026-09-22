"""
train.py - training entry point with compact loss toggles.
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ["WANDB_CACHE_DIR"] = "/tmp/wandb_cache"

import datetime
import torch

torch.set_float32_matmul_precision("medium")

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb

from dataset import VN3KDataModule
from model import TBPSLightning
from utils import (
    ROOT_DIR,
    BATCH_SIZE,
    MAX_EPOCHS,
    LEARNING_RATE,
    CHECKPOINT_DIR,
    MODEL_NAME,
)


LOSS_CONFIG = {
    "nitc": 1.0,
    "sdm": 1.0,
    "sdm_paper": 0.0,
    "ritc": 0.0,
    "citc": 0.0,
    "ss": 0.0,
    "mvs": 0.0,
    "mtp": 0.5,
}

EXPERIMENT_TAG = "gadms"


def build_trainer(run_name: str, loss_config: dict) -> tuple:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    wandb_logger = WandbLogger(
        project="VN3K_FINAL",
        name=run_name,
        tags=[EXPERIMENT_TAG],
        config={**loss_config},
        save_dir="/tmp",
        log_model=False,
    )

    checkpoint_cb = ModelCheckpoint(
        dirpath=CHECKPOINT_DIR,
        filename="best-r1-epoch{epoch:02d}",
        monitor="val/R@1_t2i",
        mode="max",
        save_top_k=1,
        auto_insert_metric_name=False,
        save_last=False,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="bf16-mixed",
        logger=wandb_logger,
        callbacks=[checkpoint_cb, lr_monitor],
        gradient_clip_val=1.0,
        accumulate_grad_batches=1,
        check_val_every_n_epoch=1,
        log_every_n_steps=20,
        num_sanity_val_steps=0,
        reload_dataloaders_every_n_epochs=1,
    )

    return trainer, checkpoint_cb


def main():
    ts = datetime.datetime.now().strftime("%m%d_%H%M")
    run_name = f"{EXPERIMENT_TAG}_{ts}"
    active_loss_config = dict(LOSS_CONFIG)

    print("\n" + "=" * 60)
    print(f"  Run: {run_name}")
    print(f"  Model: {MODEL_NAME}")
    print("  GASS: ref-faithful single path")
    print(f"  Batch size: {BATCH_SIZE}  |  Max epochs: {MAX_EPOCHS}")
    print(f"  LR: {LEARNING_RATE}")
    print("\n  Loss Configuration:")
    for k, v in active_loss_config.items():
        status = "ON " if v > 0 else "off"
        print(f"    [{status}] {k:<10} = {v:.3f}")
    print("=" * 60 + "\n")

    data_module = VN3KDataModule(
        root_dir=ROOT_DIR,
        preproc_json="vn3k_preproc_texts_final.json",
    )
    data_module.setup()

    model = TBPSLightning(
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        max_epochs=MAX_EPOCHS,
        loss_weights=active_loss_config,
    )

    trainer, checkpoint_cb = build_trainer(run_name, active_loss_config)

    trainer.fit(model, datamodule=data_module)

    best_path = checkpoint_cb.best_model_path
    if best_path and os.path.exists(best_path):
        print(f"\nLoading best model: {best_path}")
        best_model = TBPSLightning.load_from_checkpoint(
            best_path,
            loss_weights=active_loss_config,
        )
        results = trainer.test(best_model, datamodule=data_module)
    else:
        print("\nNo saved checkpoint found, testing current model.")
        results = trainer.test(model, datamodule=data_module)

    if results:
        print("\nFinal Test Results:")
        for k, v in results[0].items():
            print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    try:
        wandb.login()
    except Exception as e:
        print(f"WandB login skipped: {e}")
    main()
