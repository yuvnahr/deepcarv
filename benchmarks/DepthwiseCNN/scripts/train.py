"""
benchmarks/DepthwiseCNN/scripts/train.py
----------------------------------------
Training script for the DepthwiseCNN benchmark using the generic DeepCarv framework.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import yaml

from src.data.dataset import FragmentDataset, build_dataloader
from src.training.trainer import Trainer, TrainerConfig
from src.models.registry import build_model, register_model
from src.utils.seed import set_seed
from src.utils.logging import RunLogger
from src.utils.paths import ensure_dirs, FFT75_DATA_DIR, LOGS_DIR

from benchmarks.DepthwiseCNN.src.adapter import build_depthwisecnn_adapter

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DepthwiseCNN benchmark.")
    parser.add_argument("--config", type=Path, required=True, help="Path to benchmark.yaml config.")
    return parser.parse_args()

def main():
    args = _parse_args()
    
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
        
    train_cfg = cfg.get("training", {})
    ds_cfg = cfg.get("dataset", {})
    path_cfg = cfg.get("paths", {})
    model_cfg = cfg.get("model", {})
    
    seed = train_cfg.get("seed", 42)
    set_seed(seed)
    
    # Register the model adapter with the framework
    register_model("depthwisecnn", build_depthwisecnn_adapter, overwrite=True)
    
    data_dir = Path(ds_cfg.get("root_dir", FFT75_DATA_DIR))
    fragment_size = ds_cfg.get("fragment_size", 4096)
    cache = ds_cfg.get("cache", True)
    
    run_outputs = Path(path_cfg.get("run_outputs", "outputs/DepthwiseCNN"))
    ensure_dirs(run_outputs, LOGS_DIR)
    
    run_logger = RunLogger(run_name="depthwisecnn_fft75", base_dir=LOGS_DIR, config_path=args.config)
    run_logger.info(f"=== DepthwiseCNN Training - Fragment Size {fragment_size} ===")
    
    # Setup data
    train_ds = FragmentDataset(root_dir=data_dir, split="train", fragment_size=fragment_size, cache=cache)
    val_ds = FragmentDataset(root_dir=data_dir, split="val", fragment_size=fragment_size, cache=cache)
    
    batch_size = train_cfg.get("batch_size", 256)
    train_loader = build_dataloader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = build_dataloader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    
    # Setup model
    model = build_model(
        model_cfg["name"],
        num_classes=train_ds.num_classes,
        **model_cfg.get("kwargs", {})
    )
    
    run_logger.info(f"Model {model.name} initialized with {model.num_parameters()} parameters.")
    
    # Setup trainer
    trainer_config = TrainerConfig.from_dict(train_cfg)
    trainer = Trainer(model, trainer_config, run_outputs)
    
    # Train
    history = trainer.fit(train_loader, val_loader)
    
    run_logger.info(f"=== Training complete. Best val_acc={trainer._best_metric:.4f} ===")
    run_logger.close()

if __name__ == "__main__":
    main()
