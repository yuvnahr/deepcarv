"""
benchmarks/DepthwiseCNN/scripts/evaluate.py
-------------------------------------------
Evaluation script for the DepthwiseCNN benchmark using the generic DeepCarv framework.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import yaml
import torch

from src.data.dataset import FragmentDataset, build_dataloader
from src.models.registry import build_model, register_model
from src.utils.seed import set_seed
from src.utils.logging import RunLogger
from src.utils.paths import FFT75_DATA_DIR, LOGS_DIR

from benchmarks.DepthwiseCNN.src.adapter import build_depthwisecnn_adapter
from src.training.checkpointing import load_checkpoint
from src.evaluation.evaluator import Evaluator

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DepthwiseCNN benchmark.")
    parser.add_argument("--config", type=Path, required=True, help="Path to benchmark.yaml config.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to the trained checkpoint.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Dataset split to evaluate on.")
    return parser.parse_args()

def main():
    args = _parse_args()
    
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
        
    ds_cfg = cfg.get("dataset", {})
    model_cfg = cfg.get("model", {})
    
    set_seed(42)
    
    # Register the model adapter with the framework
    register_model("depthwisecnn", build_depthwisecnn_adapter, overwrite=True)
    
    data_dir = Path(ds_cfg.get("root_dir", FFT75_DATA_DIR))
    fragment_size = ds_cfg.get("fragment_size", 4096)
    cache = ds_cfg.get("cache", True)
    
    run_logger = RunLogger(run_name="depthwisecnn_eval", base_dir=LOGS_DIR)
    run_logger.info(f"=== DepthwiseCNN Evaluation - Split: {args.split}, Fragment Size: {fragment_size} ===")
    
    # Setup data
    eval_ds = FragmentDataset(root_dir=data_dir, split=args.split, fragment_size=fragment_size, cache=cache)
    batch_size = cfg.get("training", {}).get("batch_size", 256)
    eval_loader = build_dataloader(eval_ds, batch_size=batch_size, shuffle=False)
    
    # Setup model
    model = build_model(
        model_cfg["name"],
        num_classes=eval_ds.num_classes,
        **model_cfg.get("kwargs", {})
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Load checkpoint
    load_checkpoint(args.checkpoint, model, map_location=device)
    
    # Setup evaluator
    evaluator = Evaluator(model, device=device)
    
    # Evaluate
    results = evaluator.evaluate(eval_loader)
    
    run_logger.info(f"Loss: {results.loss:.4f}")
    run_logger.info(f"Accuracy: {results.accuracy:.4f}")
    if results.extra:
        run_logger.info(f"Extra metrics: {results.extra}")
        
    run_logger.close()

if __name__ == "__main__":
    main()
