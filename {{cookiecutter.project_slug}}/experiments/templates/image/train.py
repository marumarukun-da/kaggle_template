"""Main training script for image classification."""

import argparse
import gc
import os
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

import config
from augmentation import get_train_transforms, get_valid_transforms
from dataset import create_dataloader
from model import RECOMMENDED_MODELS, create_model
from trainer import train_fold

warnings.filterwarnings("ignore")


# =============================================================================
# Seed Everything
# =============================================================================
def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description="Image classification training")
    parser.add_argument("--fold", type=int, default=None, help="Run a specific fold only (1-indexed)")
    parser.add_argument("--debug", action="store_true", help="Debug mode (fewer epochs)")
    args = parser.parse_args()

    # =============================================================================
    # Configuration (override from config.py if needed)
    # =============================================================================
    CFG = config.CFG

    if args.debug:
        CFG.EPOCHS = 1

    print(f"Model: {CFG.MODEL_NAME}")
    print(f"Image Size: {CFG.IMG_SIZE}")
    print(f"Batch Size: {CFG.BATCH_SIZE}")
    print(f"Epochs: {CFG.EPOCHS}")
    print(f"Learning Rate: {CFG.LEARNING_RATE}")

    seed_everything(CFG.SEED)

    # Device
    device = CFG.DEVICE if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Create output directory
    CFG.MODEL_PATH.mkdir(parents=True, exist_ok=True)

    # =============================================================================
    # Load Data
    # =============================================================================
    # TODO: Update with your data loading logic
    train_df = pd.read_csv(CFG.DATA_PATH / "train.csv")

    print(f"Train samples: {len(train_df)}")
    print(f"Columns: {train_df.columns.tolist()}")
    print(f"\nLabel distribution:")
    print(train_df[CFG.TARGET_COL].value_counts())

    # =============================================================================
    # Create Folds
    # =============================================================================
    skf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)

    train_df["fold"] = -1
    for fold, (_, val_idx) in enumerate(skf.split(train_df, train_df[CFG.TARGET_COL])):
        train_df.loc[val_idx, "fold"] = fold

    print(f"Fold distribution:")
    print(train_df["fold"].value_counts().sort_index())

    # =============================================================================
    # Transforms
    # =============================================================================
    train_transforms = get_train_transforms(CFG.IMG_SIZE)
    valid_transforms = get_valid_transforms(CFG.IMG_SIZE)

    print("Train transforms:")
    print(train_transforms)
    print("\nValid transforms:")
    print(valid_transforms)

    # =============================================================================
    # Training Loop
    # =============================================================================
    # TODO: Update image_dir with your image directory
    image_dir = CFG.DATA_PATH / "images" / "train"

    folds = [args.fold - 1] if args.fold is not None else range(CFG.N_FOLDS)
    all_histories = []

    for fold in folds:
        print(f"\n{'='*60}")
        print(f"FOLD {fold + 1}/{CFG.N_FOLDS}")
        print(f"{'='*60}")

        # Split data
        train_fold_df = train_df[train_df["fold"] != fold].reset_index(drop=True)
        valid_fold_df = train_df[train_df["fold"] == fold].reset_index(drop=True)

        print(f"Train samples: {len(train_fold_df)}")
        print(f"Valid samples: {len(valid_fold_df)}")

        # Create dataloaders
        train_loader = create_dataloader(
            df=train_fold_df,
            image_dir=image_dir,
            transforms=train_transforms,
            batch_size=CFG.BATCH_SIZE,
            num_workers=CFG.NUM_WORKERS,
            is_train=True,
            label_col=CFG.TARGET_COL,
        )

        valid_loader = create_dataloader(
            df=valid_fold_df,
            image_dir=image_dir,
            transforms=valid_transforms,
            batch_size=CFG.BATCH_SIZE * 2,
            num_workers=CFG.NUM_WORKERS,
            is_train=True,  # Need labels for validation
            label_col=CFG.TARGET_COL,
        )

        # Create model
        model = create_model(
            model_name=CFG.MODEL_NAME,
            num_classes=CFG.NUM_CLASSES,
            pretrained=CFG.PRETRAINED,
            device=device,
        )

        # Train
        history = train_fold(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            epochs=CFG.EPOCHS,
            lr=CFG.LEARNING_RATE,
            weight_decay=CFG.WEIGHT_DECAY,
            scheduler_type=CFG.SCHEDULER,
            device=device,
            use_amp=CFG.USE_AMP,
            save_path=CFG.MODEL_PATH,
            fold=fold + 1,
        )

        all_histories.append(history)

        # Cleanup
        del model, train_loader, valid_loader
        gc.collect()
        torch.cuda.empty_cache()

    # =============================================================================
    # Training Summary
    # =============================================================================
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)

    for i, history in enumerate(all_histories):
        fold_num = (args.fold if args.fold is not None else i + 1)
        best_epoch = np.argmax(history["valid_acc"])
        best_acc = history["valid_acc"][best_epoch]
        print(f"Fold {fold_num}: Best Acc = {best_acc:.2f}% (Epoch {best_epoch + 1})")

    # Average best accuracy
    avg_acc = np.mean([max(h["valid_acc"]) for h in all_histories])
    print(f"\nAverage Best Accuracy: {avg_acc:.2f}%")

    # =============================================================================
    # Available Models (for reference)
    # =============================================================================
    print("\nRecommended models for Kaggle competitions:")
    for model_name in RECOMMENDED_MODELS:
        print(f"  - {model_name}")


if __name__ == "__main__":
    main()
