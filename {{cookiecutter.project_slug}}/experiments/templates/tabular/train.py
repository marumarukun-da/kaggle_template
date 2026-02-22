"""Main training script for tabular data (LightGBM)."""

import argparse
import gc
import pickle
import warnings

import config
import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from metric import score
from seed import seed_everything
from sklearn.model_selection import KFold, StratifiedKFold

warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser(description="Tabular data training")
    parser.add_argument("--fold", type=int, default=None, help="Run a specific fold only (1-indexed)")
    parser.add_argument("--debug", action="store_true", help="Debug mode (fewer boost rounds)")
    args = parser.parse_args()

    # =============================================================================
    # Configuration
    # =============================================================================
    CFG = config.CFG

    if args.debug:
        CFG.NUM_BOOST_ROUND = 100
        CFG.EARLY_STOPPING_ROUND = 10

    seed_everything(CFG.SEED)
    CFG.MODEL_PATH.mkdir(parents=True, exist_ok=True)

    # =============================================================================
    # Load Data
    # =============================================================================
    train = pl.read_csv(CFG.DATA_PATH / "train.csv")
    test = pl.read_csv(CFG.DATA_PATH / "test.csv")

    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")

    # =============================================================================
    # Feature Engineering
    # =============================================================================
    # TODO: Add your feature engineering here

    # Define features
    FEATURES = [col for col in train.columns if col not in ["id", CFG.TARGET_COL]]
    print(f"Number of features: {len(FEATURES)}")

    # =============================================================================
    # Cross-Validation Training
    # =============================================================================
    train_pd = train.to_pandas()
    test_pd = test.to_pandas()

    oof_predictions = np.zeros(len(train_pd))
    test_predictions = np.zeros(len(test_pd))

    kf = KFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)
    # For classification: kf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)

    folds_iter = list(enumerate(kf.split(train_pd), 1))
    if args.fold is not None:
        folds_iter = [folds_iter[args.fold - 1]]

    for fold, (train_idx, val_idx) in folds_iter:
        print(f"\n{'='*50}")
        print(f"Fold {fold}")
        print(f"{'='*50}")

        X_train = train_pd.loc[train_idx, FEATURES]
        y_train = train_pd.loc[train_idx, CFG.TARGET_COL]
        X_val = train_pd.loc[val_idx, FEATURES]
        y_val = train_pd.loc[val_idx, CFG.TARGET_COL]

        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        # Train model
        model = lgb.train(
            CFG.lgb_params,
            train_data,
            num_boost_round=CFG.NUM_BOOST_ROUND,
            valid_sets=[train_data, val_data],
            valid_names=["train", "valid"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=CFG.EARLY_STOPPING_ROUND),
                lgb.log_evaluation(500),
            ],
        )

        # Predict
        oof_predictions[val_idx] = model.predict(X_val)
        test_predictions += model.predict(test_pd[FEATURES]) / CFG.N_FOLDS

        # Save model
        model_path = CFG.MODEL_PATH / f"lgb_fold{fold}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        # Fold score
        fold_score = score(y_val.values, oof_predictions[val_idx])
        print(f"Fold {fold} Score: {fold_score:.6f}")

        del X_train, y_train, X_val, y_val, model
        gc.collect()

    # Overall CV score
    cv_score = score(train_pd[CFG.TARGET_COL].values, oof_predictions)
    print(f"\n{'='*50}")
    print(f"Overall CV Score: {cv_score:.6f}")
    print(f"{'='*50}")

    # =============================================================================
    # Save OOF and Test Predictions
    # =============================================================================
    # Save OOF predictions
    oof_df = pd.DataFrame({
        "id": train_pd["id"],
        "oof_pred": oof_predictions,
        "true": train_pd[CFG.TARGET_COL],
    })
    oof_df.to_csv(config.OUTPUT_DIR / "oof_predictions.csv", index=False)

    # Save test predictions (for local use)
    test_pred_df = pd.DataFrame({
        "id": test_pd["id"],
        "pred": test_predictions,
    })
    test_pred_df.to_csv(config.OUTPUT_DIR / "test_predictions.csv", index=False)

    print(f"OOF predictions saved to: {config.OUTPUT_DIR / 'oof_predictions.csv'}")
    print(f"Test predictions saved to: {config.OUTPUT_DIR / 'test_predictions.csv'}")

    # =============================================================================
    # Create Submission (for local testing)
    # =============================================================================
    sub_df = pl.read_csv(CFG.DATA_PATH / "sample_submission.csv")
    # TODO: Update with your target column and predictions
    # sub_df = sub_df.with_columns(pl.Series(CFG.TARGET_COL, test_predictions))
    sub_df.write_csv(config.OUTPUT_DIR / "submission.csv")

    print(f"Submission saved to: {config.OUTPUT_DIR / 'submission.csv'}")
    print(sub_df.head())


if __name__ == "__main__":
    main()
