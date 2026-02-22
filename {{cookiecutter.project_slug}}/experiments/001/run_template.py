"""Comprehensive ensemble training template (LightGBM / XGBoost / CatBoost)."""

import argparse
import gc
import itertools
import os
import pickle
import random
import sys
import warnings
from glob import glob
from pathlib import Path

import config
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
import scipy as sp
import torch
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from lightgbm import LGBMClassifier, LGBMRegressor
from metric import score
from scipy.optimize import minimize
from seed import seed_everything
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, TimeSeriesSplit, train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings("ignore")


# ====================================================
# Configurations
# ====================================================
class CFG:
    DRY_RUN = True
    EXP_NAME = config.EXP_NAME
    AUTHOR = "marumarukun"
    COMPETITION = config.KAGGLE_COMPETITION_NAME
    DATA_PATH = config.COMP_DATASET_DIR
    OUTPUT_DIR = config.OUTPUT_DIR
    MODEL_PATH = config.OUTPUT_DIR / "models"  # モデル作成・実験時はこちらを使用
    # MODEL_PATH = config.ARTIFACT_EXP_DIR(config.EXP_NAME) / "models"  # 提出時はこちらを使用
    METHOD_LIST = ["lightgbm", "xgboost", "catboost"]
    METHOD_WEIGHT_DICT = {"lightgbm": 0.7, "xgboost": 0.2, "catboost": 0.1}
    SEED = 319
    n_folds = 2 if DRY_RUN else 5
    target_col_list = ["target"]
    # group_col = "category1"  # Required for GroupKFold (edit as needed)
    stratified_col = "target"  # Required for StratifiedKFold (edit as needed)
    num_boost_round = 100 if DRY_RUN else 1000000
    early_stopping_round = 10 if DRY_RUN else 100  # 10÷lrで設定
    verbose = 500

    # https://lightgbm.readthedocs.io/en/latest/Parameters.html
    regression_lgb_params = {
        "objective": "regression",
        # "metric": "mae",
        "learning_rate": 0.1,
        "max_depth": 5,
        "min_child_weight": 1,
        "colsample_bytree": 0.8,
        "subsample": 0.8,
        "subsample_freq": 1,
        "seed": SEED,
        "device": "cpu",  # cpu/gpu/cuda
        "verbosity": -1,
    }
    # https://xgboost.readthedocs.io/en/stable/parameter.html
    regression_xgb_params = {
        "objective": "reg:squarederror",
        # "eval_metric": "mae",
        "learning_rate": 0.1,
        "max_depth": 5,
        "colsample_bytree": 0.8,
        "subsample": 0.8,
        "min_child_weight": 1,
        "enable_categorical": True,
        "random_state": SEED,
        "device": "cpu",  # cpu/gpu/cuda
    }
    # https://catboost.ai/docs/en/references/training-parameters/
    regression_cat_params = {
        "loss_function": "RMSE",
        "learning_rate": 0.1,
        "iterations": 100 if DRY_RUN else 1000000,
        # "depth": 5,
        "grow_policy": "Lossguide",
        "random_seed": SEED,
        "task_type": "CPU",  # CPU/GPU
    }


# ====================================================
# Training functions
# ====================================================
def lightgbm_training(
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    x_valid: pd.DataFrame,
    y_valid: pd.DataFrame,
    categorical_features: list,
):
    model = LGBMRegressor(
        **CFG.regression_lgb_params,
        n_estimators=CFG.num_boost_round,
    )
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_valid, y_valid)],
        categorical_feature=categorical_features,
        callbacks=[
            lgb.early_stopping(stopping_rounds=CFG.early_stopping_round),
            lgb.log_evaluation(CFG.verbose),
        ],
    )
    valid_pred = model.predict(x_valid)
    return model, valid_pred


def xgboost_training(
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    x_valid: pd.DataFrame,
    y_valid: pd.DataFrame,
):
    model = XGBRegressor(
        **CFG.regression_xgb_params,
        n_estimators=CFG.num_boost_round,
        early_stopping_rounds=CFG.early_stopping_round,
    )
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_valid, y_valid)],
        verbose=CFG.verbose,
    )
    valid_pred = model.predict(x_valid)
    return model, valid_pred


def catboost_training(
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    x_valid: pd.DataFrame,
    y_valid: pd.DataFrame,
    categorical_features: list,
):
    cat_train = Pool(data=x_train, label=y_train, cat_features=categorical_features)
    cat_valid = Pool(data=x_valid, label=y_valid, cat_features=categorical_features)
    model = CatBoostRegressor(**CFG.regression_cat_params)
    model.fit(
        cat_train,
        eval_set=[cat_valid],
        early_stopping_rounds=CFG.early_stopping_round,
        verbose=CFG.verbose,
        use_best_model=True,
    )
    valid_pred = model.predict(x_valid)
    return model, valid_pred


def gradient_boosting_model_cv_training(
    method: str, train_df: pd.DataFrame, target_col_list: list, features: list, categorical_features: list
):
    oof_predictions_df = pd.DataFrame(np.zeros((len(train_df), len(target_col_list))), columns=target_col_list)

    for target_col in target_col_list:
        oof_predictions = np.zeros(len(train_df))
        for fold in range(CFG.n_folds):
            print("-" * 50)
            print(f"{method} training fold {fold+1} {target_col}")
            x_train = train_df[train_df["fold"] != fold + 1][features]
            y_train = train_df[train_df["fold"] != fold + 1][target_col]
            x_valid = train_df[train_df["fold"] == fold + 1][features]
            y_valid = train_df[train_df["fold"] == fold + 1][target_col]
            if method == "lightgbm":
                model, valid_pred = lightgbm_training(x_train, y_train, x_valid, y_valid, categorical_features)
            elif method == "xgboost":
                model, valid_pred = xgboost_training(x_train, y_train, x_valid, y_valid)
            elif method == "catboost":
                model, valid_pred = catboost_training(x_train, y_train, x_valid, y_valid, categorical_features)
            else:
                raise ValueError(f"Unknown method: {method}")

            # Save best model
            save_model_path = (
                CFG.MODEL_PATH / f"{method}_{target_col}_fold{fold + 1}_seed{CFG.SEED}_ver{CFG.EXP_NAME}.pkl"
            )
            save_model_path.parent.mkdir(parents=True, exist_ok=True)
            pickle.dump(model, open(save_model_path, "wb"))
            # Add to out of folds array
            oof_predictions[train_df["fold"] == fold + 1] = valid_pred
            del x_train, x_valid, y_train, y_valid, model, valid_pred
            gc.collect()

        oof_predictions_df[target_col] = oof_predictions

        # Compute out of folds metric
        m = score(train_df[target_col].copy(), oof_predictions_df[target_col].copy())
        print("=" * 50)
        print(f"{method} our out of folds CV score is {m}")
        print("=" * 50)

    oof_predictions_df.to_csv(CFG.OUTPUT_DIR / f"oof_{method}_seed{CFG.SEED}_ver{CFG.EXP_NAME}.csv", index=False)


# ====================================================
# Inference functions
# ====================================================
def model_inference(method: str, x_test: pd.DataFrame, target_col: str):
    test_pred = np.zeros(len(x_test))
    for fold in range(CFG.n_folds):
        model_path = CFG.MODEL_PATH / f"{method}_{target_col}_fold{fold + 1}_seed{CFG.SEED}_ver{CFG.EXP_NAME}.pkl"
        model = pickle.load(open(model_path, "rb"))
        pred = model.predict(x_test)
        test_pred += pred
    return test_pred / CFG.n_folds


def gradient_boosting_model_inference(method: str, test_df: pd.DataFrame, features: list, target_col: str):
    x_test = test_df[features]
    return model_inference(method, x_test, target_col)


def predicting(input_df: pd.DataFrame, features: list):
    output_df = input_df.copy()
    for target_col in CFG.target_col_list:
        output_df[target_col] = 0
        for method in CFG.METHOD_LIST:
            output_df[f"{method}_pred_{target_col}"] = gradient_boosting_model_inference(
                method, input_df, features, target_col
            )
            output_df[target_col] += CFG.METHOD_WEIGHT_DICT[method] * output_df[f"{method}_pred_{target_col}"]
    return output_df


def main():
    parser = argparse.ArgumentParser(description="Ensemble training (LightGBM / XGBoost / CatBoost)")
    parser.add_argument("--debug", action="store_true", help="Debug mode (fewer boost rounds)")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        choices=["lightgbm", "xgboost", "catboost"],
        help="Methods to use (default: all)",
    )
    args = parser.parse_args()

    if args.debug:
        CFG.DRY_RUN = True
        CFG.n_folds = 2
        CFG.num_boost_round = 100
        CFG.early_stopping_round = 10
        CFG.regression_cat_params["iterations"] = 100

    if args.methods is not None:
        CFG.METHOD_LIST = args.methods

    # ====================================================
    # Seed everything
    # ====================================================
    seed_everything(CFG.SEED)

    # ====================================================
    # Read data
    # ====================================================
    train_pl = pl.read_csv(CFG.DATA_PATH / "train_demo.csv", try_parse_dates=True)
    test_pl = pl.read_csv(CFG.DATA_PATH / "test_demo.csv", try_parse_dates=True)

    # ====================================================
    # Make fold column
    # ====================================================
    # StratifiedKFold
    fold_array = np.zeros(train_pl.height)
    skf = StratifiedKFold(n_splits=CFG.n_folds, shuffle=True, random_state=CFG.SEED)
    for fold, (_, val_idx) in enumerate(skf.split(train_pl, train_pl[CFG.stratified_col]), start=1):
        fold_array[val_idx] = fold
    train_pl = train_pl.with_columns(pl.Series(fold_array, dtype=pl.Int8).alias("fold"))

    # ====================================================
    # Define columns and Label Encode categorical columns
    # ====================================================
    train = train_pl.to_pandas()
    test = test_pl.to_pandas()

    RMV = ["id", "fold", "target"]
    FEATURES = [c for c in train.columns if c not in RMV]
    print(f"There are {len(FEATURES)} FEATURES: {FEATURES}")

    CATS = []
    for c in FEATURES:
        if train[c].dtype == "object":
            CATS.append(c)
            train[c] = train[c].fillna("NAN")
            test[c] = test[c].fillna("NAN")
    print(f"In these features, there are {len(CATS)} CATEGORICAL FEATURES: {CATS}")

    combined = pd.concat([train, test], axis=0, ignore_index=True)

    # LABEL ENCODE CATEGORICAL FEATURES
    print("We LABEL ENCODE the CATEGORICAL FEATURES: ", end="")
    for c in FEATURES:
        if c in CATS:
            print(f"{c}, ", end="")
            combined[c], _ = combined[c].factorize()
            combined[c] -= combined[c].min()
            combined[c] = combined[c].astype("int32")
            combined[c] = combined[c].astype("category")
        else:
            if combined[c].dtype == "float64":
                combined[c] = combined[c].astype("float32")
            if combined[c].dtype == "int64":
                combined[c] = combined[c].astype("int32")
    print()

    train = combined.iloc[: len(train)].copy()
    test = combined.iloc[len(train) :].reset_index(drop=True).copy()

    # ====================================================
    # Training
    # ====================================================
    for method in CFG.METHOD_LIST:
        gradient_boosting_model_cv_training(method, train, CFG.target_col_list, FEATURES, CATS)

    # ====================================================
    # Inference
    # ====================================================
    output_df = predicting(test, FEATURES)
    print(output_df)


if __name__ == "__main__":
    main()
