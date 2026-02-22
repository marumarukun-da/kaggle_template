import logging
import os

import fire
import rootutils

from kaggle_utils.customhub import dataset_upload, model_upload

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ROOT_DIR = rootutils.setup_root(".", indicator="pyproject.toml", cwd=True, dotenv=True)
DATA_DIR = ROOT_DIR / "data"
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"
INPUT_DIR.mkdir(exist_ok=True, parents=True)

KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME")
KAGGLE_COMPETITION_NAME = os.getenv(
    "KAGGLE_COMPETITION_NAME", "{{ cookiecutter.competition_name }}"
)

assert KAGGLE_USERNAME, "KAGGLE_USERNAME is not set."


BASE_ARTIFACTS_HANDLE = f"{KAGGLE_USERNAME}/{KAGGLE_COMPETITION_NAME}-artifacts/other".lower()
CODES_HANDLE = f"{KAGGLE_USERNAME}/{KAGGLE_COMPETITION_NAME}-codes".lower()


def get_latest_version(exp_name: str) -> str:
    """既存の最新バージョン番号を取得する。"""
    exp_dir = OUTPUT_DIR / exp_name
    if not exp_dir.exists():
        return "1"
    existing = [int(d.name) for d in exp_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    return str(max(existing)) if existing else "1"


def upload_codes() -> None:
    """コードを Dataset としてアップロードする。"""
    dataset_upload(
        handle=CODES_HANDLE,
        local_dataset_dir=ROOT_DIR,
        update=True,
    )


def upload_artifacts(exp_name: str, version: str = "latest", update: bool = False) -> None:
    """実験の artifacts を Model としてアップロードする。

    Args:
        exp_name: 実験名（例: "001"）
        version: アップロードするバージョン
            - "latest": 最新のバージョンをアップロード（デフォルト）
            - 数字: 指定したバージョンをアップロード
        update: True の場合、既存の Model Instance を更新する
    """
    if version == "latest":
        version = get_latest_version(exp_name)

    local_model_dir = OUTPUT_DIR / exp_name / version
    if not local_model_dir.exists():
        raise FileNotFoundError(f"Artifact directory not found: {local_model_dir}")

    logger.info(f"Uploading artifacts from: {local_model_dir}")
    logger.info(f"Version: {version}")

    model_upload(
        handle=f"{BASE_ARTIFACTS_HANDLE}/{exp_name}",
        local_model_dir=local_model_dir,
        update=update,
    )


if __name__ == "__main__":
    fire.Fire(
        {
            "codes": upload_codes,
            "artifacts": upload_artifacts,
        }
    )
