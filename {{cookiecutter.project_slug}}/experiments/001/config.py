import os
from pathlib import Path

EXP_NAME = Path(__file__).parent.name


# ---------- # VERSION MANAGEMENT # ---------- #
def get_latest_version(artifact_dir: Path, exp_name: str) -> str:
    """既存の最新バージョン番号を取得する。存在しない場合は "1" を返す。"""
    exp_dir = artifact_dir / exp_name
    if not exp_dir.exists():
        return "1"
    existing = [int(d.name) for d in exp_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    return str(max(existing)) if existing else "1"


def get_next_version(artifact_dir: Path, exp_name: str) -> str:
    """次のバージョン番号を取得する（既存の最大値 + 1）。"""
    exp_dir = artifact_dir / exp_name
    if not exp_dir.exists():
        return "1"
    existing = [int(d.name) for d in exp_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    return str(max(existing) + 1) if existing else "1"


# ---------- # DIRECTORIES # ---------- #
IS_KAGGLE_ENV = os.getenv("KAGGLE_DATA_PROXY_TOKEN") is not None
KAGGLE_COMPETITION_NAME = os.getenv(
    "KAGGLE_COMPETITION_NAME", "{{ cookiecutter.competition_name }}"
)

if not IS_KAGGLE_ENV:
    import rootutils

    ROOT_DIR = rootutils.setup_root(
        ".",
        indicator="pyproject.toml",
        cwd=True,
        pythonpath=True,
    )
    INPUT_DIR = ROOT_DIR / "data" / "input"
    ARTIFACT_DIR = ROOT_DIR / "data" / "output"

    # バージョン管理:
    #   - EXP_VERSION 環境変数で明示的に指定可能
    #   - "latest": 最新の既存バージョンを使用（推論時など）
    #   - "next": 次のバージョンを自動生成（新規学習時）
    #   - 数字: 指定したバージョンを使用
    #   - 未指定: デフォルトは "1"
    _version_input = os.getenv("EXP_VERSION", "1")
    if _version_input == "latest":
        VERSION = get_latest_version(ARTIFACT_DIR, EXP_NAME)
    elif _version_input == "next":
        VERSION = get_next_version(ARTIFACT_DIR, EXP_NAME)
    else:
        VERSION = _version_input

    OUTPUT_DIR = ARTIFACT_DIR / EXP_NAME / VERSION

    KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME", "{{ cookiecutter.kaggle_username }}")
    ARTIFACTS_HANDLE = (
        f"{KAGGLE_USERNAME}/{KAGGLE_COMPETITION_NAME}-artifacts/other/{EXP_NAME}"
    )
    CODES_HANDLE = f"{KAGGLE_USERNAME}/{KAGGLE_COMPETITION_NAME}-codes"
else:
    ROOT_DIR = Path("/kaggle/working")
    INPUT_DIR = Path("/kaggle/input")
    ARTIFACT_DIR = INPUT_DIR / f"{KAGGLE_COMPETITION_NAME}-artifacts".lower() / "other"
    OUTPUT_DIR = ROOT_DIR  # Kaggle環境では /kaggle/working に出力
    VERSION = os.getenv("EXP_VERSION", "1")

COMP_DATASET_DIR = INPUT_DIR / KAGGLE_COMPETITION_NAME

for d in [INPUT_DIR, OUTPUT_DIR]:
    d.mkdir(exist_ok=True, parents=True)


def ARTIFACT_EXP_DIR(exp_name: str, version: str = VERSION) -> Path:
    """対象の実験の artifact が格納されている場所を返す。"""
    return ARTIFACT_DIR / exp_name / version


# ---------- # TABULAR CONFIG # ---------- #
class CFG:
    # General
    SEED = 42
    N_FOLDS = 5
    TARGET_COL = "target"  # Update with your target column

    # Paths
    DATA_PATH = COMP_DATASET_DIR
    OUTPUT_PATH = OUTPUT_DIR
    MODEL_PATH = OUTPUT_DIR / "models"

    # LightGBM parameters
    lgb_params = {
        "objective": "regression",  # or "binary", "multiclass"
        "metric": "rmse",
        "learning_rate": 0.05,
        "max_depth": 6,
        "num_leaves": 31,
        "colsample_bytree": 0.8,
        "subsample": 0.8,
        "seed": SEED,
        "verbosity": -1,
    }

    NUM_BOOST_ROUND = 10000
    EARLY_STOPPING_ROUND = 100
