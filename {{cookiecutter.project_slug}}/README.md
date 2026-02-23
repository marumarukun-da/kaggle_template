# {{ cookiecutter.project_name }}

## Quick Start Checklist

初めてこのプロジェクトを使う場合は、以下のチェックリストに従ってセットアップしてください。

- [ ] `.env.sample` をコピーして `.env` を作成
- [ ] `.env` に `KAGGLE_USERNAME` と `KAGGLE_KEY` を設定
- [ ] Docker コンテナを起動 (`docker compose up -d --build gpu` or `cpu`)
- [ ] VSCode「コンテナーで再度開く」でコンテナに接続
- [ ] 推奨拡張機能をインストール
- [ ] Kaggle上でコンペティションへの参加登録
- [ ] `sh scripts/download_competition.sh` でデータをダウンロード

---

## Setup

<details>
<summary>1. 環境変数の設定</summary>

`.env.sample`をコピーして`.env`を作成し、Kaggle認証情報を設定します：

```bash
cp .env.sample .env
```

`.env` ファイルを編集:
```bash
KAGGLE_USERNAME=your_username  # あなたのKaggleユーザー名
KAGGLE_KEY=your_key           # あなたのKaggle API key
```

> **Kaggle API Key の取得方法**:
> 1. [Kaggle](https://www.kaggle.com) にログイン
> 2. 右上のアイコン → Settings → API → Create New Token
> 3. ダウンロードされた `kaggle.json` から値をコピー

</details>

<details>
<summary>2. Docker環境の起動</summary>

GPU環境の場合：
```bash
docker compose up -d --build gpu
```

CPU環境の場合：
```bash
docker compose up -d --build cpu
```

</details>

<details>
<summary>3. VSCode接続と拡張機能</summary>

1. VSCodeの左下の「><」アイコンをクリック
2. 「コンテナーで再度開く」を選択してコンテナに接続
3. 推奨される拡張機能をインストール：
   - 右下に表示される「推奨される拡張機能をインストール」の通知をクリック
   - または、拡張機能タブ（Ctrl+Shift+X）から「推奨」セクションを確認

</details>

---

## Download Competition Dataset

```bash
# 1. Kaggle上でコンペティションへの参加登録を行う（必須）

# 2. データセットをダウンロード
sh scripts/download_competition.sh
```

---

## Submission Flow

### Step 1: 実験フォルダを作成

```bash
# テンプレートから作成
sh scripts/new_exp.sh                    # tabularテンプレート → experiments/002/
sh scripts/new_exp.sh --template image   # 画像系テンプレート

# 過去の実験をベースに作成
sh scripts/new_exp.sh --base 001               # 001をコピー → experiments/002/
sh scripts/new_exp.sh --base 002 --name 005    # 002をコピー → experiments/005/
```

作成されるファイル：
| ファイル | 役割 | 編集が必要か |
|---------|------|------------|
| `config.py` | パス設定（自動で設定される） | 基本的に不要 |
| `train.py` | 学習用スクリプト | **必須** |
| `inference.py` | Kaggle上で実行される推論コード | **必須** |

### Step 2: 実験を行う

`experiments/001/train.py` を編集・実行してモデルを学習します。

#### バージョン管理について

同じ実験でパラメータを変えて複数回学習する場合、`EXP_VERSION` 環境変数でバージョンを管理できます。

```bash
# デフォルト: バージョン 1 に出力
# data/output/001/1/ に保存される

# バージョン 2 に出力
EXP_VERSION=2 python -c "import config; print(config.OUTPUT_DIR)"
# → data/output/001/2/

# 次のバージョンを自動生成（既存の最大値 + 1）
EXP_VERSION=next python -c "import config; print(config.OUTPUT_DIR)"
# → data/output/001/3/ （001/1, 001/2 が存在する場合）

# 最新の既存バージョンを使用（推論時などに便利）
EXP_VERSION=latest python -c "import config; print(config.OUTPUT_DIR)"
# → data/output/001/2/ （001/1, 001/2 が存在する場合）
```

CLI で実行する場合：

```bash
# 実験ディレクトリに移動して実行
cd experiments/001
python train.py

# バージョンを指定して実行
EXP_VERSION=2 python train.py

# デバッグモード（少ないエポック/ラウンドで実行）
python train.py --debug

# 特定のfoldのみ実行
python train.py --fold 1
```

<details>
<summary>学習コードの例（train.py）</summary>

```python
import argparse

import config
import joblib
import polars as pl
from xgboost import XGBClassifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # データ読み込み
    train_df = pl.read_csv(config.COMP_DATASET_DIR / "train.csv")

    # モデル学習
    model = XGBClassifier(n_estimators=100, random_state=0)
    model.fit(
        X=train_df.select(["Age", "VIP", "VRDeck"]).to_numpy(),
        y=train_df["Transported"].to_numpy()
    )

    # モデル保存（OUTPUT_DIR に保存）
    model_dir = config.OUTPUT_DIR / "models"
    model_dir.mkdir(exist_ok=True, parents=True)
    joblib.dump(model, model_dir / "model.joblib")


if __name__ == "__main__":
    main()
```

</details>

### Step 3: 推論コードを作成

`experiments/001/inference.py` を編集します。これはKaggle上で実行されるコードです。

<details>
<summary>推論コードの例（inference.py）</summary>

```python
import config
import joblib
import polars as pl

# データ読み込み
test_df = pl.read_csv(config.COMP_DATASET_DIR / "test.csv")
sub_df = pl.read_csv(config.COMP_DATASET_DIR / "sample_submission.csv")

# モデル読み込み（ARTIFACT_EXP_DIR から読み込む）
model_dir = config.ARTIFACT_EXP_DIR(config.EXP_NAME) / "models"
model = joblib.load(model_dir / "model.joblib")

# 推論
test_pred = model.predict(test_df.select(["Age", "VIP", "VRDeck"]).to_numpy())

# 提出ファイル作成（OUTPUT_DIR に保存）
sub_df.with_columns(pl.Series("Transported", test_pred)).write_csv(
    config.OUTPUT_DIR / "submission.csv"
)
```

</details>

> **Note: パスの使い分け**
> - `OUTPUT_DIR`: 当該実験の出力先（学習時のモデル保存先）
> - `ARTIFACT_EXP_DIR(exp_name)`: Kaggle上でアップロードされたモデルの参照先（推論時）

### Step 4: 依存パッケージを設定（初回のみ）

> **重要**: `deps/code.ipynb` はテンプレートでは `loguru` のみがサンプルとして記載されています。
> **実際に使うパッケージに合わせて必ず編集してください。** 編集しないと Kaggle 上で import エラーになります。

`deps/code.ipynb` に、Kaggle環境にないパッケージをすべて記載します。

<details>
<summary>deps/code.ipynb の例</summary>

```python
# 必要なパッケージをダウンロード・インストール
!pip download -d /kaggle/working loguru polars
!pip install /kaggle/working/*.whl \
    --force-reinstall \
    --root-user-action ignore \
    --no-deps \
    --no-index \
    --find-links /kaggle/working
```

**ポイント:**
- `!pip download` に使うパッケージをすべて列挙する
- Kaggle環境にプリインストール済みのパッケージ（numpy, pandas, scikit-learn 等）は不要
- 不明な場合は [Kaggle Docker Image](https://github.com/Kaggle/docker-python) で確認

</details>

```bash
# 依存パッケージをKaggleにアップロード
sh scripts/push_deps.sh
```

### Step 5: Kaggle にアップロード

コードとモデルを**それぞれ個別に**アップロードします。

#### 5a. コードをアップロード

実験コード（`experiments/`, `src/` など）を Kaggle Dataset としてアップロードします。
コードを変更するたびに実行してください。

```bash
sh scripts/push_codes.sh
```

#### 5b. モデル（artifacts）をアップロード

学習済みモデルを Kaggle Model としてアップロードします。

```bash
# 初回アップロード（ローカルの最新バージョンを自動検出）
sh scripts/push_artifacts.sh 001

# 初回アップロード（ローカルのバージョン2を指定）
sh scripts/push_artifacts.sh 001 --version 2

# 2回目以降（既存の Model Instance に新バージョンを追加）
sh scripts/push_artifacts.sh 001 --update

# ローカルのバージョン2を指定して、既存 Model Instance に追加
sh scripts/push_artifacts.sh 001 --version 2 --update
```

> **重要**: アップロード後、**Kaggle 上で反映されるまで数分かかる場合があります**。
> 次の Step に進む前に、Kaggle Web でモデルが正しくアップロードされていることを確認してください。
> 確認先: https://www.kaggle.com/models → Your Work

### Step 6: 提出設定を編集

Kaggle 上でモデルの反映とバージョン番号を確認したら、以下の **2ファイル** を編集してください。

#### 1. `sub/code.ipynb` — inference.py のパスを実験番号に合わせる

```python
# 実験 001 を提出する場合:
!PYTHONPATH=/kaggle/input/datasets/{user}/{comp}-codes \
  python /kaggle/input/datasets/{user}/{comp}-codes/experiments/001/inference.py
#                                                               ^^^
#                                                   ここを提出したい実験番号に変更
```

> `{user}` と `{comp}` はテンプレート生成時に自動で置換されます。

#### 2. `sub/kernel-metadata.json` — `model_sources` の実験番号とバージョンを合わせる

```
"model_sources": [
  "{username}/{comp}-artifacts/other/001/1"
]                                    ^^^─^
                              実験番号    Kaggle上のバージョン番号
                                         （Kaggle Web で確認した値）
```

> **例: 実験 002、Kaggle バージョン 3 を提出する場合**
> - `sub/code.ipynb`: `experiments/001/inference.py` → `experiments/002/inference.py`
> - `sub/kernel-metadata.json`: `artifacts/other/001/1` → `artifacts/other/002/3`

#### model_sources のバージョンについて

**注意**: ローカルのバージョン番号と Kaggle のバージョン番号は**別物**です。

- ローカル: `data/output/001/2/` の `2` はローカルでの管理用
- Kaggle: Model Instance のバージョンは**アップロード順**に自動採番される

```
# 例: 同じ実験で3回アップロードした場合
ローカル 001/1/ → アップロード → Kaggle version 1 → model_sources: .../001/1
ローカル 001/2/ → アップロード → Kaggle version 2 → model_sources: .../001/2
ローカル 001/1/ を修正 → アップロード(--update) → Kaggle version 3 → model_sources: .../001/3
```

Kaggle上の最新バージョン番号は、Kaggle Web の Models ページで確認してください。

#### その他の設定（必要に応じて変更）

| フィールド | 説明 | 変更タイミング |
|-----------|------|---------------|
| `enable_gpu` | `"true"` でGPU有効 | **GPU推論が必要な場合**に `"true"` に変更 |
| `enable_tpu` | `"true"` でTPU有効 | **TPU推論が必要な場合**に `"true"` に変更 |
| `enable_internet` | `"true"` でインターネット有効 | 通常は `"false"`（コンペ規約に依存） |
| `is_private` | `"true"` = 非公開 | 公開したい場合に `"false"` |

<details>
<summary>kernel-metadata.json の全フィールド説明</summary>

| フィールド | 説明 | 変更タイミング |
|-----------|------|---------------|
| `id` | Kaggle上のKernel識別子 | 変更不要（自動生成） |
| `title` | Kaggle上での表示名 | 変更不要（自動生成） |
| `code_file` | 実行するノートブックファイル名 | 変更不要 |
| `language` | 実行言語 (`python`) | 変更不要 |
| `kernel_type` | `notebook` or `script` | 変更不要 |
| `dataset_sources` | コードDataset | 変更不要 |
| `competition_sources` | コンペティションデータ | 変更不要 |
| `kernel_sources` | 依存パッケージKernel | 変更不要 |
| `model_sources` | 学習済みモデル | **実験番号・バージョン変更時に編集** |

**Kaggleリソースとの対応関係:**
```
kernel-metadata.json          →  Kaggle上のリソース
─────────────────────────────────────────────────────
competition_sources           →  コンペデータ (/kaggle/input/competitions/{comp}/)
dataset_sources (comp-codes)  →  実験コード (/kaggle/input/datasets/{user}/{comp}-codes/)
kernel_sources (comp-deps)    →  依存パッケージ (/kaggle/input/{comp}-deps/) ※要確認
model_sources (comp-artifacts)→  学習済みモデル (/kaggle/input/models/{user}/{comp}-artifacts/)
```

</details>

### Step 7: カーネルを push して提出

Step 5 でアップロードしたリソースが Kaggle 上に反映されていることを確認したら、カーネルを push します。

```bash
# カーネルを push（提出）
sh scripts/push_sub.sh

# 提出状況の確認
sh scripts/status.sh
```

### 提出フローまとめ

```bash
# 初回提出の場合（すべて実行）
sh scripts/push_deps.sh              # 1. 依存パッケージをアップロード（初回のみ）
sh scripts/push_codes.sh             # 2. コードをアップロード
sh scripts/push_artifacts.sh 001     # 3. モデルをアップロード
# → Kaggle Web でモデルの反映を確認
# → sub/kernel-metadata.json と sub/code.ipynb を編集
sh scripts/push_sub.sh               # 4. カーネルを push

# コードだけ変更した場合
sh scripts/push_codes.sh
sh scripts/push_sub.sh

# モデルを更新した場合
sh scripts/push_artifacts.sh 001 --update
# → Kaggle Web でバージョン番号を確認
# → sub/kernel-metadata.json の model_sources を更新
sh scripts/push_sub.sh
```

---

## Scripts Reference

| Script | Description |
|--------|-------------|
| `scripts/push_codes.sh` | コードを Kaggle Dataset としてアップロード |
| `scripts/push_artifacts.sh <exp> [options]` | モデルを Kaggle Model としてアップロード |
| `scripts/push_deps.sh` | 依存パッケージをアップロード |
| `scripts/push_sub.sh` | 提出カーネルを push |
| `scripts/status.sh` | 最新の提出状況を確認 |
| `scripts/new_exp.sh [--template TYPE] [--base NUM]` | 新しい実験フォルダを作成 |
| `scripts/download_competition.sh` | コンペティションデータをダウンロード |

<details>
<summary>push_artifacts.sh のオプション</summary>

```bash
sh scripts/push_artifacts.sh <exp_name> [options]

Options:
  --version VER    アップロードするバージョンを指定（デフォルト: latest）
  --update         既存の Model Instance を更新する（2回目以降のアップロード）

# 使用例
sh scripts/push_artifacts.sh 001                  # 最新バージョンをアップロード
sh scripts/push_artifacts.sh 001 --version 2      # バージョン2をアップロード
sh scripts/push_artifacts.sh 001 --update         # 既存のModelを更新
```

</details>

---

## Project Structure

```
{{ cookiecutter.project_slug }}/
├── experiments/           # 実験フォルダ
│   ├── 001/              # 実験番号ごとのフォルダ
│   │   ├── config.py     # パス設定、コンペ情報
│   │   ├── train.py      # 学習用スクリプト
│   │   └── inference.py  # 推論コード
│   └── templates/        # 実験テンプレート
├── sub/                   # 提出用Kernel
│   ├── code.ipynb        # 提出時に実行されるノートブック
│   └── kernel-metadata.json
├── deps/                  # 依存パッケージ管理
├── src/                   # ユーティリティ
│   ├── upload.py         # アップロード処理
│   ├── download.py       # ダウンロード処理
│   └── kaggle_utils/     # Kaggle API ラッパー
├── scripts/              # シェルスクリプト
├── data/
│   ├── input/            # ダウンロードしたデータ
│   └── output/           # 実験の出力
└── docker/               # Dockerfile
```

<details>
<summary>データフロー図</summary>

```
[ローカル環境]                              [Kaggle]

experiments/001/
  ├─ train.py (学習)
  ├─ config.py
  └─ inference.py ──────────────────────→ Dataset: {comp}-codes

data/output/001/1/
  └─ models/*.pkl ──────────────────────→ Model: {comp}-artifacts/other/001/1

deps/
  └─ code.ipynb ────────────────────────→ Kernel: {comp}-deps (依存パッケージ)

sub/
  └─ code.ipynb ────────────────────────→ Kernel: {comp}-sub
      │                                       │
      │ (inference.pyを呼び出し)               ↓
      └───────────────────────────────→ submission.csv
```

</details>

---

## Troubleshooting

<details>
<summary>よくあるエラーと対処法</summary>

### "KAGGLE_USERNAME is not set"

**原因**: `.env` ファイルが存在しないか、設定が正しくない

**対処法**:
1. `.env.sample` を `.env` にコピー
2. `KAGGLE_USERNAME` と `KAGGLE_KEY` を正しく設定
3. 値が `your_username` や `your_key` のままになっていないか確認

---

### "Experiment directory not found"

**原因**: 指定した実験番号のフォルダが存在しない

**対処法**:
```bash
# 新しい実験フォルダを作成
sh scripts/new_exp.sh
# または手動で作成
mkdir -p experiments/001
cp experiments/templates/tabular/* experiments/001/
```

---

### "Output directory not found"

**原因**: 実験を実行していない、または出力パスが間違っている

**対処法**:
1. `experiments/001/train.py` を実行してモデルを学習・保存
2. `config.py` の `OUTPUT_DIR` を確認
3. `data/output/001/1/` にファイルが出力されているか確認

---

### Kaggle Kernel が "Error" で失敗

**対処法**:
1. `sh scripts/status.sh` でエラーログを確認
2. ログを確認: `kaggle kernels output <kernel_slug>`
3. よくある原因:
   - 依存パッケージの不足 → `deps/code.ipynb` を更新して `sh scripts/push_deps.sh`
   - パスの間違い → `sub/code.ipynb` の inference.py パスを確認
   - モデルファイルの参照エラー → `sub/kernel-metadata.json` の `model_sources` を確認
   - モデルが Kaggle 上で未反映 → Kaggle Web で反映を確認してから再度 `sh scripts/push_sub.sh`

---

### "403 - Forbidden" エラー

**原因**: コンペティションの規約に同意していない

**対処法**:
1. Kaggle Web で該当コンペティションのページを開く
2. "Join Competition" または "I Understand and Accept" をクリック
3. 再度ダウンロードを試行

---

### Model/Dataset が Kaggle 上に表示されない

**原因**: アップロード処理が完了していない、またはプライベート設定

**対処法**:
1. アップロード時のエラーメッセージを確認
2. Kaggle Web の "Your Work" → "Datasets" / "Models" で確認
3. 初回アップロード後、反映に数分かかることがある

---

### Dataset のバージョンが更新されない

**原因**: Kaggle API はファイルに変更がないとバージョンを作成しない仕様

**対処法**:
- ファイルの内容に変更がある場合は自動的に新バージョンが作成されます
- 変更がないのにバージョンを作りたい場合は、Kaggle Web から手動で作成してください

---

### 複数バージョンのモデルをアップロードしたい

**対処法**:
```bash
# バージョン 1 をアップロード（初回）
sh scripts/push_artifacts.sh 001

# バージョン 2 をアップロード
sh scripts/push_artifacts.sh 001 --version 2

# 既存の Model Instance を更新する場合は --update を使用
sh scripts/push_artifacts.sh 001 --update
```

**注意**: Kaggle の Model Instance は `{username}/{comp}-artifacts/other/{exp_name}` という
単一のパスを持ちます。バージョンは Kaggle 側で管理されます。

---

### ARTIFACT_EXP_DIR でバージョンを指定したい

**対処法**:
```python
import config

# バージョン 1 のモデル（デフォルト）
model_dir = config.ARTIFACT_EXP_DIR("001")
# → data/output/001/1/

# バージョン 2 のモデル
model_dir = config.ARTIFACT_EXP_DIR("001", version="2")
# → data/output/001/2/
```

</details>

---

## Reference
- [kaggle code competition 用のテンプレート作ってみた](https://osushinekotan.hatenablog.com/entry/2024/12/24/193145)
- [効率的なコードコンペティションの作業フロー](https://ho.lc/blog/kaggle_code_submission/)
- [Kaggleコンペ用のVScode拡張を開発した](https://ho.lc/blog/vscode_kaggle_extension/)
