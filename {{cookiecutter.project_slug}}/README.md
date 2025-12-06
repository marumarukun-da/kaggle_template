# {{ cookiecutter.project_name }}

## setup
### 1. `.env.sample`をコピーして`.env`を作成し、Kaggle認証情報を設定します：
```bash
KAGGLE_USERNAME=your_username  # あなたのKaggleユーザー名
KAGGLE_KEY=your_key           # あなたのKaggle API key
```

### 2. 環境に応じてDockerコンテナを起動します：

GPU環境の場合：
```bash
docker compose up -d --build gpu
```

CPU環境の場合：
```bash
docker compose up -d --build cpu
```

### 3. VSCodeの左下の「><」アイコンをクリックし、「コンテナーで再度開く」を選択してコンテナに接続します。

### 4. VSCodeで推奨される拡張機能をインストールします：
   - 右下に表示される「推奨される拡張機能をインストール」の通知をクリックする
   - または、拡張機能タブ（Ctrl+Shift+X）から「推奨」セクションを確認してインストール

## download competition dataset

### 1. Kaggle上でコンペティションへの参加登録を行う

### 2. データセットをダウンロード

  ```bash
  sh scripts/download_competition.sh
  ```

## submission flow

### 1. `experiments` に実験フォルダを作成する
- 例: `experiments/001/`
  - 必要なファイル：
    - `code.ipynb`: 実験/学習用コード
    - `config.py`: 設定ファイル
    - `inference.py`: 推論用コード

### 2. 実験を行う
- モデルの学習と保存は `code.ipynb` で行う

    <details>
    <summary>例</summary>

    ```python
    import config
    import joblib
    import polars as pl
    from xgboost import XGBClassifier

    train_df = pl.read_csv(config.COMP_DATASET_DIR / "train.csv")
    model_dir = config.OUTPUT_DIR / "models"
    model_dir.mkdir(exist_ok=True, parents=True)

    model = XGBClassifier(n_estimators=100, random_state=0)
    model.fit(X=train_df.select(["Age", "VIP", "VRDeck"]).to_numpy(), 
            y=train_df["Transported"].to_numpy())

    joblib.dump(model, model_dir / "model.joblib")
    ```
    </details>



- 推論と提出用ファイルの作成は `inference.py` で行う

    <details>
    <summary>推論と提出用ファイルの作成 (inference.py)</summary>

    ```python
    import config
    import joblib
    import polars as pl

    test_df = pl.read_csv(config.COMP_DATASET_DIR / "test.csv")
    sub_df = pl.read_csv(config.COMP_DATASET_DIR / "sample_submission.csv")
    model_dir = config.ARTIFACT_EXP_DIR(config.EXP_NAME) / "models"

    # ARTIFACT_EXP_DIR を参照してモデルをロード
    model = joblib.load(model_dir / "model.joblib")
    test_pred = model.predict(test_df.select(["Age", "VIP", "VRDeck"]).to_numpy())

    # OUTPUT_DIR に提出用ファイルを保存
    sub_df.with_columns(pl.Series("Transported", test_pred)).write_csv(
        config.OUTPUT_DIR / "submission.csv")
    ```
    </details>


> **Note**
> - path周りの設定は全て`config.py`に書く
> - 当該notebook/codeの生成物は`OUTPUT_DIR`に出力する
> - 当該notebook/code以外の生成物のロードは`ARTIFACT_EXP_DIR`を参照する


### 3. 以下のいずれかの方法を使って、サブミッション時に使用するコードやモデルを upload する

  - スクリプトの実行

    ```bash
    sh scripts/push_experiment.sh 001  # 001は実験番号
    ```

  - コードの実行

    ```python
    from src.kaggle_utils.customhub import dataset_upload, model_upload

    model_upload(
      handle=config.ARTIFACTS_HANDLE,
      local_model_dir=config.OUTPUT_DIR,
      update=False,
    )
    dataset_upload(
      handle=config.CODES_HANDLE,
      local_dataset_dir=config.ROOT_DIR,
      update=True,
    )
    ```


### 4. 必要な dependencies を push する
- `deps/code.ipynb` を必要なパッケージをインストールするように編集
- スクリプトの実行

  ```sh
  sh scripts/push_deps.sh
  ```

### 5. submission
  - `sub/code.ipynb` を編集して使用するモデルやコードを指定
  - 複数のモデルを参照したい場合は、`sub/kernel-metadata.json` の `model_sources` に追加
  - 編集後、以下のコマンドを実行

    ```sh
    sh scripts/push_sub.sh
    ```


## Reference
- [kaggle code competition 用のテンプレート作ってみた](https://osushinekotan.hatenablog.com/entry/2024/12/24/193145)
- [効率的なコードコンペティションの作業フロー](https://ho.lc/blog/kaggle_code_submission/)
- [Kaggleコンペ用のVScode拡張を開発した](https://ho.lc/blog/vscode_kaggle_extension/)
