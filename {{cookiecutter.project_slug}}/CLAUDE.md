# {{ cookiecutter.project_name }}

Kaggle コードコンペティション「{{ cookiecutter.competition_name }}」用プロジェクト。

## コンペ情報

詳細は `docs/` 配下を参照:

- `docs/competition_overview.md` — コンペ概要・ルール・制約
- `docs/dataset_description.md` — データセットの詳細
- `docs/evaluation_metric.md` — 評価指標の解説
- `docs/domain_knowledge.md` — ドメイン知識・有効なアプローチ

## プロジェクト構造

- `experiments/{NNN}/` — 実験フォルダ（config.py / train.py / inference.py）
- `src/kaggle_utils/` — Kaggle APIラッパー（upload/download）
- `data/input/` — コンペデータ、`data/output/` — 学習成果物
- `sub/` — 提出用カーネル、`deps/` — 依存パッケージ用カーネル
- `scripts/` — ワークフロー用シェルスクリプト

## 実験の仕組み

- 各実験の `config.py` が `IS_KAGGLE_ENV` でローカル/Kaggle環境を自動判別しパスを切り替える
- `EXP_NAME` は `Path(__file__).parent.name` から自動取得（フォルダ名 = 実験名）
- `EXP_VERSION` 環境変数でバージョン管理（`next` で新規、`latest` で最新）

## 主要コマンド

```sh
sh scripts/new_exp.sh                  # 新規実験作成
sh scripts/download_competition.sh     # データダウンロード
sh scripts/push_codes.sh              # コードをKaggle Datasetにアップロード
sh scripts/push_artifacts.sh {NNN}    # モデルをKaggle Modelにアップロード
sh scripts/push_deps.sh              # 依存パッケージをアップロード
sh scripts/push_sub.sh               # 提出カーネルをpush
sh scripts/status.sh                 # 提出状況を確認
```

細かい仕様やオプションは `README.md` を参照。
