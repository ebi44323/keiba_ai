\# keiba-ebye プロジェクト



\## 概要

競馬AI予測システム。Streamlit + LightGBM。HuggingFace Spacesにデプロイ済み。



\## ファイル構成

\- app.py: メインUI

\- src/core\_model.py: モデル学習

\- src/inference.py: 推論・スクレイピング

\- src/features\_engine.py: 特徴量エンジン

\- src/discord\_utils.py: Discord通知(HFキュー経由)

\- src/optuna\_tuner.py: Optunaチューニング(ウォークフォワードCV版)

\- update\_data.py: 週次データ更新



\## 重要な注意点

\- コース統計は expanding window でリーク修正済み

\- 馬IDは必ず .zfill(10) でゼロ埋めすること

\- Discord通知はHF DatasetキューをGitHub Actionsが5分ごとに送信（現在はうまくいってない）

\- AUCは市場勝率特徴量の影響で高めに出る(0.80〜0.85は正常範囲)



\## 現在の課題

\- Optunaチューニング結果をcore\_model.pyに反映待ち

