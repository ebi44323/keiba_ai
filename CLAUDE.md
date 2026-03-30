# keiba-ebye プロジェクト

## 概要
競馬AI予測システム。Streamlit + LightGBM 3モデルアンサンブル。
HuggingFace Spaces にデプロイ。GitHub push → sync.yml → HF自動反映。

## バージョン管理ルール
- `app.py` 33行目: `st.caption("vYYYY-MM-DDx")` を**毎回の変更時に必ず更新**すること
- 形式: `v年-月-日+アルファベット`（同日複数回は a→b→c と増やす）
- 例: `v2026-03-28a`、`v2026-03-28b`
- **用途**: HF Space Filesタブでコード到達確認・アプリ画面で起動確認に使う
- 現在: `v2026-03-28a`

---

## ファイル構成

```
app.py                     メインStreamlit UI
features_engine.py         src/features_engine.py の re-export のみ（実体はsrc側）
update_data.py             週次データ更新スクリプト（python update_data.py で実行）
backfill_race_name.py      レース名補完スクリプト
check_market_rate_auc.py   市場勝率のAUC寄与検証スクリプト
train_and_push.py          ローカル再学習→HF Hub保存スクリプト（手動実行用）

src/
  __init__.py              パッケージ宣言（HuggingFace Linux環境に必須）
  config.py                ★共通定数・関数（PLACE_DICT/VENUE_MAWARI等）の一元管理
  features_engine.py       特徴量定義(NUM_FEATURES/CAT_FEATURES)＋create_features()
  core_model.py            モデル学習・HF Hub保存/ロード
  inference.py             リアルタイム推論・スクレイピング
  optuna_tuner.py          Optunaチューニング（ウォークフォワードCV・AUC目的関数）
  backtest.py              時系列バックテスト
  scraper.py               netkeiba/Yahoo競馬スクレイピング
  discord_utils.py         Discord通知（Webhook直接送信）
  reports.py               PDF/テキストレポート生成
  utils.py                 resolve_name/classify_race_class（定数はconfig.pyから再エクスポート）

学習データ:
  learning_data_perfect_tier.zip   34MB（HuggingFace LFSで管理）
  learning_data_perfect_tier.csv   140MB・197,877行・82カラム（gitignore対象）
  pedigree_master_all.csv          血統マスター
  ped_cache.db                     血統取得sqliteキャッシュ（update_data.py使用）
```

---

## モデル構成

3モデルアンサンブル（重み: A×0.35 + B×0.50 + C×0.15）

| モデル | 種別 | 目的変数 | 備考 |
|---|---|---|---|
| A | LGBMRanker | 馬券内(3着以内) | n_estimators=500 |
| B | LGBMRanker | 1着 | Optunaチューニング対象 |
| C | LGBMRegressor | 着順パーセント | n_estimators=300 |
| D | LGBMClassifier | 穴馬(人気5以下の1着) | scale_pos_weight≈57、予想◎には影響しない・穴馬マーク専用 |

---

## AUCに関する重要な数値

- **市場勝率あり AUC: 0.8313**（単勝オッズ由来の特徴量が混在するため高め）
- **市場勝率なし AUC: 0.7528**（← これが真のモデル力）
- Optunaは「市場勝率なし」で回すこと（UIのチェックボックスがデフォルトON）
- 目標AUC: 0.74〜0.78（市場勝率なしの基準）

## Optunaチューニング結果（最新: 2026-03-25 実施）

- **CV AUC: 0.7615**（3fold ウォークフォワードCV、50試行、市場勝率除外）
- 適用済みパラメータ（モデルB）:
  ```json
  {"n_estimators":235,"learning_rate":0.023034,"num_leaves":82,
   "max_bin":171,"cat_smooth":48.53,"colsample_bytree":0.5056,
   "subsample":0.6815,"min_child_samples":25}
  ```
- ⚠️ `min_child_samples:25`（前回96から大幅減）→ 過学習リスクあり、実戦回収率で検証推奨
- 前回結果（2026-03-24）: CV AUC 0.7596、min_child_samples=96（より保守的）

---

## 特徴量

### NUM_FEATURES（src/features_engine.py で一元管理）
枠番・馬番・年齢・距離・斤量・出走頭数・馬体重系・前走着順系・
スピード指数系（過去5走）・コース適性・各種フラグ（乗り替わり・馬場替わり等）・
穴馬フラグ4種・キャリア数・前走上り順位率・前走ペース値・
馬場指数・レースクラスコード・市場勝率・
ベスト3走_中央値スピード指数・長期休養フラグ・レース格上挑戦フラグ・
コース初挑戦フラグ・近5走_スピード指数安定性・斤量_前走差（**計51特徴量**）

追加特徴量の概要（2026-03-25 追加）:
- `ベスト3走_中央値スピード指数`: 近5走の上位3走median（ピーク能力）
- `長期休養フラグ`: 休養180日以上（0/1）
- `レース格上挑戦フラグ`: 前走より上クラスへの出走（0/1）
- `コース初挑戦フラグ`: 競馬場×芝ダートの初出走（0/1）
- `近5走_スピード指数安定性`: 近5走の標準偏差（低いほど安定）

追加特徴量の概要（2026-03-30 追加）:
- `斤量_前走差`: 今走斤量 - 前走斤量（ハンデ戦の斤量増減を捉える、初出走時はNaN）

⚠️ ユーザー方針: 騎手直近勝率・人気由来の特徴量は追加しない（後発情報・信頼性の問題）

### CAT_FEATURES
競馬場・芝ダート・天候・馬場・父系・母系・母父系・
前走芝ダート・回り・コース地形・脚質カテゴリ・騎手×競馬場・騎手×距離

### Target Encoding（TE_COLS）
調教師・父・母父・騎手

---

## リーク防止の実装

- **コース統計**: `shift(1).expanding(min_periods=3).mean()` で過去データのみ使用
  - フォールバックも expanding window (min_periods=1) に修正済み（2026-03-30）→ 以前の `transform('mean')` リーク修正
- **コース適性_着順パーセント**: 同上、`min_periods=3`（データ3件未満は0.5でfillna）
- **Target Encoding**: foldごとにtrainデータのみから計算
- **バックテスト/振り返り**: `skip_live_scrape=True` で `fetch_horse_last_race()` をスキップ
- **未出走判定** (`has_unraced`): レーステキストの「新馬」「未出走」のみで判断
  - ⚠️ `df_test['前走_着順'].isna().any()` はNGだった（リーク防止コードでNaN上書きされるため全レース誤判定）→ 修正済み

---

## 重要なルール・注意点

- 馬IDは必ず `.zfill(10)` でゼロ埋め
- レースIDは12桁、地方競馬(place_code 11〜)は除外
- LFSで管理: `*.zip`, `*.pkl`, `*.joblib`
- `src/config.py` が定数の真実の源（PLACE_DICT, VENUE_MAWARI, VENUE_CHIKEI等）
  - `src/utils.py` はconfigから再エクスポートするだけ
  - `update_data.py`, `backfill_race_name.py` も同様
- 血統取得は `_get_pedigree_cached()` を使う（sqlite キャッシュ: ped_cache.db）
- **.gitignoreに必ず含めること**: `*.png`, `*.jpg`, `*.jpeg`, `learning_data_perfect_tier.csv`
  - PNG等のバイナリをコミットするとHuggingFace Spacesへのpushが拒否される

---

## デプロイフロー

```
git push origin main
  → GitHub Actions: .github/workflows/sync.yml
    → rm -rf .git && git init（履歴リセット）
    → git lfs track "*.zip" "*.pkl" "*.joblib"
    → git add . && git push --force hf main
      → HuggingFace Space 自動再ビルド（2〜3分）
```

### ⚠️ デプロイ時の注意
- **PNGなどの画像ファイルは絶対にコミットしない**（HF Spacesがバイナリを拒否してビルド失敗になる）
- **requirements.txtのバージョン制約変更は慎重に**（pip キャッシュ無効化 → ビルドが15〜30分になる）
- sync.ymlは `git add .` で全ファイルを拾うため、不要ファイルは .gitignore に追加すること

---

## HuggingFace Hub モデル管理

- **モデル保存先**: `ebi44323/keiba-ebye-models`（Dataset リポジトリ）
  - `keiba_model.pkl` — 学習済みモデルbundle
  - `keiba_model_meta.json` — データ識別子（`data_mtime: "size:35515667"`形式）
  - `ai_daily_history.csv` — 振り返り日次成績
  - `discord_queue.json` — Discord通知キュー

- **data_mtime の仕組み**: ZIPのファイルサイズ (`size:XXXXX`) を使用
  - DockerのCOPYでOSのmtimeがリセットされるため、サイズで比較するよう変更済み（2026-03-25）
  - ZIPを更新したら必ず再学習 → HF Hub保存が必要

- **再学習が必要なタイミング**:
  - `learning_data_perfect_tier.zip` を更新したとき
  - 特徴量・調教師正規化など学習パイプラインを変更したとき

- **ローカル再学習コマンド**:
  ```powershell
  $env:HF_TOKEN="hf_xxxxxxxxxxxx"
  $env:HF_REPO_ID="ebi44323/keiba-ebye-models"
  python train_and_push.py
  ```

- **HF SpaceのSecrets設定（必須）**:
  - `HF_TOKEN` — HuggingFace APIトークン（read/write権限）
  - `HF_REPO_ID` — `ebi44323/keiba-ebye-models`

---

## 現在の状況（2026-03-30 時点）

### 完了済み修正
- 特徴量リスト重複排除（root features_engine.py → re-export化）
- EXTRA_NUM を NUM_FEATURES に統合（core_model.py から削除）
- src/config.py 作成（定数一元管理）
- src/__init__.py 作成（HuggingFace パッケージimport修正）
- コース適性 expanding min_periods=3 追加（リーク軽減）
- 血統取得 sqlite キャッシュ化（update_data.py）
- Optuna 目的関数: 回収率 → AUC に変更
- Optuna UI: 市場勝率除外スイッチ追加（デフォルトON）、コピペ用コード表示
- 振り返り「全レース見送り推奨」バグ修正（has_unraced の誤判定）
- **Optuna結果をモデルBに適用**（CV AUC 0.7615, 50試行 @ 2026-03-25）
- **Isotonic Calibration追加**（bundle要素数 18→19）
- **新特徴量5種追加**（計50特徴量）
- **調教師名の正規化**: `[東]/[西]` プレフィックスを除去（core_model.py・load_jockey_base）
  - 学習データの `[東] 矢作芳人` → `矢作芳人` に統一（195k行以上に影響）
  - 再学習済み（2026-03-25 17:01 JST、HF Hub保存済み）
- **オッズ取得の修正**: 枠順未確定時に馬名→馬番の優先順位で正確にマッピング
- **スライダー操作で予想が消えるバグ修正**: session_stateキャッシュ方式に変更
- **穴馬マーク詳細表示**: tab2（買い目展開）に穴馬スコアとヒント表示を追加
- **EV優先回収率の長期追跡**: ai_daily_history.csvに `EV優先単勝回収率`/`EV優先複勝回収率` 列追加
- **長期成績分析にEV優先比較**: 標準◎とEV優先の日別/月次比較を表示
- **騎手データベース追加**: 15タブの詳細分析（脚質・上がり性能・馬体重・相性調教師など）
- **Discord週次レポートにEV優先比較追加**
- **HF Space再起動ループ修正**: data_mtimeをファイルサイズ比較に変更、PNG gitignore追加
- **Discord完全自動化**: predict_auto.py・auto_morning.py・auto_review.py・auto_weekend_summary.py + 各GitHub Actionsワークフロー追加
- **EV優先◎をデフォルトに変更**: app.py の EV優先モード checkbox を value=True に
- **EV推奨候補ウィジェット追加**: EV>=1.0 の上位3馬をカード表示
- **オッズキャッシュバスティング追加**: `&_={timestamp}` + no-cache ヘッダーでリアルタイムオッズ化
- **コース統計フォールバックのリーク修正**: `transform('mean')` → `expanding(min_periods=1)` に変更（2026-03-30）
- **`斤量_前走差` 特徴量追加**: 前走からの斤量増減（ハンデ戦対応）、計51特徴量（2026-03-30）
- **Optuna多目的関数化**: AUC×weight + 回収率×weight のブレンドスコア対応（UIでウェイト調整可能）
- **アンサンブル重み最適化追加**: Optunaで wa/wb/wc を EV回収率最大化方向で探索する機能追加
- ⚠️ 特徴量変更（斤量_前走差追加）→ **再学習が必要**

### 3月振り返り結果から判明した課題（2026-03-25）
- 本命単勝回収率: 39〜88%（全日100%未満）→ 本命一辺倒では儲からない
- 穴馬単勝回収率: 245.6%など大きいが、**0%の日が多い（穴馬をほとんど推奨しない）**
- 穴馬複勝回収率: 185.7%・134.4% → 穴馬を複勝圏には入れているが◎にしていない
- **根本原因**: 現在◎=max(AI勝率)。ランキングモデルは市場合意と相関するため1番人気多発

---

## 次にやること（優先度順）

### 高優先度（予測精度・回収率改善）
1. **再学習の実施**（最優先）
   - `斤量_前走差` 特徴量追加 + コース統計フォールバックのリーク修正が入ったため再学習必須
   - `python train_and_push.py` で HF Hub に保存
2. **min_child_samples=25 の過学習検証**
   - 実戦数週間後に回収率を確認し、必要なら保守的な96に戻す
3. **Optunaの再チューニング（再学習後）**
   - 新特徴量（斤量_前走差）追加の効果確認 + AUC/回収率ブレンド目的関数での最適化
   - アンサンブル重み最適化も実施推奨

### 中優先度（機能拡張）
4. **週次データ更新の実施** (`python update_data.py`)
   - 最新レース結果を学習データに追加 → 再学習でモデル更新
5. **EV優先データの蓄積と比較検証**
   - 今後数週間の振り返りでEV優先 vs 標準◎ の実績を比較
6. **Discord通知の動作確認**
   - 次の土日に GitHub Actions ワークフローが正常動作するか確認

### 低優先度（将来的な改善）
7. **騎手データベースの拡充**
   - 現在15タブだが、さらに項目を追加する余地あり
8. **モデルD（穴馬専用）の活用強化**
   - 現在◎への影響なし。穴馬複勝推奨への活用を検討

---

## よく使うコマンド

```bash
# データ更新（先週分）
python update_data.py

# データ更新（全未取得分）
python update_data.py --all

# レース名補完
python backfill_race_name.py

# 市場勝率AUC検証
python check_market_rate_auc.py

# ローカル動作確認
streamlit run app.py

# ローカル再学習 → HF Hub保存（ZIPを更新した後や学習パイプライン変更後に実行）
# PowerShellで:
#   $env:HF_TOKEN="hf_xxxxxxxxxxxx"
#   $env:HF_REPO_ID="ebi44323/keiba-ebye-models"
python train_and_push.py
```
