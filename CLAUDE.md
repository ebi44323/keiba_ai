# keiba-ebye プロジェクト

## 概要
競馬AI予測システム。Streamlit + LightGBM 3モデルアンサンブル。
HuggingFace Spaces にデプロイ。GitHub push → sync.yml → HF自動反映。

## バージョン管理ルール
- `app.py` 33行目: `st.caption("vYYYY-MM-DDx")` を**毎回の変更時に必ず更新**すること
- 形式: `v年-月-日+アルファベット`（同日複数回は a→b→c と増やす）
- 例: `v2026-03-28a`、`v2026-03-28b`
- **用途**: HF Space Filesタブでコード到達確認・アプリ画面で起動確認に使う
- 現在: `v2026-03-30e`

---

## ファイル構成

```
app.py                     メインStreamlit UI
features_engine.py         src/features_engine.py の re-export のみ（実体はsrc側）
update_data.py             週次データ更新スクリプト（python update_data.py で実行）
backfill_race_name.py      レース名補完スクリプト
check_market_rate_auc.py   市場勝率のAUC寄与検証スクリプト
train_and_push.py          ローカル再学習→HF Hub保存スクリプト（手動実行用）
predict_auto.py            発走前自動予想スクリプト（GitHub Actions経由）
auto_morning.py            朝8時全レース予想→Discord投稿スクリプト
auto_review.py             当日振り返りスクリプト（17時以降）
auto_weekend_summary.py    土日合算まとめ→Discord投稿スクリプト

src/
  __init__.py              パッケージ宣言（HuggingFace Linux環境に必須）
  config.py                ★共通定数・関数（PLACE_DICT/VENUE_MAWARI等）の一元管理
  features_engine.py       特徴量定義(NUM_FEATURES/CAT_FEATURES)＋create_features()
  core_model.py            モデル学習・HF Hub保存/ロード
  inference.py             リアルタイム推論・スクレイピング
  optuna_tuner.py          Optunaチューニング（ウォークフォワードCV・多目的関数）
  backtest.py              時系列バックテスト
  scraper.py               netkeiba/Yahoo競馬スクレイピング
  discord_utils.py         Discord通知（キュー経由・Webhook直接送信）
  reports.py               PDF/テキストレポート生成
  utils.py                 resolve_name/classify_race_class（定数はconfig.pyから再エクスポート）

.github/workflows/
  sync.yml                 push → HF Space 自動デプロイ（CLAUDE.md等は除外）
  retrain.yml              手動トリガーによる再学習 → HF Hub保存
  auto_predict.yml         土日 15分ごと 発走前予想→Discord（JST 9:00〜16:45）
  auto_morning.yml         土日 朝8時 全レース予想ファイル→Discord
  auto_review.yml          土日 17時 振り返り→Discord
  auto_weekend_summary.yml 日曜 20時 土日合算まとめ→Discord
  discord_notify.yml       5分ごと Discordキューを読んで送信
  discord_weekly_report.yml 週次レポート→Discord
  weekly_update.yml        週次データ更新

学習データ:
  learning_data_perfect_tier.zip   34MB（git管理・GitHub Actions retrain.yml が使用）
  learning_data_perfect_tier.csv   140MB・197,877行・82カラム（gitignore対象）
  pedigree_master_all.csv          血統マスター
  ped_cache.db                     血統取得sqliteキャッシュ（update_data.py使用）
```

---

## 馬カテゴリ定義（全ファイル共通・2026-03-30 統一）

| カテゴリ | 定義 | 使用箇所 |
|---|---|---|
| **超狙い馬** | AIランク上位5頭（`index < 5`）かつ `期待値 >= 1.5` | app.py / auto_review.py / auto_weekend_summary.py / discord_utils.py |
| **穴馬** | AIランク6位以下（`index >= 5`）かつ `期待値 >= 1.5` | 同上 |

- `index` は `res_df` の行インデックス（`reset_index(drop=True)` 後の0始まり）
- **注意**: `index < 5` は0〜4（上位5頭）、`index >= 5` は5以降（6位以下）
- ai_daily_history.csv の列名: `超狙い馬単勝回収率` / `超狙い馬複勝回収率` / `穴馬単勝回収率` / `穴馬複勝回収率`
- **旧名称**: `ev_invest` / `ev_tan_hits` 等は廃止。`choko_*` (超狙い馬) / `ana_*` (穴馬) に統一

---

## モデル構成

3モデルアンサンブル（重み: A×0.0581 + B×0.8159 + C×0.1261）
※ アンサンブル重み最適化 @ 2026-03-30 で更新（EV回収率最大化方向で探索）

| モデル | 種別 | 目的変数 | 備考 |
|---|---|---|---|
| A | LGBMRanker | 馬券内(3着以内) | n_estimators=500 |
| B | LGBMRanker | 1着 | Optunaチューニング対象・重み最大 |
| C | LGBMRegressor | 着順パーセント | n_estimators=300 |
| D | LGBMClassifier | 穴馬(人気5以下の1着) | scale_pos_weight≈57、予想◎には影響しない・穴馬マーク専用 |

---

## AUCに関する重要な数値

- **市場勝率あり AUC: 0.8313**（単勝オッズ由来の特徴量が混在するため高め）
- **市場勝率なし AUC: 0.7528**（← これが真のモデル力）
- Optunaは「市場勝率なし」で回すこと（UIのチェックボックスがデフォルトON）
- 目標AUC: 0.74〜0.78（市場勝率なしの基準）

## Optunaチューニング結果（最新: 2026-03-30 実施）

- **CVスコア: 0.6770**（AUC×0.7 + 正規化回収率×0.3、3fold ウォークフォワードCV、50試行、市場勝率除外）
- ※ スコアは合成値のため旧AUC(0.7615)と直接比較不可。実質AUC≈0.75〜0.76
- 適用済みパラメータ（モデルB @ 2026-03-30）:
  ```json
  {"n_estimators":700,"learning_rate":0.012275,"num_leaves":32,
   "max_bin":162,"cat_smooth":31.99,"colsample_bytree":0.7124,
   "subsample":0.8923,"min_child_samples":77}
  ```
- `min_child_samples:77`（前回25から大幅増）→ 過学習リスク軽減 ✅
- 前回結果（2026-03-25）: CV AUC 0.7615、n_estimators=235、min_child_samples=25

## アンサンブル重み最適化結果（2026-03-30 実施）

- **最適EV回収率: 74.1%**（最新20%ホールドアウトでの評価）
- ※ 訓練データ上の評価のため絶対値より重み比率が重要
- 結果: wa=0.0581 / wb=0.8159 / wc=0.1261
  - モデルB（1着予測）が支配的 → EV=AI勝率×オッズ に最も直結するため理にかなっている
- 適用箇所: `src/core_model.py` L331 と `src/inference.py` L461 の両方を更新済み

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
  - ⚠️ 実際には斤量が均一な場合が多く（牡55kg固定等）モデルへの寄与は限定的

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
  - さらに距離を無視した場×芝ダ全体フォールバックも追加（完全新設コース用）
- **コース適性_着順パーセント**: 同上、`min_periods=3`（データ3件未満は0.5でfillna）
- **Target Encoding**: foldごとにtrainデータのみから計算
- **バックテスト/振り返り**: `skip_live_scrape=True` で `fetch_horse_last_race()` をスキップ
  - `最新_日付 >= race_date` の馬の `最新_*` / `前走_*` 系をNaNマスク済み（`inference.py:254`）
  - ⚠️ **未修正リーク**: `horse_course_dict`（`core_model.py:239`）は全学習データの `.mean()` で生成。振り返り日より未来のレースも含まれるため `コース適性_着順パーセント` が若干楽観的になる。`ai_daily_history.csv` はモニタリング用途のみなので実害は限定的。直すなら振り返り日以前に絞って再計算が必要。
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
- **CLAUDE.md は sync.yml で HF Space への push から除外済み**（ビルド不要なため）

---

## デプロイフロー

```
git push origin main
  → GitHub Actions: .github/workflows/sync.yml
    → rm -rf .git && git init（履歴リセット）
    → CLAUDE.md / HANDOVER*.md / HF_TOKEN.txt / .claude/ を削除（HF不要ファイル）
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
  - `keiba_model.pkl` — 学習済みモデルbundle（35.7MB）
  - `keiba_model_meta.json` — データ識別子（`data_mtime: "size:35515667"`形式）
  - `ai_daily_history.csv` — 振り返り日次成績
  - `discord_queue.json` — Discord通知キュー

- **data_mtime の仕組み**: ZIPのファイルサイズ (`size:XXXXX`) を使用
  - DockerのCOPYでOSのmtimeがリセットされるため、サイズで比較するよう変更済み（2026-03-25）
  - ZIPを更新したら必ず再学習 → HF Hub保存が必要

- **再学習が必要なタイミング**:
  - `learning_data_perfect_tier.zip` を更新したとき（data_mtimeが変わるため自動検知）
  - 特徴量・モデルパラメータなど学習パイプラインを変更したとき

- **再学習不要なタイミング**:
  - `inference.py` の推論ロジック変更（アンサンブル重み変更など）→ sync.yml だけで反映
  - `app.py` の UI 変更 → 同上

- **ローカル再学習コマンド**:
  ```powershell
  $env:HF_TOKEN="hf_xxxxxxxxxxxx"
  $env:HF_REPO_ID="ebi44323/keiba-ebye-models"
  python train_and_push.py
  ```

- **GitHub Actions 再学習**:
  GitHub → Actions → 「Retrain and Push Model」→「Run workflow」

- **HF SpaceのSecrets設定（必須）**:
  - `HF_TOKEN` — HuggingFace APIトークン（read/write権限）
  - `HF_REPO_ID` — `ebi44323/keiba-ebye-models`

---

## Discord 自動通知システム

### ワークフロー一覧
| ワークフロー | トリガー | 内容 |
|---|---|---|
| auto_morning.yml | 土日 8:00 JST | 全レース予想を .txt/.html で投稿 |
| auto_predict.yml | 土日 9:00〜16:45 JST 15分ごと | 発走10〜60分前のレースを自動予想 |
| auto_review.yml | 土日 17:00 JST | 当日振り返りを投稿 |
| auto_weekend_summary.yml | 日曜 20:00 JST | 土日合算まとめを投稿 |
| discord_notify.yml | 5分ごと | HF Hub キューを読んで Discord に送信 |

### ⚠️ GitHub Actions の制約
- cron の最小間隔は5分・実行ラグが数分あるため「発走5分前通知」は不可能
- 15分ポーリング × ウィンドウ幅50分（10〜60分前）で全レースをカバー

---

## 現在の状況（2026-04-01 時点）

### 2026-04-01 判明した事項
- **1か月分の振り返りデータ蓄積完了**（2026/02/28〜2026/03/29、10日分）
- **本命複勝回収率が非常に安定**: 10日中9日が100%超、平均~134%
- **AIスコアのキャリブレーション課題**: 本命平均AIスコア~22% に対し実際の勝者の平均AI勝率~13-14%。モデルが上位予測馬を過信気味。単勝回収率のバラつきと関連。
- **振り返り時の軽微なリーク確認**: `horse_course_dict` が全学習データ平均のため振り返り指標が若干楽観的（詳細はリーク防止セクション参照）

### 2026-03-30 完了した作業
- **コース統計フォールバックのリーク修正**: `transform('mean')` → expanding window に変更
- **`斤量_前走差` 特徴量追加**: 計51特徴量（効果は限定的だが追加済み）
- **Optuna多目的関数化**: AUC×0.7 + 正規化回収率×0.3 のブレンドスコア
- **Optunaチューニング実施**: CVスコア 0.677（n_estimators=700, min_child_samples=77）
- **アンサンブル重み最適化実施**: wa=0.058 / wb=0.816 / wc=0.126
- **inference.py・core_model.py の重み更新**: 両ファイルに反映済み
- **train_and_push.py 2重保存バグ修正**: `_save_model_to_hub` の重複呼び出しを削除
- **CLAUDE.md を HF Space から除外**: sync.yml に `rm -f CLAUDE.md` 追加

### 過去の主要完了済み修正
- EV優先◎をデフォルトに変更（app.py）
- EV推奨候補ウィジェット追加（EV>=1.0 の上位3馬カード表示）
- オッズキャッシュバスティング（リアルタイムオッズ化）
- Discord完全自動化（4スクリプト + 4ワークフロー）
- Isotonic Calibration追加
- 調教師名正規化（[東]/[西] プレフィックス除去）
- 騎手データベース追加（15タブ詳細分析）

### 今後の検証事項
- Discord自動通知の本番動作継続確認（4月以降）
- 本命複勝の安定性継続確認（現状10日間のみ・サンプル追加中）
- AIスコアキャリブレーション改善の検討（本命AI勝率~22% vs 実際勝者~13%のギャップ）
- EV優先 vs 標準◎ の実績比較継続

---

## 次にやること（優先度順）

### 高優先度
1. **週次データ更新** (`python update_data.py`) → 最新レース結果を追加 → 再学習
2. **Discord 自動通知の動作継続確認** → 4月の本番データで検証
3. **新アンサンブル重みの実戦検証** → 数週間後に回収率を確認（現状10日分は本命複勝が安定）

### 中優先度
4. **Optunaの再チューニング**（数週間の実戦データ蓄積後）
   - 新特徴量・新重みでの効果確認
5. **EV優先データの蓄積と比較検証**
   - 今後数週間の振り返りでEV優先 vs 標準◎ の実績を比較

### 低優先度
6. **モデルD（穴馬専用）の活用強化**
   - 現在◎への影響なし。穴馬複勝推奨への活用を検討
7. **騎手データベースの拡充**

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

# ローカル動作確認（HF_TOKEN・HF_REPO_ID を環境変数にセット後）
streamlit run app.py

# ローカル再学習 → HF Hub保存
# PowerShellで:
#   $env:HF_TOKEN="hf_xxxxxxxxxxxx"
#   $env:HF_REPO_ID="ebi44323/keiba-ebye-models"
python train_and_push.py
```
