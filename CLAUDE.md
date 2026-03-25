# keiba-ebye プロジェクト

## 概要
競馬AI予測システム。Streamlit + LightGBM 3モデルアンサンブル。
HuggingFace Spaces にデプロイ。GitHub push → sync.yml → HF自動反映。

---

## ファイル構成

```
app.py                     メインStreamlit UI
features_engine.py         src/features_engine.py の re-export のみ（実体はsrc側）
update_data.py             週次データ更新スクリプト（python update_data.py で実行）
backfill_race_name.py      レース名補完スクリプト
check_market_rate_auc.py   市場勝率のAUC寄与検証スクリプト

src/
  __init__.py              パッケージ宣言（HuggingFace Linux環境に必須）
  config.py                ★共通定数・関数（PLACE_DICT/VENUE_MAWARI等）の一元管理
  features_engine.py       特徴量定義(NUM_FEATURES/CAT_FEATURES)＋create_features()
  core_model.py            モデル学習・HF Hub保存/ロード
  inference.py             リアルタイム推論・スクレイピング
  optuna_tuner.py          Optunaチューニング（ウォークフォワードCV・AUC目的関数）
  backtest.py              時系列バックテスト
  scraper.py               netkeiba/Yahoo競馬スクレイピング
  discord_utils.py         Discord通知（HFキュー経由・現在動作不安定）
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
コース初挑戦フラグ・近5走_スピード指数安定性（**計50特徴量**）

追加特徴量の概要（2026-03-25 追加）:
- `ベスト3走_中央値スピード指数`: 近5走の上位3走median（ピーク能力）
- `長期休養フラグ`: 休養180日以上（0/1）
- `レース格上挑戦フラグ`: 前走より上クラスへの出走（0/1）
- `コース初挑戦フラグ`: 競馬場×芝ダートの初出走（0/1）
- `近5走_スピード指数安定性`: 近5走の標準偏差（低いほど安定）

⚠️ ユーザー方針: 騎手直近勝率・人気由来の特徴量は追加しない（後発情報・信頼性の問題）

### CAT_FEATURES
競馬場・芝ダート・天候・馬場・父系・母系・母父系・
前走芝ダート・回り・コース地形・脚質カテゴリ・騎手×競馬場・騎手×距離

### Target Encoding（TE_COLS）
調教師・父・母父・騎手

---

## リーク防止の実装

- **コース統計**: `shift(1).expanding(min_periods=3).mean()` で過去データのみ使用
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

---

## 現在の状況（2026-03-25 時点）

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
- **Optuna結果をモデルBに適用**（CV AUC 0.7596, 50試行 @ 2026-03-24）
- **Isotonic Calibration追加**（core_model.py学習時にfitting、inference.py推論時に適用）
  - bundle要素数: 18→19（calibratorを末尾に追加）
  - 後方互換: bundle展開を `*_extra` で受ける形に変更（app.py 2箇所・inference.py）
- **新特徴量5種追加**（計50特徴量）: ベスト3走_中央値スピード指数・長期休養フラグ・
  レース格上挑戦フラグ・コース初挑戦フラグ・近5走_スピード指数安定性

### 3月振り返り結果から判明した課題（2026-03-25）
- 本命単勝回収率: 39〜88%（全日100%未満）→ 本命一辺倒では儲からない
- 穴馬単勝回収率: 245.6%など大きいが、**0%の日が多い（穴馬をほとんど推奨しない）**
- 穴馬複勝回収率: 185.7%・134.4% → 穴馬を複勝圏には入れているが◎にしていない
- **根本原因**: 現在◎=max(AI勝率)。ランキングモデルは市場合意と相関するため1番人気多発

### 次にやること（優先度順）
1. **◎選択ロジックをEV優先に変更**（最重要・穴馬問題の本質的解決）
   - 現在: ◎ = max(AI勝率)
   - 改善: EV=AI勝率×オッズ が1.0超かつAI勝率>=0.10 の馬がいれば、その馬を◎に
   - 実装箇所: `src/inference.py` の ◎判定ロジック（app.pyにもUI切り替えスイッチを追加）
2. **穴馬推奨モード（モデルD）の検討**
   - 目的変数: 着順=1 かつ 人気>=5（穴馬が勝つケース）
   - 特徴量: 長期休養フラグ・コース初挑戦フラグ・レース格上挑戦フラグ・ベスト3走スピード指数など
3. git push → HF デプロイ（現在の変更を本番反映）
4. Discord通知キューの安定化（GitHub Actions 5分ポーリング → Webhook直接化を検討）

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
```
