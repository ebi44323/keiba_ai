# keiba-ebye プロジェクト

## 概要
競馬AI予測システム。Streamlit + LightGBM 3モデルアンサンブル。
HuggingFace Spaces にデプロイ。GitHub push → sync.yml → HF自動反映。

## バージョン管理ルール
- `app.py` 33行目: `st.caption("vYYYY-MM-DDx")` を**毎回の変更時に必ず更新**すること
- 形式: `v年-月-日+アルファベット`（同日複数回は a→b→c と増やす）
- 例: `v2026-03-28a`、`v2026-03-28b`
- **用途**: HF Space Filesタブでコード到達確認・アプリ画面で起動確認に使う
- 現在: `v2026-04-27b`

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

## Optunaチューニング結果（最新: 2026-04-03 実施）

- **CVスコア: 0.6949**（AUC×0.7 + 正規化回収率×0.3、3fold ウォークフォワードCV、50試行、市場勝率除外）
- ※ スコアは合成値のため純粋AUCとの直接比較不可。実質AUC≈0.75〜0.76
- 適用済みパラメータ（モデルB @ 2026-04-03）:
  ```json
  {"n_estimators":685,"learning_rate":0.029074,"num_leaves":15,
   "max_bin":228,"cat_smooth":39.95,"colsample_bytree":0.8848,
   "subsample":0.7033,"min_child_samples":70}
  ```
- `num_leaves:15`（前回32から大幅減）→ シンプルな木構造・過学習リスク軽減 ✅
- `learning_rate:0.029`（前回0.012から倍増）→ num_leaves減少とセットでバランス
- 前回結果（2026-03-30）: CVスコア 0.6770、n_estimators=700、num_leaves=32、min_child_samples=77

## アンサンブル重み最適化結果（2026-03-30 実施）

- **最適EV回収率: 74.1%**（最新20%ホールドアウトでの評価）
- ※ 訓練データ上の評価のため絶対値より重み比率が重要
- 結果: wa=0.0581 / wb=0.8159 / wc=0.1261
  - モデルB（1着予測）が支配的 → EV=AI勝率×オッズ に最も直結するため理にかなっている
- 適用箇所: `src/core_model.py` L331 と `src/inference.py` L461 の両方を更新済み

---

## 特徴量

### NUM_FEATURES（src/features_engine.py で一元管理）**計57特徴量**

| # | 特徴量名 | 概要 | 追加時期 |
|---|---|---|---|
| 1 | 枠番 | | 初期 |
| 2 | 馬番 | | 初期 |
| 3 | 年齢 | 性齢から数値抽出 | 初期 |
| 4 | 距離 | | 初期 |
| 5 | 斤量 | | 初期 |
| 6 | 出走頭数 | | 初期 |
| 7 | 馬体重_num | | 初期 |
| 8 | 馬体重増減 | | 初期 |
| 9 | 斤量差 | | 初期 |
| 10 | 斤量_前走差 | 今走-前走斤量（初出走時NaN）。均一斤量が多くモデル寄与は限定的 | 2026-03-30 |
| 11 | 休養日数 | | 初期 |
| 12 | 前走_着順 | | 初期 |
| 13 | 2走前_着順 | | 初期 |
| 14 | 3走前_着順 | | 初期 |
| 15 | 過去3走平均着順 | | 初期 |
| 16 | 前走着順パーセント | | 初期 |
| 17 | 直近3走着順パーセント | | 初期 |
| 18 | 前走_スピード指数 | | 初期 |
| 19 | 2走前_スピード指数 | | 初期 |
| 20 | 3走前_スピード指数 | | 初期 |
| 21 | 過去3走平均スピード指数 | | 初期 |
| 22 | 近5走_中央値スピード指数 | | 初期 |
| 23 | 近5走_最高スピード指数 | | 初期 |
| 24 | 上昇度_スピード指数 | | 初期 |
| 25 | 前走距離補正タイム差 | | 初期 |
| 26 | 前走上り偏差 | | 初期 |
| 27 | 位置取りショック | | 初期 |
| 28 | 同レース逃げ馬頭数 | | 初期 |
| 29 | 同レース先行馬頭数 | | 初期 |
| 30 | コース適性_着順パーセント | expanding mean（min_periods=3）でリーク防止 | 初期 |
| 31 | 乗り替わりフラグ | | 初期 |
| 32 | 馬場替わりフラグ | | 初期 |
| 33 | 距離変更フラグ | | 初期 |
| 34 | 前走失速フラグ | | 初期 |
| 35 | 前走大敗フラグ | | 初期 |
| 36 | 穴馬_距離変更一変 | | 初期 |
| 37 | 穴馬_馬場替わり一変 | | 初期 |
| 38 | 穴馬_勝負の乗り替わり | | 初期 |
| 39 | 穴馬_実力馬の巻き返し | | 初期 |
| 40 | キャリア数 | 累計出走回数（新馬・キャリア浅い馬の識別） | 初期 |
| 41 | 前走_上り順位率 | 前走末脚順位率（shift済み・リーク無し） | 初期 |
| 42 | 前走_前半ペース値 | 展開適性 | 初期 |
| 43 | 前走_後半ペース値 | 展開適性 | 初期 |
| 44 | 馬場指数 | 良=0〜不良=3 | 初期 |
| 45 | レースクラスコード | 新馬=0〜G1=9 | 初期 |
| 46 | ベスト3走_中央値スピード指数 | 近5走の上位3走median（ピーク能力） | 2026-03-25 |
| 47 | 長期休養フラグ | 休養180日以上（0/1） | 2026-03-25 |
| 48 | レース格上挑戦フラグ | 前走より上クラスへの出走（0/1） | 2026-03-25 |
| 49 | コース初挑戦フラグ | 競馬場×芝ダートの初出走（0/1） | 2026-03-25 |
| 50 | 近5走_スピード指数安定性 | 近5走の標準偏差（低いほど安定） | 2026-03-25 |
| 51 | 新馬フラグ | 初出走（キャリア数=0）: 0/1 | 初期 |
| 52 | 血統距離適性スコア | 父×距離カテゴリ×芝ダート別の歴史的着順パーセント | 初期 |
| 53 | 馬_重馬場_着順パーセント | 当該馬の重・不良馬場での過去着順パーセント（expanding mean） | 2026-04-07 |
| 54 | 父_重馬場_着順パーセント | 種牡馬産駒の重・不良馬場での着順パーセント（expanding mean） | 2026-04-07 |
| 55 | 逃げ_単独優位スコア | 逃げフラグ × max(0, 3-同レース逃げ馬頭数): 単独逃げほど高い | 2026-04-27 |
| 56 | 追込_展開向き度 | 差し・追込フラグ × 同レース逃げ馬頭数: ハイペース恩恵度 | 2026-04-27 |
| 57 | 前走_ペース補正スピード指数 | 前走SIをペース偏差×逃げフラグで補正（スロー逃げの過大評価抑制） | 2026-04-27 |
| 58 | 騎手_通算着順パーセント | 騎手の全期間 expanding mean 着順パーセント（shift+min_periods=5・リーク防止） | 2026-05-22 |
| 59 | 騎手_競馬場_着順パーセント | 騎手×競馬場 expanding mean（min_periods=3・不足時は通算でフォールバック） | 2026-05-22 |

⚠️ ユーザー方針: 騎手直近勝率・人気由来の特徴量は追加しない（後発情報・信頼性の問題）

### CAT_FEATURES
競馬場・芝ダート・天候・馬場・父系・母系・母父系・
前走芝ダート・回り・コース地形・脚質カテゴリ
（騎手×競馬場・騎手×距離は2026-05-22に削除。代わりにNUM_FEATURESの騎手能力特徴量を使用）

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

## 現在の状況（2026-07-25 時点）

### 2026-07-25 完了した作業（予想システム精度改善・6課題対応）
ユーザー提起の6つの課題に対応。**★再学習が必須**（勝率算出の刷新 + 未反映の騎手特徴量のため）。

- **勝率算出を絶対スコア基準へ全面刷新**（課題1・2・6の核心 / 要再学習）
  - **旧**: 推論時にレース内 min-max 正規化（`_sa=(_sa-_sa.min())/(...)`）→ 馬の絶対能力が消え常に相対順位のみ。さらに学習(全体min-max・T=1)と推論(レース内min-max・T=1.5)が**不整合**でIsotonic calibratorが誤適用されていた。
  - **新**: 学習データ全体のスコア分布(1〜99%tile)を固定基準に normalize し、その定数(`score_norms`)と softmax温度(`SOFTMAX_TEMPERATURE=1.0`)を bundle に保存。推論時も同一定数を再利用 → レース間で比較可能な「絶対的な強さ」になり、拮抗レースは本命勝率が下がり（旧31%→新~20%）、小頭数での弱い馬の過大評価が解消。学習・推論が完全一致しcalibratorが正しく効く。
  - bundle 追加: `_extra[7]=score_norms`（(lo,hi)×3モデル）, `_extra[8]=SOFTMAX_TEMPERATURE`
  - **後方互換**: 旧bundle(定数なし)では従来のレース内min-max+T=1.5にフォールバック → 再学習前でもアプリは動作する
- **EV優先モードを頭数連動で厳格化**（課題1・6 / 再学習不要・即反映）
  - EV昇格の勝率フロアを `max(0.25, 1.4/出走頭数, min_win_prob)` に（5頭:0.28 / 8頭以上:0.25）。小頭数で高オッズだけで人気薄が◎昇格する問題を抑制。
  - `新馬フラグ==0` 条件を撤廃（下記#3と整合）。勝率フロアで弱い未出走馬を自然に排除。
- **未出走馬の強制除外を撤廃**（課題3 / 即反映）
  - 旧: `新馬フラグ==1` を必ず上位5頭の外へ強制ソート + EV昇格対象外 + 「🛑見送り推奨」上書き。
  - 新: 純粋に勝率(AI予測)順でソートしモデル評価(`新馬フラグ`/`血統距離適性スコア`)に委ねる。未出走混在時は confidence_text に注意書きを付記するのみ（強制見送りしない）。
- **勝負/回避レースラベルを追加**（課題5 / 即反映）
  - confidence_text 先頭に `🔥勝負レース`（本命p1>=0.25 かつ score_diff>=0.10）/ `⚠️回避（様子見）レース`（未出走混在 or 拮抗）/ `🟡通常レース` を明示。全通知経路(app/朝刊/直前/振り返り)に自動反映。
- **馬体重の影響について（課題6の回答）**: 馬体重は59特徴量中2つ(`馬体重_num`/`馬体重増減`)で影響は限定的。朝刊/直前の激変の**主因はオッズ変動→EV優先の◎入替**。上記EV厳格化 + 絶対スコア化(小差増幅の解消)で緩和される。
- バージョン: `v2026-07-25a`

### ★次の必須アクション
1. **再学習 → HF Hub保存**（`python train_and_push.py` または GitHub Actions "Retrain and Push Model"）
   - これで `score_norms`/`SOFTMAX_TEMPERATURE` が bundle に入り、絶対スコア勝率が有効化される
   - 同時に 2026-05-22 の騎手特徴量も反映される
2. 再学習後、数週間の振り返りで小頭数・朝刊/直前差・回収率を検証

---

## 現在の状況（2026-05-22 時点）

### 2026-05-22 完了した作業
- **不具合修正1**: `src/config.py` `get_headers()` にReferer等完全なHTTPヘッダーを追加
  → GitHub Actionsがnetkeibaにブロックされ2026-04-20以降のデータ取得が止まっていた根本原因を修正
- **不具合修正2**: `sync.yml` に `keiba_model.pkl` の除外を追加
  → 34.9MBのpklが含まれHuggingFace Spacesデプロイが失敗していた
- **データ更新**: `learning_data_perfect_tier.zip` を2026-05-17まで更新（197,877行 → 205,535行）
- **騎手能力特徴量を追加（再学習必要）**:
  - CAT_FEATURES から `騎手_競馬場`・`騎手_距離` を削除（importance 54%独占問題 + 学習/推論間のID/名前不整合バグを修正）
  - NUM_FEATURES に `騎手_通算着順パーセント`（全期間expanding mean）・`騎手_競馬場_着順パーセント`（騎手×会場expanding mean）を追加
  - bundleに `jockey_overall_dict`・`jockey_venue_dict` を追加（`_extra[5][6]`）
- **recover_april.py**: 4月・5月初旬の欠落振り返りデータを再計算するスクリプト追加

## 現在の状況（2026-04-27 時点）

### 2026-04-27 完了した作業（後半）
- **モデル精度改善: 展開×脚質 交互作用特徴量 3つ追加**（features_engine.py + inference.py、計54特徴量）
  - `逃げ_単独優位スコア`: 逃げフラグ × max(0, 3-同レース逃げ馬頭数) → 単独逃げで最大値、競合多いと0
  - `追込_展開向き度`: 差し・追込フラグ × 同レース逃げ馬頭数 → ハイペース時の差し馬恩恵を明示
  - `前走_ペース補正スピード指数`: 前走SI - 逃げフラグ × ペース偏差 × 2.0（スロー逃げのSI過大評価を補正）
  - ⚠️ **再学習が必要**（特徴量追加のため retrain.yml または train_and_push.py を実行すること）
- **Temperatureスケーリングを 1.0 → 1.5 に変更**（inference.py L592、再学習不要）
  - 実績: 本命平均AIスコア25% vs 実際勝者14.5% = 1.72倍過信 → T=1.5で15-16%に圧縮
- **三連複の買い方変更**: ◎〇▲ BOX 1点 → ◎軸 × 2〜5位ながし 6点（auto_review.py + app.py）
  - 理由: 3頭BOX1点は7日間連続0%。6点流しで的中確率を大幅向上
- **バージョン**: `v2026-04-27b`

### 2026-04-27 完了した作業（前半）
- **Discord自動通知をすべて直接送信に統一** (`auto_review.py` / `auto_weekend_summary.py`)
  - 旧: HF Hubキュー書き込み → `discord_notify.yml` が5分後に配送（queue依存）
  - 新: GitHub Actionsから直接 Discord Webhook にPOST（predict_auto.py と同方式）
  - `send_discord_review`（discord_utils.py）への依存を削除
  - `_send_review_direct()` をauto_review.pyに内包
  - `DISCORD_WEBHOOK_URL` / `DISCORD_REVIEW_WEBHOOK_URL` を各スクリプトで直接参照
- **YAMLのstreamlitバージョン不一致修正**: `auto_review.yml`/`auto_weekend_summary.yml` の `streamlit==1.55.0` → `1.56.0`
- **app.py 振り返りタブに累積競馬場別成績を追加**
  - `ai_daily_history.csv` の `競馬場別` カラム（JSON）を全日分パース・集計
  - 競馬場ごとの R数・的中数・勝率・単勝回収率をバーチャートと表で表示
  - 長期成績ダッシュボード末尾（週別ヒートマップの後）に配置
- **バージョン**: `v2026-04-27a`

### 2026-04-15 完了した作業
- **Discord自動予想通知を直接送信に変更** (`predict_auto.py`)
  - 旧: HF Hubキュー書き込み → `discord_notify.yml` が5分後に配送（2ステップ）
  - 新: GitHub Actionsから直接 `DISCORD_WEBHOOK_URL` にPOST（1ステップ）
  - `send_discord_prediction`（discord_utils.py）への依存を削除
  - `DISCORD_WEBHOOK_URL` は `auto_predict.yml` のenvに定義済み
- **朝刊HTMLフォーマットを `src/reports.py` に統一** (`auto_morning.py`)
  - 旧: `auto_morning.py` 内の独自 `format_race_txt`/`format_race_html_row` 関数
  - 新: `src/reports.py` の `generate_pdf_report`/`generate_txt_report` を使用（アプリ内ダウンロードと同じ形式）
  - `track_type` / `distance` も `run_real_prediction` 戻り値から取得してレポートに含める
  - Geminiデータのキーマッピングも対応（`honmei`→`gemini_honmei` 等）

### 2026-04-03 完了した作業
- **Optunaチューニング実施（2回目）**: CVスコア 0.6949（前回比 +0.018 改善）
  - `num_leaves: 32→15`（大幅シンプル化・過学習リスク軽減）
  - `learning_rate: 0.0123→0.0291`（num_leaves減少とセットでバランス調整）
  - `src/core_model.py` のモデルBパラメータ更新済み
  - ⚠️ **再学習が必要**（パラメータ変更のため train_and_push.py を実行すること）

### 2026-04-01 判明した事項
- **1か月分の振り返りデータ蓄積完了**（2026/02/28〜2026/03/29、10日分）
- **本命複勝回収率が非常に安定**: 10日中9日が100%超、平均~134%
- **AIスコアのキャリブレーション課題**: 本命平均AIスコア~22% に対し実際の勝者の平均AI勝率~13-14%。モデルが上位予測馬を過信気味。単勝回収率のバラつきと関連。
- **振り返り時の軽微なリーク確認**: `horse_course_dict` が全学習データ平均のため振り返り指標が若干楽観的（詳細はリーク防止セクション参照）

### 2026-03-30 完了した作業
- **コース統計フォールバックのリーク修正**: `transform('mean')` → expanding window に変更
- **`斤量_前走差` 特徴量追加**: 計51特徴量（効果は限定的だが追加済み）
- **Optuna多目的関数化**: AUC×0.7 + 正規化回収率×0.3 のブレンドスコア
- **Optunaチューニング実施（1回目）**: CVスコア 0.677（n_estimators=700, min_child_samples=77）
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
1. **【必須】再学習 → HF Hub保存**（騎手特徴量変更を反映するため必須）
   - GitHub Actions → 「Retrain and Push Model」→「Run workflow」
   - ※ ローカル実行なら push 前でも可（`python train_and_push.py`）
3. **週次データ更新** (`python update_data.py`) → 最新レース結果を追加 → 再学習
3. **Discord 自動通知の動作継続確認** → 4月の本番データで検証
4. **新パラメータの実戦検証** → 数週間後に回収率を確認

### 中優先度
5. **EV優先データの蓄積と比較検証**
   - 今後数週間の振り返りでEV優先 vs 標準◎ の実績を比較
6. **Optunaの再チューニング**（さらに実戦データ蓄積後）

### 低優先度
7. **モデルD（穴馬専用）の活用強化**
   - 現在◎への影響なし。穴馬複勝推奨への活用を検討
8. **騎手データベースの拡充**

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
