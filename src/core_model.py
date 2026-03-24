import os
import json
import datetime
import pandas as pd
import numpy as np
import lightgbm as lgb
import logging
import pytz
import random
import time
import re
import streamlit as st
from src.features_engine import NUM_FEATURES, CAT_FEATURES, TE_COLS, classify_style
from src.utils import VENUE_MAWARI, VENUE_CHIKEI, TRACK_CONDITION_MAP, classify_race_class

logger = logging.getLogger('keiba_ebye')

_HF_TOKEN   = os.environ.get("HF_TOKEN", "")
_HF_REPO_ID = os.environ.get("HF_REPO_ID", "")   
_MODEL_FILE = "keiba_model.pkl"                    
_META_FILE  = "keiba_model_meta.json"             

def _get_zip_mtime():
    """学習データZIPの最終更新日時(文字列)を返す"""
    for p in ['learning_data_perfect_tier.zip', 'learning_data_perfect_tier.csv']:
        if os.path.exists(p):
            import time as _t
            return _t.strftime('%Y-%m-%dT%H:%M:%S', _t.gmtime(os.path.getmtime(p)))
    return 'unknown'

def _try_load_model_from_hub():
    """
    HF Hubからモデルをロードする。
    【重要】repo_type="dataset" を使用。
    "space"を使うとファイルアップロード時にSpaceが再起動してしまう。
    返値: (model, features, cat_features, ...) タプル or None
    """
    if not _HF_TOKEN or not _HF_REPO_ID:
        return None
    try:
        import joblib
        from huggingface_hub import hf_hub_download

        # メタデータを確認: データが更新されていれば再学習
        try:
            meta_path = hf_hub_download(
                repo_id=_HF_REPO_ID, filename=_META_FILE,
                repo_type="dataset", token=_HF_TOKEN, cache_dir="/tmp/hf_cache"
            )
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            hub_data_mtime = meta.get('data_mtime', '')
            local_mtime    = _get_zip_mtime()
            if local_mtime != hub_data_mtime:
                return None  # データが更新されているので再学習
        except Exception as _e:
            logger.info(f'HF Hubメタデータなし（初回）: {_e}')  # 初回 → そのままロードを試みる

        # モデル本体をロード
        model_path = hf_hub_download(
            repo_id=_HF_REPO_ID, filename=_MODEL_FILE,
            repo_type="dataset", token=_HF_TOKEN, cache_dir="/tmp/hf_cache"
        )
        bundle = joblib.load(model_path)
        return bundle
    except Exception:
        return None  # Hubにモデルなし or エラー → 学習へ

def _save_model_to_hub(bundle):
    """
    学習済みモデルをHF Dataset Hubにアップロードする。
    【重要】repo_type="dataset" を使用。
    "space"を使うとアップロードのたびにSpaceが再起動してしまう。
    bundle: prepare_model_and_data() の返値タプル
    """
    if not _HF_TOKEN or not _HF_REPO_ID:
        return False
    try:
        import joblib, io
        from huggingface_hub import HfApi
        api = HfApi(token=_HF_TOKEN)

        # Datasetリポジトリが存在しなければ自動作成
        try:
            api.create_repo(repo_id=_HF_REPO_ID, repo_type="dataset",
                            private=True, exist_ok=True, token=_HF_TOKEN)
        except Exception:
            pass

        # モデルをバイトにシリアライズしてアップロード
        buf = io.BytesIO()
        joblib.dump(bundle, buf)
        buf.seek(0)
        api.upload_file(
            path_or_fileobj=buf,
            path_in_repo=_MODEL_FILE,
            repo_id=_HF_REPO_ID,
            repo_type="dataset",          # ← Spaceではなくdataset!
            commit_message=f"モデル更新 {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
            token=_HF_TOKEN,
        )

        # メタデータ保存
        meta = {
            'data_mtime': _get_zip_mtime(),
            'trained_at': datetime.datetime.now().isoformat(),
        }
        meta_buf = io.BytesIO(json.dumps(meta, ensure_ascii=False, indent=2).encode())
        api.upload_file(
            path_or_fileobj=meta_buf,
            path_in_repo=_META_FILE,
            repo_id=_HF_REPO_ID,
            repo_type="dataset",
            commit_message="モデルメタデータ更新",
            token=_HF_TOKEN,
        )
        return True
    except Exception:
        return False

@st.cache_resource
def prepare_model_and_data(force_retrain=False):
    """
    force_retrain=True: Hubのキャッシュを無視して強制再学習
    """
    # ── HF Hubからロードを試みる ─────────────────────────────
    if not force_retrain:
        cached = _try_load_model_from_hub()
        if cached is not None:
            return cached  # キャッシュ済みモデルを即返す

    # ── 以下: 学習処理 ────────────────────────────────────────
    num_features = list(NUM_FEATURES)
    cat_features = list(CAT_FEATURES)
    te_cols = list(TE_COLS)

    # ── 追加特徴量（CSVに存在するが未使用だったもの）──────────────
    EXTRA_NUM = [
        'キャリア数',        # 累計出走回数（新馬・1勝馬の識別）
        # '上り順位率' は除外: 学習時に現レースの上り3Fから計算されるリーク特徴量
        #   → 前走_上り順位率（shift済み）のみ使用
        '前走_上り順位率',   # 前走末脚順位（安全: 前走データのshift）
        '前走_前半ペース値', # 前走前半ペース（展開適性）
        '前走_後半ペース値', # 前走後半ペース（展開適性）
        '馬場指数',          # 馬場状態の数値（良=0〜不良=3）
        'レースクラスコード', # レースグレード（新馬=0〜G1=9）
        '市場勝率',          # 単勝オッズの逆数（市場評価・歪み検出用）
    ]
    for f in EXTRA_NUM:
        if f not in num_features:
            num_features.append(f)

    try:
        df = pd.read_csv('learning_data_perfect_tier.zip', compression='zip', dtype=str)
    except FileNotFoundError:
        df = pd.read_csv('learning_data_perfect_tier.csv', dtype=str)

    from src.features_engine import create_features
    df, _ = create_features(df)

    # latest_horse_data: ★修正 - 推論で必要な全列を保存
    df_latest = df.groupby('馬ID').tail(1).copy()
    rn = {'着順':'最新_着順','スピード指数':'最新_スピード指数','人気':'最新_人気','上り':'最新_上り',
          '距離':'最新_距離','斤量':'最新_斤量','馬体重_num':'最新_馬体重','日付':'最新_日付','通過':'最新_通過',
          '騎手':'最新_騎手','芝/ダート':'最新_芝ダート','着順パーセント':'最新_着順パーセント'}
    for src,dst in [('前走失速フラグ','最新_失速フラグ'),('失速フラグ','最新_失速フラグ'),
                    ('前走上り偏差','最新_上り偏差'),('前走距離補正タイム差','最新_距離補正タイム差'),
                    ('直近3走着順パーセント','最新_直近3走着順パーセント'),('馬体重増減','最新_馬体重増減')]:
        if src in df_latest.columns: rn[src]=dst
    df_latest = df_latest.rename(columns=rn)
    ck = ['馬ID','父','父系','母','母系','母父','母父系',
          '最新_着順','最新_スピード指数','最新_人気','最新_上り','最新_距離','最新_斤量','最新_馬体重','最新_日付','最新_通過',
          '最新_騎手','最新_芝ダート','最新_着順パーセント',
          '最新_失速フラグ','最新_上り偏差','最新_距離補正タイム差','最新_直近3走着順パーセント','最新_馬体重増減',
          '前走_着順','2走前_着順','3走前_着順','過去3走平均着順',
          '前走_スピード指数','2走前_スピード指数','3走前_スピード指数','4走前_スピード指数','5走前_スピード指数',
          '過去3走平均スピード指数','近5走_中央値スピード指数','近5走_最高スピード指数','上昇度_スピード指数',
          '前走_通過','2走前_通過','前走_最終コーナー','2走前_最終コーナー',
          'キャリア数','前走_上り順位率']  # 上り順位率は除外（現レースリーク）
    ck = [c for c in ck if c in df_latest.columns]
    latest_horse_data = df_latest[ck].copy()
    horse_course_dict = df.groupby(['馬ID','競馬場','芝/ダート'])['着順パーセント'].mean().to_dict()

    df_valid = df.dropna(subset=['着順','単勝']).copy()
    df_valid['馬券内'] = (df_valid['着順']<=3).astype(int)
    for col in num_features:
        if col not in df_valid.columns: df_valid[col] = np.nan
        else:
             # ★追加: 必ず数値型（float等）に変換する安全策
             df_valid[col] = pd.to_numeric(df_valid[col], errors='coerce')

    cat_categories_dict = {}
    for col in cat_features:
        if col not in df_valid.columns: df_valid[col] = '不明'
        df_valid[col] = df_valid[col].fillna('不明').astype('category')
        cats = list(df_valid[col].cat.categories)
        if '不明' not in cats: cats.append('不明')
        cat_categories_dict[col] = cats

    known_jockeys  = df_valid['騎手'].dropna().unique().tolist()
    known_trainers = df_valid['調教師'].dropna().unique().tolist()
    df_valid = df_valid.sort_values(['レースID','馬番'])

    max_date = df_valid['日付'].max()
    test_start_date = max_date - pd.Timedelta(days=30)
    train_df = df_valid[df_valid['日付']<test_start_date].copy()
    test_df  = df_valid[df_valid['日付']>=test_start_date].copy()

    # ★修正BUG2: TE は TE_COLS と完全一致
    te_dicts = {}
    global_mean = train_df['馬券内'].mean()
    for col in te_cols:
        if col not in train_df.columns: continue
        te_dicts[col] = train_df.groupby(col)['馬券内'].mean().to_dict()
        train_df[f'{col}_TE'] = train_df[col].map(te_dicts[col]).fillna(global_mean)
        test_df[f'{col}_TE']  = test_df[col].map(te_dicts[col]).fillna(global_mean)
        if f'{col}_TE' not in num_features: num_features.append(f'{col}_TE')

    features = [f for f in (cat_features+num_features) if f in train_df.columns]
    train_groups = train_df.groupby('レースID',sort=False).size().values
    test_groups  = test_df.groupby('レースID',sort=False).size().values

    # ★修正BUG3: model定義とfitの間に無関係コードを入れない
    # ── モデルA: 複勝(3着以内)Ranker ─────────────────────────────
    model = lgb.LGBMRanker(n_estimators=500, learning_rate=0.01, num_leaves=63, max_bin=255,
                            cat_smooth=10, random_state=42, importance_type='gain',
                            colsample_bytree=0.7, subsample=0.8)
    model.fit(train_df[features], train_df['馬券内'], group=train_groups,
              categorical_feature=[f for f in cat_features if f in features],
              eval_set=[(test_df[features], test_df['馬券内'])], eval_group=[test_groups])

    # ── モデルB: 1着予測Ranker（アンサンブル用）──────────────────
    train_df['win_label'] = (train_df['着順'] == 1).astype(int) if '着順' in train_df.columns else train_df['馬券内']
    test_df['win_label']  = (test_df['着順']  == 1).astype(int) if '着順' in test_df.columns  else test_df['馬券内']
    # ── モデルBパラメータ: Optunaチューニング済み（ウォークフォワードCV版）──
    # 前回の229%はリーク込みのため参考程度。再チューニング後に置き換え推奨。
    model_win = lgb.LGBMRanker(
        n_estimators=477,
        learning_rate=0.020536,
        num_leaves=19,
        max_bin=197,
        cat_smooth=36.63,
        colsample_bytree=0.6777,
        subsample=0.6289,
        min_child_samples=20,
        random_state=123,
        importance_type='gain',
    )
    model_win.fit(train_df[features], train_df['win_label'], group=train_groups,
                  categorical_feature=[f for f in cat_features if f in features],
                  eval_set=[(test_df[features], test_df['win_label'])], eval_group=[test_groups])


    # ── モデルC: 着順パーセント予測Regressor（アンサンブル用）────────────────
    model_reg = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=31, max_bin=255,
                                  random_state=777,
                                  colsample_bytree=0.7, subsample=0.8)
    model_reg.fit(train_df[features], train_df['着順パーセント'].fillna(0.5),
                  categorical_feature=[f for f in cat_features if f in features])

    # ── アンサンブルスコア ────────────────────────────────────────
    score_a = model.predict(test_df[features])
    score_b = model_win.predict(test_df[features])
    score_c = 1.0 - model_reg.predict(test_df[features])  # 低い方が上位なので反転

    def _norm_scores(s):
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn + 1e-9)
    _sa_norm = _norm_scores(score_a)
    _sb_norm = _norm_scores(score_b)
    _sc_norm = _norm_scores(score_c)

    # 複勝0.35, 1着0.5, 着順回帰0.15
    test_df['予測スコア'] = _sa_norm * 0.35 + _sb_norm * 0.50 + _sc_norm * 0.15
    test_df['exp_score'] = np.exp(test_df['予測スコア']-test_df.groupby('レースID')['予測スコア'].transform('max'))
    test_df['AI勝率'] = test_df['exp_score']/test_df.groupby('レースID')['exp_score'].transform('sum')
    top_preds = test_df.sort_values(['レースID','AI勝率'],ascending=[True,False]).groupby('レースID').head(1)
    win_hits  = top_preds[pd.to_numeric(top_preds['着順'],errors='coerce')==1]
    invest_amount = len(top_preds)*100
    win_return = (pd.to_numeric(win_hits['単勝'],errors='coerce')*100).sum()
    recent_return_rate = (win_return/invest_amount*100) if invest_amount>0 else 0

    # ── AUC計算（コース統計リーク修正後: expanding window使用で信頼性向上）────
    auc_win = auc_place = 0.0
    try:
        from sklearn.metrics import roc_auc_score
        test_df['win_true']   = (pd.to_numeric(test_df['着順'], errors='coerce') == 1).astype(int)
        test_df['place_true'] = (pd.to_numeric(test_df['着順'], errors='coerce') <= 3).astype(int)
        auc_win   = roc_auc_score(test_df['win_true'],   _sb_norm)
        auc_place = roc_auc_score(test_df['place_true'], _sa_norm)
        logger.info(f'モデルAUC: 1着={auc_win:.4f} / 複勝={auc_place:.4f}')
    except Exception as _e:
        logger.warning(f'AUC計算失敗: {_e}')

    try:
        ped_df = pd.read_csv('pedigree_master_all.csv', dtype=str)
        ped_df['馬ID'] = ped_df['馬ID'].astype(str).str.zfill(10)
        ped_dict = ped_df.set_index('馬ID')[['父','父系','母','母系','母父','母父系']].to_dict('index')
    except Exception as _e:
        logger.warning(f'pedigree_master_all.csv 読み込み失敗: {_e}')
        ped_dict = {}

    best_weight = 0.35  # 後方互換性および参照用
    bundle = (model, model_win, model_reg, features, cat_features, num_features, cat_categories_dict,
              latest_horse_data, horse_course_dict, ped_dict,
              known_jockeys, known_trainers, te_dicts, global_mean, recent_return_rate, best_weight,
              auc_win, auc_place)

    # ── HF Hubにアップロード ──────────────────────────────────
    _save_model_to_hub(bundle)

    return bundle
