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
        'キャリア数',       # 累計出走回数（新馬・1勝馬の識別）
        '上り順位率',       # レース内末脚順位（0〜1、低いほど末脚◎）
        '前走_上り順位率',  # 前走末脚順位
        '前走_前半ペース値', # 前走前半ペース（展開適性）
        '前走_後半ペース値', # 前走後半ペース（展開適性）
        '馬場指数',          # 馬場状態の数値（良=0〜不良=3）
        'レースクラスコード', # レースグレード（新馬=0〜G1=9）
        '市場勝率',          # 単勝オッズの逆数（大衆の評価する勝率）
    ]
    for f in EXTRA_NUM:
        if f not in num_features:
            num_features.append(f)

    try:
        df = pd.read_csv('learning_data_perfect_tier.zip', compression='zip', dtype=str)
    except FileNotFoundError:
        df = pd.read_csv('learning_data_perfect_tier.csv', dtype=str)

    df['日付'] = pd.to_datetime(df['日付'], format='mixed', errors='coerce')
    df = df.dropna(subset=['日付'])

    for col in ['着順','単勝','人気','斤量','距離','上り','枠番','馬番']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['性別'] = df['性齢'].astype(str).str.extract(r'([牡牝セ])')[0]
    df['年齢'] = pd.to_numeric(df['性齢'].astype(str).str.extract(r'(\d+)')[0], errors='coerce')
    df['馬体重_num'] = pd.to_numeric(df['馬体重'].astype(str).str.extract(r'(\d+)')[0], errors='coerce')

    # ★修正BUG1: CSV の新列を正しく数値変換（学習時）
    for col in ['馬体重増減','斤量差','前走着順パーセント','直近3走着順パーセント',
                '前走距離補正タイム差','前走上り偏差','乗り替わりフラグ','馬場替わりフラグ',
                '距離変更フラグ','前走失速フラグ','前走大敗フラグ',
                '穴馬_距離変更一変','穴馬_馬場替わり一変','穴馬_勝負の乗り替わり','穴馬_実力馬の巻き返し','当日馬体重']:
        if col in df.columns:
            df[col] = df[col].replace({'True':1,'False':0,'true':1,'false':0})
            df[col] = df[col].astype(str).str.replace('%','',regex=False).str.replace(',','',regex=False).str.replace('+','',regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if '当日馬体重' in df.columns:
        df['馬体重_num'] = df['当日馬体重'].combine_first(df['馬体重_num'])

    def t2s(t):
        try:
            m = re.match(r'(\d+):(\d+\.\d+)', str(t))
            return float(m.group(1))*60+float(m.group(2)) if m else float(t)
        except: return np.nan
    df['走破タイム秒'] = df['タイム'].apply(t2s)

    df['出走頭数'] = df.groupby('レースID')['馬ID'].transform('count')
    df['着順パーセント'] = (df['着順']-1)/(df['出走頭数']-1).replace(0,1)

    # ★修正: マージキーの型を統一してからマージ（型不一致でマージ失敗→列不在エラーを防ぐ）
    for _c in ['競馬場','芝/ダート','距離']:
        df[_c] = df[_c].astype(str).str.strip()
    cs = df.groupby(['競馬場','芝/ダート','距離'])['走破タイム秒'].agg(['mean','std']).reset_index()
    cs.columns = ['競馬場','芝/ダート','距離','コース平均','コース標準偏差']
    # 既に同名列があれば先に除去（二重マージ防止）
    for _c in ['コース平均','コース標準偏差']:
        if _c in df.columns: df = df.drop(columns=[_c])
    df = pd.merge(df, cs, on=['競馬場','芝/ダート','距離'], how='left')
    # ★追加：マージが終わったら、AIが計算できるように「距離」を数値型に戻す
    df['距離'] = pd.to_numeric(df['距離'], errors='coerce')
    # マージ後に列が存在しない場合の安全フォールバック
    if 'コース標準偏差' not in df.columns:
        df['コース平均'] = df['走破タイム秒'].mean()
        df['コース標準偏差'] = df['走破タイム秒'].std()
    df['スピード指数'] = np.where(df['コース標準偏差'].fillna(0)>0,
        50-((df['走破タイム秒']-df['コース平均'])/df['コース標準偏差'])*10, 50)
    df['調教師_騎手'] = df['調教師'].astype(str)+'_'+df['騎手'].astype(str)

    # ── 馬場指数（学習データに馬場列があれば使用）────────────────────
    if '馬場' in df.columns:
        df['馬場指数'] = df['馬場'].map(TRACK_CONDITION_MAP).fillna(0).astype(float)
    else:
        df['馬場指数'] = 0.0  # データなし時はデフォルト「良」

    # ── レースクラスコード（レース名から判別）────────────────────────
    if 'レース名' in df.columns:
        df['レースクラスコード'] = df['レース名'].apply(classify_race_class).astype(float)
    else:
        df['レースクラスコード'] = 5.0  # デフォルト オープン

    # ── 市場勝率（単勝オッズの逆数 = 大衆が評価する勝率）──────────────
    # 学習データの単勝列は単勝払戻額(例:1230→12.3倍)ではなく
    # オッズ値(例:5.3)として扱う（run_real_predictionの回収計算と対応）
    df['市場勝率'] = pd.to_numeric(df['単勝'], errors='coerce').replace(0, np.nan)
    df['市場勝率'] = (1.0 / df['市場勝率']).clip(0, 1)

    df = df.sort_values(['馬ID','日付']).reset_index(drop=True)

    # ── 新特徴量1: キャリア数（累計出走回数）─────────────────────
    # 新馬・キャリア浅い馬の識別に使う（スピード指数等がNaNの馬を正しく評価）
    df['キャリア数'] = df.groupby('馬ID').cumcount()  # 0始まり（初出走=0）

    # ── 新特徴量2: 上り順位（レース内での末脚の相対順位）──────────
    # 上り絶対値から順位を計算（1=最も末脚が速い）
    if '上り' in df.columns:
        df['上り'] = pd.to_numeric(df['上り'], errors='coerce')
        df['上り順位'] = df.groupby('レースID')['上り'].rank(method='min', ascending=True)
        df['上り順位率'] = df['上り順位'] / df['出走頭数']  # 0〜1で正規化（低いほど末脚◎）
    else:
        df['上り順位'] = np.nan
        df['上り順位率'] = np.nan

    # ── 新特徴量3: 前走上り順位（前走での末脚順位）─────────────────
    df['前走_上り順位率'] = df.groupby('馬ID')['上り順位率'].shift(1)

    # ── 新特徴量4: 前半・後半ペース値（展開適性）────────────────────
    for col in ['前半ペース値', '後半ペース値']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[f'前走_{col}'] = df.groupby('馬ID')[col].shift(1)

    df['前走_着順']  = df.groupby('馬ID')['着順'].shift(1)
    df['2走前_着順'] = df.groupby('馬ID')['着順'].shift(2)
    df['3走前_着順'] = df.groupby('馬ID')['着順'].shift(3)
    df['過去3走平均着順'] = df[['前走_着順','2走前_着順','3走前_着順']].mean(axis=1)
    for n in [1,2,3,4,5]:
        df[f'{"前走" if n==1 else str(n)+"走前"}_スピード指数'] = df.groupby('馬ID')['スピード指数'].shift(n)
    df['前走_スピード指数']  = df.groupby('馬ID')['スピード指数'].shift(1)
    df['2走前_スピード指数'] = df.groupby('馬ID')['スピード指数'].shift(2)
    df['3走前_スピード指数'] = df.groupby('馬ID')['スピード指数'].shift(3)
    df['4走前_スピード指数'] = df.groupby('馬ID')['スピード指数'].shift(4)
    df['5走前_スピード指数'] = df.groupby('馬ID')['スピード指数'].shift(5)
    df['過去3走平均スピード指数']  = df[['前走_スピード指数','2走前_スピード指数','3走前_スピード指数']].mean(axis=1)
    df['近5走_中央値スピード指数'] = df[['前走_スピード指数','2走前_スピード指数','3走前_スピード指数','4走前_スピード指数','5走前_スピード指数']].median(axis=1)
    df['近5走_最高スピード指数']   = df[['前走_スピード指数','2走前_スピード指数','3走前_スピード指数','4走前_スピード指数','5走前_スピード指数']].max(axis=1)
    df['上昇度_スピード指数'] = df['前走_スピード指数']-df['近5走_中央値スピード指数']

    df['前走_通過']  = df.groupby('馬ID')['通過'].shift(1)
    df['2走前_通過'] = df.groupby('馬ID')['通過'].shift(2)
    def parse_corner(x):
        s=str(x); return s.split('-')[-1] if '-' in s else (s if s.isdigit() else np.nan)
    df['前走_最終コーナー']  = pd.to_numeric(df['前走_通過'].fillna('').astype(str).apply(parse_corner), errors='coerce')
    df['2走前_最終コーナー'] = pd.to_numeric(df['2走前_通過'].fillna('').astype(str).apply(parse_corner), errors='coerce')
    df['脚質カテゴリ'] = df['前走_最終コーナー'].apply(classify_style)
    df['前走逃げフラグ']  = (df['前走_最終コーナー']<=2).astype(int)
    df['前走先行フラグ']  = ((df['前走_最終コーナー']>2)&(df['前走_最終コーナー']<=5)).astype(int)
    df['同レース逃げ馬頭数'] = df.groupby('レースID')['前走逃げフラグ'].transform('sum')
    df['同レース先行馬頭数'] = df.groupby('レースID')['前走先行フラグ'].transform('sum')
    df['コース適性_着順パーセント'] = df.groupby(['馬ID','競馬場','芝/ダート'])['着順パーセント'].transform(lambda x: x.shift(1).expanding().mean()).fillna(0.5)
    df['位置取りショック'] = df['前走_最終コーナー']-df['2走前_最終コーナー']
    df['前走_日付'] = df.groupby('馬ID')['日付'].shift(1)
    df['休養日数'] = (df['日付']-df['前走_日付']).dt.days

    # ★修正BUG1: CAT_FEATURES の新列を補完
    for col in ['回り','コース地形']:
        if col not in df.columns or df[col].isna().all():
            df[col] = df['競馬場'].map(VENUE_MAWARI if col=='回り' else VENUE_CHIKEI).fillna('不明')
    if '騎手_競馬場' not in df.columns:
        df['騎手_競馬場'] = df['騎手ID'].astype(str)+'_'+df['競馬場'].astype(str)
    if '騎手_距離' not in df.columns:
        df['騎手_距離'] = df['騎手ID'].astype(str)+'_'+df['距離'].astype(str)
    for col in ['天候','前走芝ダート']:
        if col in df.columns: df[col] = df[col].fillna('不明').astype(str)
        else: df[col] = '不明'

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
          'キャリア数','上り順位率','前走_上り順位率']
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
    model_win = lgb.LGBMRanker(n_estimators=400, learning_rate=0.02, num_leaves=48, max_bin=255,
                                cat_smooth=10, random_state=123, importance_type='gain',
                                colsample_bytree=0.7, subsample=0.8)
    model_win.fit(train_df[features], train_df['win_label'], group=train_groups,
                  categorical_feature=[f for f in cat_features if f in features],
                  eval_set=[(test_df[features], test_df['win_label'])], eval_group=[test_groups])

    # ── アンサンブルスコア ────────────────────────────────────────
    score_a = model.predict(test_df[features])
    score_b = model_win.predict(test_df[features])
    def _norm_scores(s):
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn + 1e-9)
    _sa_norm = _norm_scores(score_a)
    _sb_norm = _norm_scores(score_b)

    # ⚠️ 重み自動最適化は無効化（理由: コース統計がfull dataで計算されておりリーク込みの回収率になるため）
    # 実績（3/21: 本命単勝86%, 3/22: 穴馬EV単勝180%）から1着モデル寄りが有効と判断し
    # 複勝0.4 / 1着0.6 の固定値を使用する。Optuna導入時に正しく再最適化予定。
    best_weight = 0.4  # 複勝モデルの重み（0.4=複勝寄り, 0.6=1着モデル寄り）
    logger.info(f'アンサンブル重み: 複勝={best_weight:.1f} / 1着={1-best_weight:.1f} (固定値・リーク修正後に再最適化予定)')
    test_df['予測スコア'] = _sa_norm * best_weight + _sb_norm * (1 - best_weight)
    test_df['exp_score'] = np.exp(test_df['予測スコア']-test_df.groupby('レースID')['予測スコア'].transform('max'))
    test_df['AI勝率'] = test_df['exp_score']/test_df.groupby('レースID')['exp_score'].transform('sum')
    top_preds = test_df.sort_values(['レースID','AI勝率'],ascending=[True,False]).groupby('レースID').head(1)
    win_hits  = top_preds[pd.to_numeric(top_preds['着順'],errors='coerce')==1]
    invest_amount = len(top_preds)*100
    win_return = (pd.to_numeric(win_hits['単勝'],errors='coerce')*100).sum()
    recent_return_rate = (win_return/invest_amount*100) if invest_amount>0 else 0

    # ── AUC計算（1着予測 & 複勝予測）──────────────────────────────────
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

    bundle = (model, model_win, features, cat_features, num_features, cat_categories_dict,
              latest_horse_data, horse_course_dict, ped_dict,
              known_jockeys, known_trainers, te_dicts, global_mean, recent_return_rate, best_weight,
              auc_win, auc_place)

    # ── HF Hubにアップロード ──────────────────────────────────
    _save_model_to_hub(bundle)

    return bundle
