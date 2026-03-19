import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import requests
from bs4 import BeautifulSoup
import re
import os
import datetime
import pytz
import traceback
import time
import json
import random

from features_engine import NUM_FEATURES, CAT_FEATURES, TE_COLS, classify_style

st.set_page_config(page_title="keiba-ebye 予測ダッシュボード", page_icon="🐴", layout="wide")
st.title("🐴 keiba-ebye 予測ダッシュボード")
st.markdown("えーびーあい (ebi × AI × Eye) が、極限まで高められた精度でお宝馬を暴き出すかも。。。。")

VENUE_MAWARI = {'札幌':'右回り','函館':'右回り','福島':'右回り','新潟':'左回り','東京':'左回り','中山':'右回り','中京':'左回り','京都':'右回り','阪神':'右回り','小倉':'右回り'}
VENUE_CHIKEI = {'札幌':'平坦','函館':'平坦','福島':'急坂','新潟':'平坦','東京':'急坂','中山':'急坂','中京':'急坂','京都':'緩坂','阪神':'急坂','小倉':'平坦'}

def resolve_name(short_name, known_names):
    if pd.isna(short_name) or short_name == '不明': return '不明'
    clean_name = re.sub(r'[☆▲△◇★\n\s　]', '', str(short_name))
    clean_name = re.sub(r'\[[東西地外]\]', '', clean_name)
    clean_name = re.sub(r'(栗東|美浦)', '', clean_name)
    if not clean_name: return '不明'
    aliases = {"鮫島駿":"鮫島克駿","鮫島良":"鮫島良太","吉田隼":"吉田隼人","武幸":"武幸四郎","菅原明":"菅原明良"}
    if clean_name in aliases: clean_name = aliases[clean_name]
    normalized_dict = {}
    for kn in known_names:
        if pd.isna(kn): continue
        norm_kn = re.sub(r'[☆▲△◇★\n\s　]','',str(kn)); norm_kn = re.sub(r'\[[東西地外]\]','',norm_kn); norm_kn = re.sub(r'(栗東|美浦)','',norm_kn)
        if norm_kn not in normalized_dict: normalized_dict[norm_kn] = []
        normalized_dict[norm_kn].append(kn)
    if clean_name in normalized_dict: return sorted(normalized_dict[clean_name], key=len)[0]
    fwd = [n for nk,orig in normalized_dict.items() if nk.startswith(clean_name) for n in orig]
    if fwd: return sorted(fwd, key=len)[0]
    par = [n for nk,orig in normalized_dict.items() if clean_name in nk for n in orig]
    if par: return sorted(par, key=len)[0]
    return clean_name

_UA_LIST = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
]
def get_headers(): return {"User-Agent": random.choice(_UA_LIST)}

_HF_TOKEN   = os.environ.get("HF_TOKEN", "")
_HF_REPO_ID = os.environ.get("HF_REPO_ID", "")   # 例: "username/keiba-ebye"
_MODEL_FILE = "keiba_model.pkl"                    # Hub上のファイル名
_META_FILE  = "keiba_model_meta.json"             # 学習日時などのメタデータ

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
        except Exception:
            pass  # メタデータなし = 初回 → そのままロードを試みる

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
    df = df.sort_values(['馬ID','日付']).reset_index(drop=True)

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
          '前走_通過','2走前_通過','前走_最終コーナー','2走前_最終コーナー']
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
    model = lgb.LGBMRanker(n_estimators=500, learning_rate=0.01, num_leaves=63, max_bin=255,
                            cat_smooth=10, random_state=42, importance_type='gain',
                            colsample_bytree=0.7, subsample=0.8)
    model.fit(train_df[features], train_df['馬券内'], group=train_groups,
              categorical_feature=[f for f in cat_features if f in features],
              eval_set=[(test_df[features], test_df['馬券内'])], eval_group=[test_groups])

    test_df['予測スコア'] = model.predict(test_df[features])
    test_df['exp_score'] = np.exp(test_df['予測スコア']-test_df.groupby('レースID')['予測スコア'].transform('max'))
    test_df['AI勝率'] = test_df['exp_score']/test_df.groupby('レースID')['exp_score'].transform('sum')
    top_preds = test_df.sort_values(['レースID','AI勝率'],ascending=[True,False]).groupby('レースID').head(1)
    win_hits  = top_preds[pd.to_numeric(top_preds['着順'],errors='coerce')==1]
    invest_amount = len(top_preds)*100
    win_return = (pd.to_numeric(win_hits['単勝'],errors='coerce')*100).sum()
    recent_return_rate = (win_return/invest_amount*100) if invest_amount>0 else 0

    try:
        ped_df = pd.read_csv('pedigree_master_all.csv', dtype=str)
        ped_df['馬ID'] = ped_df['馬ID'].astype(str).str.zfill(10)
        ped_dict = ped_df.set_index('馬ID')[['父','父系','母','母系','母父','母父系']].to_dict('index')
    except: ped_dict = {}

    bundle = (model, features, cat_features, num_features, cat_categories_dict,
              latest_horse_data, horse_course_dict, ped_dict,
              known_jockeys, known_trainers, te_dicts, global_mean, recent_return_rate)

    # ── HF Hubにアップロード ──────────────────────────────────
    _save_model_to_hub(bundle)

    return bundle

_hub_available = bool(_HF_TOKEN and _HF_REPO_ID)
_hub_label = "HF Hub" if _hub_available else "ローカル学習"

with st.spinner(f'AIエンジン起動中... ({_hub_label}からロード試行)'):
    (model, features, cat_features, num_features, cat_categories_dict,
     latest_horse_data, horse_course_dict, ped_dict,
     known_jockeys, known_trainers, te_dicts, global_mean, recent_return_rate) = prepare_model_and_data()

# ==========================================
# 2. スクレイピング ＆ アナリティクス関数群 (省略せず記載)
# ==========================================
def get_todays_races(date_str=None):
    races = []
    tokyo_tz = pytz.timezone('Asia/Tokyo')
    now = datetime.datetime.now(tokyo_tz)
    target_date_str = date_str if date_str else now.strftime('%Y%m%d')
    added_ids = set()
    
    urls_to_try = [
        f'https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={target_date_str}',
        f'https://race.netkeiba.com/top/race_list.html?kaisai_date={target_date_str}'
    ]
    for url in urls_to_try:
        try:
            # 🌟 文字化け対策: encodingの強制指定を削除
            res = requests.get(url, headers=get_headers(), timeout=10)
            # 🌟 res.textではなく、res.content(生データ)を渡してAIに自動判定させる！
            soup = BeautifulSoup(res.content, 'html.parser')
            
            for a_tag in soup.find_all('a', href=re.compile(r'race_id=(\d{12})')):
                r_id = re.search(r'race_id=(\d{12})', a_tag.get('href')).group(1)
                if not (1 <= int(r_id[4:6]) <= 10): continue
                if r_id in added_ids: continue
                added_ids.add(r_id)
                
                place_dict = {'01':'札幌','02':'函館','03':'福島','04':'新潟','05':'東京','06':'中山','07':'中京','08':'京都','09':'阪神','10':'小倉'}
                place = place_dict.get(r_id[4:6], '不明')
                r_num = int(r_id[10:12])
                
                parent = a_tag.find_parent('li') or a_tag.find_parent('dl') or a_tag.find_parent('div')
                time_span = parent.find(class_=re.compile(r'time', re.I)) if parent else None
                title_span = parent.find(class_=re.compile(r'Title', re.I)) if parent else None
                
                if time_span and title_span and time_span.text.strip():
                    try: 
                        time_str = re.search(r'\d{2}:\d{2}', time_span.text).group(0)
                        start_dt = tokyo_tz.localize(datetime.datetime.strptime(f"{target_date_str} {time_str}", "%Y%m%d %H:%M"))
                    except: start_dt = tokyo_tz.localize(datetime.datetime.strptime(f"{target_date_str} 12:00", "%Y%m%d %H:%M"))
                    title = title_span.text.strip()
                else:
                    start_dt = tokyo_tz.localize(datetime.datetime.strptime(f"{target_date_str} 12:00", "%Y%m%d %H:%M"))
                    title = f"{place} {r_num}R"
                races.append({'id': r_id, 'place': place, 'num': r_num, 'title': title, 'time': start_dt, 'sort_key': f"{r_id[4:6]}{r_num:02d}"})
        except: pass
        if races: break

    if not races:
        url = f'https://db.netkeiba.com/race/list/{target_date_str}/'
        try:
            # 🌟 ここも同様に文字化け対策
            res = requests.get(url, headers=get_headers(), timeout=10)
            soup = BeautifulSoup(res.content, 'html.parser')
            ids = set(re.findall(r'/race/(\d{12})', res.text))
            for r_id in ids:
                if not (1 <= int(r_id[4:6]) <= 10): continue
                place = {'01':'札幌','02':'函館','03':'福島','04':'新潟','05':'東京','06':'中山','07':'中京','08':'京都','09':'阪神','10':'小倉'}.get(r_id[4:6], '不明')
                r_num = int(r_id[10:12])
                dummy_time = tokyo_tz.localize(datetime.datetime.strptime(f"{target_date_str} 12:00", "%Y%m%d %H:%M"))
                races.append({'id': r_id, 'place': place, 'num': r_num, 'title': f"{place} {r_num}R", 'time': dummy_time, 'sort_key': f"{r_id[4:6]}{r_num:02d}"})
        except: pass
    return sorted(races, key=lambda x: x['sort_key'])

def get_weekend_dates():
    tokyo_tz = pytz.timezone('Asia/Tokyo')
    now = datetime.datetime.now(tokyo_tz)
    saturday = now + datetime.timedelta(days=(5 - now.weekday()) % 7)
    sunday = saturday + datetime.timedelta(days=1)
    return saturday.strftime('%Y%m%d'), sunday.strftime('%Y%m%d')

def get_payouts(race_id):
    tansho_dict, fukusho_dict = {}, {}
    urls = [f"https://race.netkeiba.com/race/result.html?race_id={race_id}", f"https://db.netkeiba.com/race/{race_id}/"]
    for url in urls:
        try:
            res = requests.get(url, headers=get_headers(), timeout=10); res.encoding = 'euc-jp'
            soup = BeautifulSoup(res.text, 'html.parser')
            tables = soup.find_all('table', class_=re.compile(r'Pay_Table_01|pay_table_01'))
            if not tables: tables = soup.find_all('table', summary='払い戻し')
            for tbl in tables:
                for tr in tbl.find_all('tr'):
                    th = tr.find('th')
                    if not th: continue
                    if th.text.strip() in ['単勝', '複勝']:
                        res_td = tr.find('td', class_=re.compile(r'Result'))
                        if not res_td: res_td = tr.find_all('td')[0] if len(tr.find_all('td')) > 0 else None
                        pay_td = tr.find('td', class_=re.compile(r'Payout'))
                        if not pay_td: pay_td = tr.find_all('td')[1] if len(tr.find_all('td')) > 1 else None
                        if res_td and pay_td:
                            umbans = [re.sub(r'\D', '', s) for s in res_td.stripped_strings if re.sub(r'\D', '', s)]
                            pays = [re.sub(r'\D', '', s) for s in pay_td.stripped_strings if re.sub(r'\D', '', s)]
                            for u, p in zip(umbans, pays):
                                if u and p:
                                    if th.text.strip() == '単勝': tansho_dict[int(u)] = int(p)
                                    else: fukusho_dict[int(u)] = int(p)
            if tansho_dict: break
        except: pass
    return tansho_dict, fukusho_dict

def get_all_payouts(race_id):
    payouts = {'tansho': {}, 'fukusho': {}, 'umaren': {}, 'wide': {}}
    headers = {"User-Agent": "Mozilla/5.0"}
    
    # 🌟 どんなHTMLタグも確実に「改行」に粉砕してリスト化する最強の関数
    def parse_td(td_element):
        if not td_element: return []
        html = str(td_element).replace('<br>', '\n').replace('<br/>', '\n').replace('<br />', '\n')
        html = re.sub(r'</?(div|li|ul|p|span|strong)[^>]*>', '\n', html, flags=re.I)
        lines = BeautifulSoup(html, 'html.parser').get_text().split('\n')
        return [line.strip() for line in lines if line.strip()]

    # 1. netkeiba (出馬表ページ ＆ 過去データベース 両対応)
    for url in [f"https://race.netkeiba.com/race/result.html?race_id={race_id}", f"https://db.netkeiba.com/race/{race_id}/"]:
        try:
            res = requests.get(url, headers=get_headers(), timeout=10)
            html_bytes = res.content
            html_text = html_bytes.decode('euc-jp', errors='ignore') # netkeibaはEUC-JP固定でOK
            soup = BeautifulSoup(html_text, 'html.parser')
            
            tables = soup.find_all('table', class_=re.compile(r'Pay_Table_01|pay_table_01', re.I))
            if not tables: tables = soup.find_all('table', summary='払い戻し')
            
            for tbl in tables:
                current_kind = None # 🌟 前の行の「券種」を記憶する変数
                
                for tr in tbl.find_all('tr'):
                    th = tr.find('th')
                    if th:
                        # 見出しがあれば券種を上書き
                        th_text = re.sub(r'\s+', '', th.text)
                        th_class = " ".join(th.get('class', [])).lower()
                        if 'tansho' in th_class or '単勝' in th_text: current_kind = '単勝'
                        elif 'fukusho' in th_class or '複勝' in th_text: current_kind = '複勝'
                        elif 'umaren' in th_class or '馬連' in th_text: current_kind = '馬連'
                        elif 'wide' in th_class or 'ワイド' in th_text: current_kind = 'ワイド'
                        else: current_kind = None
                    
                    if not current_kind: continue # 単勝・複勝・馬連・ワイド以外は無視
                    
                    tds = tr.find_all('td')
                    if not tds: continue
                    
                    res_td = tr.find('td', class_=re.compile(r'Result', re.I))
                    pay_td = tr.find('td', class_=re.compile(r'Payout', re.I))
                    
                    # 🌟 クラス名が無い「過去データベース版」への対応
                    if not res_td and len(tds) >= 2:
                        res_td = tds[0]
                        pay_td = tds[1]
                        
                    if not res_td or not pay_td: continue
                    
                    r_lines = parse_td(res_td)
                    p_lines = parse_td(pay_td)
                    
                    r_clean, p_clean = [], []
                    for r in r_lines:
                        nums = [int(x) for x in re.findall(r'\d+', r)]
                        if nums: r_clean.append(nums)
                        
                    for p in p_lines:
                        if '人気' in p: continue
                        val = re.sub(r'\D', '', p.replace(',', ''))
                        if val: p_clean.append(int(val))
                        
                    if not r_clean or not p_clean: continue
                    
                    if len(p_clean) > len(r_clean) and len(p_clean) % len(r_clean) == 0:
                        step = len(p_clean) // len(r_clean)
                        p_clean = p_clean[0::step]
                        
                    for nums, pay in zip(r_clean, p_clean):
                        if current_kind == '単勝' and len(nums) >= 1: payouts['tansho'][nums[0]] = pay
                        elif current_kind == '複勝' and len(nums) >= 1: payouts['fukusho'][nums[0]] = pay
                        elif current_kind == '馬連' and len(nums) >= 2: payouts['umaren'][tuple(sorted(nums[:2]))] = pay
                        elif current_kind == 'ワイド' and len(nums) >= 2: payouts['wide'][tuple(sorted(nums[:2]))] = pay

            if payouts['tansho'] and payouts['wide']: return payouts
        except: pass

    # 2. Yahoo!競馬 (裏ルート保険版)
    try:
        yahoo_id = str(race_id)[2:]
        url_yh = f"https://sports.yahoo.co.jp/keiba/race/result/{yahoo_id}/"
        res_y = requests.get(url_yh, headers=get_headers(), timeout=10)
        soup_y = BeautifulSoup(res_y.text, 'html.parser')
        
        current_kind = None # 🌟 ここでも券種を記憶
        for tr in soup_y.find_all('tr'):
            th = tr.find('th')
            if th:
                th_text = th.text.strip()
                if th_text in ['単勝', '複勝', '馬連', 'ワイド']: current_kind = th_text
                else: current_kind = None
            
            if not current_kind: continue
            
            tds = tr.find_all('td')
            if len(tds) < 2: continue
            
            r_lines = parse_td(tds[0])
            p_lines = parse_td(tds[1])
            
            r_clean, p_clean = [], []
            for r in r_lines:
                nums = [int(x) for x in re.findall(r'\d+', r)]
                if nums: r_clean.append(nums)
                
            for p in p_lines:
                if '人気' in p: continue
                val = re.sub(r'\D', '', p.replace(',', ''))
                if val: p_clean.append(int(val))
                
            if not r_clean or not p_clean: continue
            
            if len(p_clean) > len(r_clean) and len(p_clean) % len(r_clean) == 0:
                step = len(p_clean) // len(r_clean)
                p_clean = p_clean[0::step]
                
            for nums, pay in zip(r_clean, p_clean):
                if current_kind == '単勝' and len(nums) >= 1: payouts['tansho'][nums[0]] = pay
                elif current_kind == '複勝' and len(nums) >= 1: payouts['fukusho'][nums[0]] = pay
                elif current_kind == '馬連' and len(nums) >= 2: payouts['umaren'][tuple(sorted(nums[:2]))] = pay
                elif current_kind == 'ワイド' and len(nums) >= 2: payouts['wide'][tuple(sorted(nums[:2]))] = pay
    except: pass

    return payouts

def get_odds_from_soup(s_soup):
    o_dict = {}
    tgt_table = s_soup.select_one('.Shutuba_Table') or s_soup.select_one('.RaceTable01') or s_soup.select_one('.race_table_01') or s_soup.select_one('#All_Result_Table')
    if not tgt_table: return o_dict
    u_idx, o_idx = -1, -1
    for i, th in enumerate(tgt_table.find_all('th')):
        c_txt = re.sub(r'\s+', '', th.text)
        if '馬番' in c_txt: u_idx = i
        if '単勝' in c_txt or 'オッズ' in c_txt or '予想' in c_txt or '人気' in c_txt: o_idx = i
        
    try:
        for tr in tgt_table.find_all('tr')[1:]:
            tds = tr.find_all('td')
            umaban = -1
            if u_idx != -1 and len(tds) > u_idx:
                u_m = re.search(r'\d+', tds[u_idx].text)
                if u_m: umaban = int(u_m.group(0))
            if umaban == -1: continue
            
            odds_val = 0.0
            if o_idx != -1 and len(tds) > o_idx:
                o_m = re.search(r'\d{1,4}\.\d+', tds[o_idx].text)
                if o_m: odds_val = float(o_m.group(0))
                        
            # 斤量誤爆を防ぐため、怪しいクラスから強引に数字を拾う処理を削除
            if odds_val > 0.0: o_dict[umaban] = odds_val
    except: pass
    return o_dict

def generate_pdf_report(results_list, ev_threshold=1.5):
    """
    予想レポートをHTML形式で生成してbytesを返す。
    ブラウザで開いてCtrl+P(Cmd+P)で印刷・PDF保存できる。
    reportlabのフォント問題を回避するためHTML方式に変更。
    """
    try:
        ev_summary = []
        races_html = ""
        for r in results_list:
            df = r['df']
            rows_html = ""
            for rank, row in df.iterrows():
                try:
                    ev_val = float(row.get('期待値', 0) or 0)
                except: ev_val = 0.0
                bg = ' style="background:#fff0f0"' if ev_val >= ev_threshold else (' style="background:#fffde7"' if rank == 0 else '')
                ev_color = ' color:#cc0000;font-weight:bold' if ev_val >= ev_threshold else ''
                rows_html += f"""<tr{bg}>
                  <td>{row.get('印','')}</td>
                  <td>{int(float(row.get('馬番',0)))}</td>
                  <td>{row.get('馬名','')}</td>
                  <td>{row.get('脚質カテゴリ','')}</td>
                  <td>{float(row.get('単勝オッズ',0)):.1f}</td>
                  <td>{float(row.get('勝率(AI予測)',0))*100:.1f}%</td>
                  <td>{float(row.get('複勝率(AI予測)',0))*100:.1f}%</td>
                  <td style="{ev_color}">{ev_val:.2f}</td>
                </tr>"""
                if rank < 5 and ev_val >= ev_threshold:
                    ev_summary.append({'レース': f"{r['place']}{r['num']}R", '印': row.get('印',''), '馬名': row.get('馬名',''), 'EV': ev_val})
            races_html += f"""
            <div class="race-block">
              <h3>■ {r['place']} {r['num']}R ({r['track']}{r['dist']}m)</h3>
              <p class="conf">{r.get('confidence','')[:60]}</p>
              <p class="pace">展開: {r.get('pace','')[:80]}</p>
              <table>
                <thead><tr><th>印</th><th>馬番</th><th>馬名</th><th>脚質</th><th>オッズ</th><th>勝率</th><th>複勝率</th><th>EV</th></tr></thead>
                <tbody>{rows_html}</tbody>
              </table>
              <p class="reco">推奨: {r.get('reco','')[:100]}</p>
            </div>"""
        
        ev_html = ""
        if ev_summary:
            ev_rows = "".join(f"<tr><td>{r['印']}</td><td>{r['レース']}</td><td>{r['馬名']}</td><td style='color:#cc0000;font-weight:bold'>{r['EV']:.2f}</td></tr>"
                             for r in sorted(ev_summary, key=lambda x: x['EV'], reverse=True))
            ev_html = f"""<div class="ev-summary">
              <h3>💰 本日の注目馬 (EV{ev_threshold}以上・上位5頭内)</h3>
              <table><thead><tr><th>印</th><th>レース</th><th>馬名</th><th>EV</th></tr></thead>
              <tbody>{ev_rows}</tbody></table></div>"""
        
        date_str = results_list[0]['date'] if results_list else ''
        html = f"""<!DOCTYPE html>
<html lang="ja"><head><meta charset="UTF-8">
<title>keiba-ebye 予想レポート {date_str}</title>
<style>
  body {{ font-family: 'Hiragino Kaku Gothic ProN','Noto Sans JP','Meiryo',sans-serif; font-size:11px; margin:10px; color:#222; }}
  h1 {{ font-size:16px; border-bottom:2px solid #2c3e7a; padding-bottom:4px; color:#2c3e7a; }}
  h3 {{ font-size:12px; margin:12px 0 4px; color:#2c3e7a; border-left:3px solid #2c3e7a; padding-left:6px; }}
  .race-block {{ margin-bottom:16px; page-break-inside:avoid; }}
  .conf {{ font-size:10px; color:#555; margin:2px 0; }}
  .pace {{ font-size:10px; color:#444; margin:2px 0; }}
  .reco {{ font-size:10px; background:#f0f8ff; padding:4px 6px; border-radius:3px; margin-top:4px; }}
  table {{ border-collapse:collapse; width:100%; margin:4px 0; font-size:10px; }}
  th {{ background:#2c3e7a; color:white; padding:3px 5px; text-align:center; }}
  td {{ border:1px solid #ddd; padding:2px 4px; text-align:center; }}
  tr:nth-child(even) {{ background:#f8f8f8; }}
  .ev-summary {{ margin-top:16px; padding:8px; background:#fff8f0; border:1px solid #ffd0a0; border-radius:4px; }}
  @media print {{ body {{ margin:5mm; }} .race-block {{ page-break-inside:avoid; }} }}
</style></head>
<body>
<h1>🐴 keiba-ebye AI予想レポート — {date_str}</h1>
{races_html}
{ev_html}
<p style="font-size:9px;color:#888;margin-top:20px">Generated by keiba-ebye / ブラウザでCtrl+P→PDFで保存できます</p>
</body></html>"""
        return html.encode('utf-8')
    except Exception as e:
        return None



def generate_txt_report(results_list, ev_threshold=1.5):
    txt = "=== 🏇 keiba-ebye 予想レポート ===\n\n"
    ev_summary_rows = []

    for r in results_list:
        txt += "="*50 + "\n"
        txt += f"■ {r['date']} | {r['place']} {r['num']}R ({r['track']}{r['dist']}m) ■\n"
        txt += f"🐎 展開予想: {r['pace']}\n"
        txt += f"🔮 AI自信度: {r['confidence']}\n"
        txt += "-"*50 + "\n"
        for rank, row in r['df'].iterrows():
            ev_str = f" [EV:{row['期待値']:.2f}]" if row['期待値'] >= ev_threshold else ""
            kyuyo  = row.get('休養日数', np.nan)
            kyuyo_str = f" 休養{int(kyuyo)}日" if not pd.isna(kyuyo) and kyuyo > 0 else ""
            jflag  = " [乗替]" if row.get('乗り替わりフラグ', 0) == 1 else ""
            sflag  = " [馬場変]" if row.get('馬場替わりフラグ', 0) == 1 else ""
            dflag  = " [距離変]" if row.get('距離変更フラグ', 0) == 1 else ""
            txt += (f" {row['印']} {rank+1}位: [{row['枠番']}枠{row['馬番']}番] {row['馬名']}"
                    f" ({row['脚質カテゴリ']}) "
                    f"勝率{row['勝率(AI予測)']*100:.1f}% / 複勝率{row['複勝率(AI予測)']*100:.1f}%"
                    f" オッズ{row['単勝オッズ']}倍{ev_str}{kyuyo_str}{jflag}{sflag}{dflag}\n")
            # EV横断一覧用
            if rank < 5 and row['期待値'] >= ev_threshold:
                ev_summary_rows.append({
                    'レース': f"{r['place']}{r['num']}R",
                    '印': row['印'],
                    '馬番': row['馬番'],
                    '馬名': row['馬名'],
                    'EV': row['期待値'],
                    '勝率': f"{row['勝率(AI予測)']*100:.1f}%",
                    'オッズ': row['単勝オッズ'],
                })
        txt += "-"*50 + "\n"
        if r.get('topics'):
            txt += "📝 要注目トピック馬:\n"
            for t in r['topics']: txt += f"  {t}\n"
            txt += "-"*50 + "\n"
        txt += f"🤖 推奨買い目:\n  {r['reco']}\n"
        txt += "="*50 + "\n\n"

    # 横断EV一覧を末尾に追加
    if ev_summary_rows:
        txt += "\n" + "="*50 + "\n"
        txt += f"💰 本日の期待値注目馬一覧 (印上位5頭内 EV>={ev_threshold}) \n"
        txt += "="*50 + "\n"
        ev_summary_rows.sort(key=lambda x: x['EV'], reverse=True)
        for row in ev_summary_rows:
            txt += f" {row['印']} {row['レース']} {row['馬番']}番 {row['馬名']}  EV:{row['EV']:.2f} / 勝率:{row['勝率']} / オッズ:{row['オッズ']}倍\n"
        txt += "="*50 + "\n"
    return txt


# ==========================================
# 3. 本格AI予測関数 (★BUG修正版)

# 騎手名の既知の表記ゆれ辞書（出馬表の短縮表記 → 正式名）
_JOCKEY_ABBR = {
    # 出馬表での短縮表記 → 正式名
    # ※ 危険な短縮（別騎手名の一部にもマッチする可能性があるもの）は入れない
    '角田和':   '角田大和',
    '石神深道':  '石神深道',   # フルネームのまま
    '石神深一':  '石神深一',
    '石川裕紀':  '石川裕紀人',  # 石川倭との混同防止（裕紀 → 裕紀人）
    '小林凌':   '小林凌大',
    '菱田裕':   '菱田裕二',
    '富田暁':   '富田暁斗',
    '木幡初':   '木幡初也',
    '木幡巧':   '木幡巧也',
    '水口優':   '水口優也',
    '亀田温':   '亀田温心',
    '団野大':   '団野大成',
    '西村淳':   '西村淳也',
    '大江':     '大江原圭',   # 大江→大江原圭
    '永島まな': '永島まなみ',
    '古川奈':   '古川奈穂',
}

# resolve_name()とは別に、出馬表の騎手名から正式名を推定するマッピング
# fetch_horse_last_race()の結果と比較するために使用
_JOCKEY_NORMALIZE_EXTRA = {
    # 短縮名 → 正式名（4文字以上の部分一致で誤爆しやすいケースを明示）
    '石川裕紀人': '石川裕紀人',
    '石川倭':    '石川倭',
}

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_horse_last_race(horse_id: str) -> dict:
    """
    netkeibaの馬ページから「最後のレース情報」をスクレイプして返す。
    戻り値: {
        '前走日付':   '2025/03/08',
        '前走距離':   1600.0,
        '前走芝ダート': '芝',
        '前走騎手':   '川田将雅',
        '前走着順':   2,
    }
    取得失敗時は空dict{}を返す。
    """
    result = {}
    try:
        url = f"https://db.netkeiba.com/horse/{horse_id}/"
        r = requests.get(url, headers=get_headers(), timeout=8)
        r.encoding = 'euc-jp'
        soup = BeautifulSoup(r.text, 'html.parser')

        # 競走成績テーブルを探す
        table = soup.find('table', class_='race_table_01') or soup.find('table', summary='新着情報')
        if not table:
            return result

        rows = table.find_all('tr')
        # 1行目はheader、2行目が最新レース
        if len(rows) < 2:
            return result

        # ヘッダー列名取得
        headers = [th.text.strip().replace('\n','') for th in rows[0].find_all(['th','td'])]
        def gi(kws):
            for i, h in enumerate(headers):
                if any(k in h for k in kws): return i
            return -1

        date_i  = gi(['日付','年月日'])
        dist_i  = gi(['距離'])
        jock_i  = gi(['騎手'])
        rank_i  = gi(['着順','着'])

        last_row_tds = rows[1].find_all('td')
        if len(last_row_tds) < 4:
            return result

        def g(i):
            return last_row_tds[i].text.strip() if i != -1 and i < len(last_row_tds) else ''

        # 日付
        date_str = g(date_i)
        if re.match(r'\d{4}/\d{1,2}/\d{1,2}', date_str):
            result['前走日付'] = date_str

        # 距離と芝/ダート (例: "芝1600" or "ダ1200")
        dist_text = g(dist_i)
        dist_m = re.search(r'(\d{3,4})', dist_text)
        if dist_m:
            result['前走距離'] = float(dist_m.group(1))
        if dist_text.startswith('芝') or '芝' in dist_text:
            result['前走芝ダート'] = '芝'
        elif dist_text.startswith('ダ') or 'ダ' in dist_text or 'ダート' in dist_text:
            result['前走芝ダート'] = 'ダート'
        elif dist_text.startswith('障') or '障' in dist_text:
            result['前走芝ダート'] = '障害'

        # 騎手
        jock_td = last_row_tds[jock_i] if jock_i != -1 and jock_i < len(last_row_tds) else None
        if jock_td:
            ja = jock_td.find('a')
            jname = ja.text.strip() if ja else jock_td.text.strip()
            result['前走騎手'] = jname

        # 着順
        rank_str = g(rank_i)
        rank_m = re.search(r'^\d+$', rank_str)
        if rank_m:
            result['前走着順'] = int(rank_str)

    except Exception:
        pass
    return result


def _safe_col(df, col, default=np.nan):
    """列が存在しない or スカラーになっている場合でも必ずSeriesを返す安全ラッパー"""
    if col not in df.columns:
        return pd.Series([default] * len(df), index=df.index)
    val = df[col]
    if isinstance(val, pd.Series):
        return val
    return pd.Series([val] * len(df), index=df.index)

# ==========================================
def run_real_prediction(race_id, race_date_str, skip_live_scrape=False):
    """
    skip_live_scrape=True: バックテスト時に使用。
      fetch_horse_last_race()を呼ばない（速度維持＆日付ズレ防止）
    """
    error_log = []
    odds_dict = {}
    html_text = ""

    try:
        odds_api_url = f'https://race.netkeiba.com/api/api_get_jra_odds.html?type=1&action=init&race_id={race_id}'
        api_headers = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36","Referer":f"https://race.netkeiba.com/odds/index.html?type=b1&race_id={race_id}","X-Requested-With":"XMLHttpRequest"}
        r_api = requests.get(odds_api_url, headers=api_headers, timeout=5)
        api_data = json.loads(r_api.text)
        if 'data' in api_data and 'odds' in api_data['data'] and '1' in api_data['data']['odds']:
            # APIレスポンスには馬名も含まれる場合がある → 馬名でマッチング
            odds_raw = api_data['data']['odds']['1']
            # 馬名がAPIに含まれていれば馬名→オッズのdictも作る
            name_odds_dict = {}
            if 'horses' in api_data.get('data', {}):
                for h in api_data['data']['horses']:
                    hname = h.get('name', '').strip()
                    hnum  = h.get('num', '')
                    if hname and hnum and str(hnum) in odds_raw:
                        name_odds_dict[hname] = float(odds_raw[str(hnum)][0])
                        odds_dict[int(hnum)] = float(odds_raw[str(hnum)][0])
            if not name_odds_dict:
                # 旧形式: キーが馬番か人気順か不明 → とりあえず馬番として格納
                for uma_num, odds_list in odds_raw.items():
                    if str(uma_num).isdigit():
                        odds_dict[int(uma_num)] = float(odds_list[0])
    except Exception as e: error_log.append(f"netkeiba APIオッズ取得失敗: {e}")

    if not odds_dict:
        try:
            r_yahoo = requests.get(f"https://sports.yahoo.co.jp/keiba/race/odds/tfw/{str(race_id)[2:]}/", headers=get_headers(), timeout=5)
            soup_y = BeautifulSoup(r_yahoo.text, 'html.parser')
            for tr in soup_y.find_all('tr'):
                tds = tr.find_all('td')
                if len(tds) >= 4:
                    u_m = re.search(r'^\s*(\d+)\s*$', tds[1].text)
                    odds_span = tr.find('span', class_='fB')
                    o_m = re.search(r'\d{1,4}\.\d+', odds_span.text) if odds_span else None
                    if u_m and o_m: odds_dict[int(u_m.group(1))] = float(o_m.group(0))
        except Exception as e: error_log.append(f"Yahoo競馬オッズ取得失敗: {e}")

    for fetch_url in [f'https://race.netkeiba.com/race/shutuba.html?race_id={race_id}',f'https://race.netkeiba.com/race/result.html?race_id={race_id}',f'https://db.netkeiba.com/race/{race_id}/']:
        try:
            r = requests.get(fetch_url, headers=get_headers(), timeout=10); r.encoding = 'euc-jp'
            soup = BeautifulSoup(r.text, 'html.parser')
            if soup.select_one('.Shutuba_Table') or soup.select_one('.RaceTable01') or soup.select_one('.race_table_01') or soup.select_one('#All_Result_Table'):
                html_text = r.text; break
        except: pass

    if not html_text: return None,None,None,None,None,None,None,None,["❌ 出馬表が取得できませんでした。"]
    soup = BeautifulSoup(html_text, 'html.parser')
    race_data_box = soup.find('div', class_='RaceData01') or soup.find('dl', class_='racedata')
    if not race_data_box: return None,None,None,None,None,None,None,None,["❌ レース条件が見つかりません。"]

    race_text = race_data_box.text.replace('\n','')
    baba_match = re.search(r'馬場:([良稍重不良]+)', race_text)
    todays_baba = baba_match.group(1) if baba_match else '良'
    tdm = re.search(r'(芝|ダ|障|障害).*?(\d+)m', race_text)
    track_type = "芝" if tdm and tdm.group(1)=="芝" else "ダート" if tdm and "ダ" in tdm.group(1) else "障害"
    distance = float(tdm.group(2)) if tdm else 1600.0
    place = {'01':'札幌','02':'函館','03':'福島','04':'新潟','05':'東京','06':'中山','07':'中京','08':'京都','09':'阪神','10':'小倉'}.get(str(race_id)[4:6], '東京')
    weather_m = re.search(r'天候:([晴曇雨小雪]+)', race_text)
    todays_tenki = weather_m.group(1) if weather_m else '晴'

    table = soup.select_one('.Shutuba_Table') or soup.select_one('.RaceTable01') or soup.select_one('.race_table_01') or soup.select_one('#All_Result_Table')
    if not table: return None,None,None,None,None,None,None,None,["❌ 出走馬の一覧表が見つかりません。"]
    ths = table.find_all('th')
    headers_text = [th.text.strip().replace('\n','') for th in ths]
    def get_idx(kws):
        for i,h in enumerate(headers_text):
            for kw in kws:
                if kw in h: return i
        return -1

    waku_idx=get_idx(['枠']); uma_idx=get_idx(['馬番']); kinryo_idx=get_idx(['斤量'])
    weight_idx=get_idx(['馬体重']); odds_idx=get_idx(['単勝','オッズ','予想','人気'])
    sex_age_idx=get_idx(['性齢']); jockey_idx=get_idx(['騎手']); trainer_idx=get_idx(['調教師','厩舎'])

    # 枠番確定か事前チェック（最初の3行を確認）
    pre_waku_confirmed = False
    for tr in table.find_all('tr')[1:4]:
        tds = tr.find_all('td')
        if len(tds) < 5: continue
        if waku_idx != -1 and len(tds) > waku_idx:
            w_m = re.search(r'\d+', tds[waku_idx].text.strip())
            if w_m and int(w_m.group(0)) > 0:
                pre_waku_confirmed = True; break

    horses = []
    row_pos = 0  # テーブル内の行番号（1始まり）
    for tr in table.find_all('tr')[1:]:
        tds = tr.find_all('td')
        if len(tds) < 5: continue
        try:
            row_text = tr.get_text()
            if re.search(r'取消|除外|中止|取り消し', row_text): continue
            row_class = ' '.join(tr.get('class', []))
            if any(c in row_class.lower() for c in ['cancel','scratch','dns','dnf']): continue

            row_pos += 1
            umaban = int(re.search(r'\d+', tds[uma_idx].text).group(0)) if uma_idx!=-1 and len(tds)>uma_idx and re.search(r'\d+',tds[uma_idx].text) else row_pos
            waku_raw = tds[waku_idx].text.strip() if waku_idx!=-1 and len(tds)>waku_idx else ""
            waku_m = re.search(r'\d+', waku_raw)
            waku = int(waku_m.group(0)) if waku_m and int(waku_m.group(0)) > 0 else 0

            horse_a = tr.find('a', href=re.compile(r'/horse/'))
            if not horse_a: continue
            horse_name = horse_a.text.strip()
            horse_id = re.search(r'\d+', horse_a['href']).group(0)
            jockey_name  = resolve_name(tds[jockey_idx].text.strip() if jockey_idx!=-1 and len(tds)>jockey_idx else "不明", known_jockeys)
            trainer_name = resolve_name(tds[trainer_idx].text.strip() if trainer_idx!=-1 and len(tds)>trainer_idx else "不明", known_trainers)
            km = re.search(r'\d+(\.\d+)?', tds[kinryo_idx].text if kinryo_idx!=-1 and len(tds)>kinryo_idx else "55.0")
            kinryo = float(km.group(0)) if km else 55.0
            wm = re.search(r'^(\d{3})', (tds[weight_idx].text if weight_idx!=-1 and len(tds)>weight_idx else "").strip())
            weight_val = float(wm.group(1)) if wm else np.nan

            # ── オッズ取得 ─────────────────────────────────────────────
            # 枠番確定後 → APIキー=馬番  → umaban で引く
            # 枠番未確定 → APIキー=テーブル行順 → row_pos で引く
            if pre_waku_confirmed:
                odds_val = odds_dict.get(umaban, 0.0)
            else:
                odds_val = odds_dict.get(row_pos, 0.0)  # 行順マッピング

            # name_odds_dict(馬名キー)があれば補完
            if odds_val == 0.0:
                try: odds_val = name_odds_dict.get(horse_name, 0.0)
                except: pass

            # ページ内オッズ列
            if odds_val == 0.0 and odds_idx != -1 and len(tds) > odds_idx:
                om = re.search(r'\d{1,4}\.\d+', tds[odds_idx].text)
                if om: odds_val = float(om.group(0))

            # クラス属性
            if odds_val == 0.0:
                for td in tds:
                    if any(c in ['Odds','Popular','txt_c'] for c in td.get('class',[])):
                        om = re.search(r'\d{1,4}\.\d+', td.text)
                        if om: odds_val = float(om.group(0)); break

            if odds_val == 0.0: odds_val = 10.0

            sex_age = tds[sex_age_idx].text.strip() if sex_age_idx!=-1 and len(tds)>sex_age_idx else "牡3"
            horses.append({'枠番':waku,'馬番':umaban,'馬名':horse_name,'馬ID':horse_id,'性齢':sex_age,'斤量':kinryo,'騎手':jockey_name,'調教師':trainer_name,'距離':distance,'競馬場':place,'芝/ダート':track_type,'馬場':todays_baba,'天候':todays_tenki,'馬体重_num':weight_val,'単勝オッズ':odds_val})
        except: pass

    if not pre_waku_confirmed and horses:
        error_log.append("⚠️ 枠順未確定のため枠番=0・オッズは暫定値です。枠順確定後に再実行してください。")

    if not horses: return None,None,None,None,None,None,None,None,["❌ 出走馬データの読み取りに失敗しました。"]

    try:
        df_test = pd.DataFrame(horses)
        df_test['出走頭数'] = len(df_test)
        df_test = pd.merge(df_test, latest_horse_data, on='馬ID', how='left')

        for col in ['父','父系','母','母系','母父','母父系']:
            if col not in df_test.columns: df_test[col] = np.nan
        for i, row in df_test.iterrows():
            hid = row['馬ID']
            if pd.isna(row.get('父')) or row.get('父')=='不明':
                ped = ped_dict.get(hid, {})
                for col in ['父','父系','母','母系','母父','母父系']:
                    df_test.at[i, col] = ped.get(col, '不明')

        df_test['性別'] = df_test['性齢'].astype(str).str.extract(r'([牡牝セ])')[0]
        df_test['年齢'] = pd.to_numeric(df_test['性齢'].astype(str).str.extract(r'(\d+)')[0], errors='coerce')
        df_test['調教師_騎手'] = df_test['調教師'].astype(str)+'_'+df_test['騎手'].astype(str)

        # _safe_col を使い全列をSeriesで取得（スカラー返却を防止）
        df_test['3走前_着順'] = _safe_col(df_test, '2走前_着順')
        df_test['2走前_着順'] = _safe_col(df_test, '前走_着順')
        df_test['前走_着順']  = _safe_col(df_test, '最新_着順')
        df_test['過去3走平均着順'] = df_test[['前走_着順','2走前_着順','3走前_着順']].mean(axis=1)
        df_test['5走前_スピード指数'] = _safe_col(df_test, '4走前_スピード指数')
        df_test['4走前_スピード指数'] = _safe_col(df_test, '3走前_スピード指数')
        df_test['3走前_スピード指数'] = _safe_col(df_test, '2走前_スピード指数')
        df_test['2走前_スピード指数'] = _safe_col(df_test, '前走_スピード指数')
        df_test['前走_スピード指数']  = _safe_col(df_test, '最新_スピード指数')
        df_test['過去3走平均スピード指数']  = df_test[['前走_スピード指数','2走前_スピード指数','3走前_スピード指数']].mean(axis=1)
        df_test['近5走_中央値スピード指数'] = df_test[['前走_スピード指数','2走前_スピード指数','3走前_スピード指数','4走前_スピード指数','5走前_スピード指数']].median(axis=1)
        df_test['近5走_最高スピード指数']   = df_test[['前走_スピード指数','2走前_スピード指数','3走前_スピード指数','4走前_スピード指数','5走前_スピード指数']].max(axis=1)
        df_test['上昇度_スピード指数'] = df_test['前走_スピード指数']-df_test['近5走_中央値スピード指数']

        df_test['前走_通過'] = _safe_col(df_test, '最新_通過', '')
        def parse_corner(x):
            s=str(x); return s.split('-')[-1] if '-' in s else (s if s.isdigit() else np.nan)
        df_test['前走_最終コーナー'] = pd.to_numeric(df_test['前走_通過'].fillna('').astype(str).apply(parse_corner), errors='coerce')
        df_test['脚質カテゴリ'] = df_test['前走_最終コーナー'].apply(classify_style)
        df_test['前走逃げフラグ']  = (df_test['前走_最終コーナー']<=2).astype(int)
        df_test['前走先行フラグ']  = ((df_test['前走_最終コーナー']>2)&(df_test['前走_最終コーナー']<=5)).astype(int)
        df_test['同レース逃げ馬頭数'] = df_test['前走逃げフラグ'].sum()
        df_test['同レース先行馬頭数'] = df_test['前走先行フラグ'].sum()
        df_test['コース適性_着順パーセント'] = df_test.set_index(['馬ID','競馬場','芝/ダート']).index.map(horse_course_dict).fillna(0.5)
        df_test['位置取りショック'] = df_test['前走_最終コーナー'] - pd.to_numeric(_safe_col(df_test, '2走前_最終コーナー'), errors='coerce')

        race_date_obj = pd.to_datetime(race_date_str)

        # ★ データリーク防止: バックテスト・振り返り時は
        # 「最新_日付 >= race_date」の馬の前走情報をNaNにマスクする
        if skip_live_scrape and '最新_日付' in df_test.columns:
            future_mask = pd.to_datetime(df_test['最新_日付'], errors='coerce') >= race_date_obj
            leak_cols = ['最新_日付','最新_着順','最新_スピード指数','最新_人気','最新_上り',
                         '最新_距離','最新_馬体重','最新_騎手','最新_芝ダート','最新_着順パーセント',
                         '最新_失速フラグ','最新_上り偏差','最新_距離補正タイム差','最新_直近3走着順パーセント',
                         '前走_着順','2走前_着順','3走前_着順',
                         '前走_スピード指数','2走前_スピード指数','3走前_スピード指数',
                         '4走前_スピード指数','5走前_スピード指数',
                         '前走_最終コーナー','2走前_最終コーナー']
            for col in leak_cols:
                if col in df_test.columns:
                    df_test.loc[future_mask, col] = np.nan

        # =====================================================================
        # 前走情報をnetkeibaから直接スクレイプして上書き
        # CSVベースのlatest_horse_dataより確実（表記ゆれ・古いデータ問題を解決）
        # =====================================================================
        def _norm_name(s):
            if pd.isna(s): return ''
            # 全角スペース・記号除去
            n = re.sub(r'[\s\u3000\u2606\u25b2\u25b3\u25c7\u2605\[\]]', '', str(s))
            # 既知の短縮表記を正式名に変換
            return _JOCKEY_ABBR.get(n, n)

        # 各馬の前走情報をnetkeibaから直接スクレイプして上書き
        # skip_live_scrape=True(バックテスト): スキップして高速化
        # 前走日付 >= race_date の場合もスキップ（バックテスト日より未来の情報は使えない）
        if not skip_live_scrape:
            for idx, row in df_test.iterrows():
                hid = str(row['馬ID'])
                prev = fetch_horse_last_race(hid)
                if not prev:
                    continue

                # バックテスト日付より未来の前走は使わない
                if '前走日付' in prev:
                    try:
                        prev_dt = pd.to_datetime(prev['前走日付'], errors='coerce')
                        if pd.notna(prev_dt) and prev_dt >= race_date_obj:
                            continue  # 未来データは無視してCSVデータのまま
                    except: pass

                if '前走日付'   in prev: df_test.at[idx, '最新_日付']     = prev['前走日付']
                if '前走距離'   in prev: df_test.at[idx, '最新_距離']     = prev['前走距離']
                if '前走芝ダート' in prev: df_test.at[idx, '最新_芝ダート'] = prev['前走芝ダート']
                if '前走騎手'   in prev: df_test.at[idx, '最新_騎手']     = prev['前走騎手']
                if '前走着順'   in prev: df_test.at[idx, '最新_着順']     = prev['前走着順']

        # 休養日数: スクレイプ済みの最新_日付から計算
        if '最新_日付' in df_test.columns:
            last_dates = pd.to_datetime(df_test['最新_日付'], errors='coerce')
            kyuyo_raw  = (race_date_obj - last_dates).dt.days.astype('float64')
            # 0以下は同日/未来(バックテスト混入) → NaN
            df_test['休養日数'] = kyuyo_raw.where(kyuyo_raw > 0, other=np.nan)
        else:
            df_test['休養日数'] = np.nan

        # 乗り替わりフラグ: 正規化名で比較
        if '最新_騎手' in df_test.columns:
            now_j  = df_test['騎手'].apply(_norm_name)
            prev_j = df_test['最新_騎手'].apply(_norm_name)
            df_test['乗り替わりフラグ'] = ((now_j != prev_j) & (prev_j != '')).astype(int)
            df_test['_前走騎手']        = df_test['最新_騎手'].fillna('不明')
        else:
            df_test['乗り替わりフラグ'] = 0
            df_test['_前走騎手']        = '不明'

        # 馬場替わりフラグ
        # 障害レースは「障害」カテゴリとして扱い、芝/ダート→障害は常に「変更」とする
        # ただし障害→障害は変化なし
        if '最新_芝ダート' in df_test.columns:
            now_s  = df_test['芝/ダート'].fillna('').astype(str).str.strip()
            prev_s = df_test['最新_芝ダート'].fillna('').astype(str).str.strip()
            # 障害同士は変化なし扱い（今回も前走も障害なら変化なし）
            both_shogai = (now_s.str.contains('障') & prev_s.str.contains('障'))
            surf_changed = ((now_s != prev_s) & (prev_s != '') & ~both_shogai)
            df_test['馬場替わりフラグ'] = surf_changed.astype(int)
            df_test['_前走馬場']        = df_test['最新_芝ダート'].fillna('不明')
        else:
            df_test['馬場替わりフラグ'] = 0
            df_test['_前走馬場']        = '不明'
        df_test['前走芝ダート'] = df_test['_前走馬場']

        # 距離変更フラグ: float比較
        if '最新_距離' in df_test.columns:
            now_d  = pd.to_numeric(df_test['距離'],      errors='coerce')
            prev_d = pd.to_numeric(df_test['最新_距離'], errors='coerce')
            df_test['距離変更フラグ'] = ((now_d != prev_d) & prev_d.notna()).astype(int)
            df_test['_前走距離']      = prev_d
        else:
            df_test['距離変更フラグ'] = 0
            df_test['_前走距離']      = np.nan
        # _safe_col: 列の存在・スカラー/Series問わず常にSeriesを返す安全ラッパー使用
        df_test['前走失速フラグ']        = pd.to_numeric(_safe_col(df_test, '最新_失速フラグ',        0),   errors='coerce').fillna(0)
        df_test['前走上り偏差']          = pd.to_numeric(_safe_col(df_test, '最新_上り偏差',          np.nan), errors='coerce')
        df_test['前走着順パーセント']    = pd.to_numeric(_safe_col(df_test, '最新_着順パーセント',    np.nan), errors='coerce')
        df_test['直近3走着順パーセント'] = pd.to_numeric(_safe_col(df_test, '最新_直近3走着順パーセント', 0.5), errors='coerce').fillna(0.5)
        df_test['前走距離補正タイム差']  = pd.to_numeric(_safe_col(df_test, '最新_距離補正タイム差',  np.nan), errors='coerce')
        df_test['前走大敗フラグ']        = (pd.to_numeric(_safe_col(df_test, '最新_着順', np.nan), errors='coerce') / df_test['出走頭数'].replace(0,1) > 0.7).astype(int)
        df_test['馬体重増減']            = df_test['馬体重_num'] - pd.to_numeric(_safe_col(df_test, '最新_馬体重', np.nan), errors='coerce')
        df_test['斤量差'] = pd.to_numeric(df_test['斤量'],errors='coerce') - pd.to_numeric(df_test['斤量'],errors='coerce').mean()
        df_test['穴馬_距離変更一変']     = ((df_test['距離変更フラグ']==1)&(df_test['直近3走着順パーセント']<0.4)).astype(int)
        df_test['穴馬_馬場替わり一変']   = ((df_test['馬場替わりフラグ']==1)&(df_test['直近3走着順パーセント']<0.4)).astype(int)
        df_test['穴馬_勝負の乗り替わり'] = ((df_test['乗り替わりフラグ']==1)&(df_test['直近3走着順パーセント']<0.5)).astype(int)
        df_test['穴馬_実力馬の巻き返し'] = ((df_test['前走大敗フラグ']==1)&(df_test['近5走_最高スピード指数']>=55)).astype(int)
        df_test['回り']       = df_test['競馬場'].map(VENUE_MAWARI).fillna('不明')
        df_test['コース地形'] = df_test['競馬場'].map(VENUE_CHIKEI).fillna('不明')
        df_test['騎手_競馬場'] = df_test['騎手'].astype(str)+'_'+df_test['競馬場'].astype(str)
        df_test['騎手_距離']   = df_test['騎手'].astype(str)+'_'+df_test['距離'].astype(str)

        # ★修正BUG2: TE は TE_COLS と完全一致させる
        for col in TE_COLS:
            df_test[f'{col}_TE'] = df_test[col].map(te_dicts.get(col,{})).fillna(global_mean) if col in df_test.columns else global_mean

        for col in num_features:
            if col not in df_test.columns: df_test[col] = np.nan
            df_test[col] = pd.to_numeric(df_test[col], errors='coerce')

        for col in cat_features:
            if col not in df_test.columns: df_test[col] = '不明'
            cats = cat_categories_dict.get(col, ['不明'])
            if '不明' not in cats: cats.append('不明')
            df_test[col] = pd.Categorical(df_test[col].fillna('不明'), categories=cats)

        nige_count  = int(df_test['同レース逃げ馬頭数'].iloc[0]) if not df_test.empty else 0
        senko_count = int(df_test['同レース先行馬頭数'].iloc[0]) if not df_test.empty else 0
        if nige_count>=3: pace_text=f"🔥 【ハイペース濃厚】 前走逃げた馬が{nige_count}頭もおり先行争いが激化。差し・追込馬の台頭に警戒！"
        elif nige_count==0: pace_text=f"🐌 【スローペース濃厚】 確たる逃げ馬が不在。先行馬({senko_count}頭)の押し切り、前残りに注意。"
        else: pace_text=f"🐎 【ミドルペース】 逃げ馬{nige_count}頭、先行馬{senko_count}頭。平均的なペースで実力が反映されやすい展開。"

        raw_scores = model.predict(df_test[features])
        exp_scores = np.exp(raw_scores-np.max(raw_scores))
        win_probs  = exp_scores/np.sum(exp_scores)
        df_test['勝率(AI予測)']   = win_probs
        df_test['複勝率(AI予測)'] = np.clip(win_probs*2.8, 0, 0.99)
        df_test['期待値'] = df_test['勝率(AI予測)']*df_test['単勝オッズ']
        df_test['期待値'] = df_test['期待値'].clip(upper=50.0)  # 取消馬などの異常EV防止
        df_test = df_test.sort_values('勝率(AI予測)', ascending=False).reset_index(drop=True)
        marks = ['◎','〇','▲','△','☆']+['']*(len(df_test)-5)
        df_test['印'] = marks[:len(df_test)]

        p1,p2 = df_test.loc[0,'勝率(AI予測)'],df_test.loc[1,'勝率(AI予測)']
        score_diff = p1-p2
        top1_umaban = df_test.loc[0,'馬番']
        himo_umabans = df_test.loc[1:4,'馬番'].astype(str).tolist() if len(df_test)>=5 else df_test.loc[1:,'馬番'].astype(str).tolist()
        himo_str = "・".join(himo_umabans)
        has_unraced = ('新馬' in race_text) or ('未出走' in race_text) or df_test['前走_着順'].isna().any()
        ana_horse_nums = []; topics_list = []
        for rank, row in df_test.iterrows():
            if not has_unraced and rank>=4 and row['期待値']>=1.5:
                topics_list.append(f"📌 {row['馬名']} (期待値特大の穴馬！)")
                if f"{row['馬番']}番" not in ana_horse_nums: ana_horse_nums.append(f"{row['馬番']}番")
        ana_str = "・".join(str(n) for n in ana_horse_nums[:3]) if ana_horse_nums else ""

        if has_unraced:
            confidence_text = "🛑 【見送り推奨・未出走混在】 過去データのない馬が含まれており、AIの予測精度が担保できません。"
            reco = f"⚠️ **購入見送り** (データ不足によるリスク大)\n※観戦に留めるか、どうしても買う場合は◎ {top1_umaban}番 の単複を少額で。"
        elif p1>=0.25 and score_diff>=0.10:
            confidence_text = f"💎 【鉄板レース】 ◎が抜けた存在({p1*100:.1f}%)！ 軸は不動です。"
            reco = f"🎯 【本命・単勝勝負】 ◎ {top1_umaban}番 の単勝。\n  🔗 馬単・3連単: {top1_umaban}着固定 → 相手: {himo_str}"
            if ana_str: reco += f"\n  💣 余裕があれば穴馬({ana_str}番)へのヒモ流しも推奨。"
        elif score_diff<=0.03 and p1<0.20:
            confidence_text = "🌪️ 【波乱レース】 上位の実力が拮抗の大混戦！ 穴馬からのヒモ荒れに警戒してください。"
            reco = f"⚠️ 【ボックス推奨】 上位陣 ({top1_umaban}・{himo_str}番) の馬連・3連複ボックス。"
            if ana_str: reco += f"\n  💣 大穴狙い: 穴馬({ana_str}番)を絡めたワイドや3連複が面白いです。"
        else:
            confidence_text = "⚖️ 【中穴狙いレース】 上位はまとまっていますが、展開次第で伏兵の台頭もあります。"
            reco = f"🎯 【馬連・ワイド】 ◎ {top1_umaban}番 から相手 ({himo_str}番) への流し。"
            if ana_str: reco += f"\n  💣 妙味狙い: {top1_umaban}番から穴馬({ana_str}番)へのワイドで高配当！"

        # =====================================================
        # SHAP値による本命馬の推し理由テキスト生成
        # =====================================================
        shap_reason = ""
        try:
            best_horse_name = df_test.iloc[0]['馬名']
            X_best = df_test.iloc[[0]][features].copy()
            # カテゴリ列を整数コードに変換してからnumpy配列化
            # （DataFrameのままだとcategorical_feature不一致エラーが出るため）
            for col in cat_features:
                if col in X_best.columns and hasattr(X_best[col], 'cat'):
                    X_best[col] = X_best[col].cat.codes.astype(float)
                elif col in X_best.columns:
                    X_best[col] = 0.0
            X_arr = X_best.astype(float).fillna(0).values  # pure numpy
            booster = getattr(model, 'booster_', getattr(model, '_Booster', None))
            if booster is not None:
                shap_vals = booster.predict(X_arr, pred_contrib=True)
                contribs = shap_vals[0, :-1]  # 最後の列はbias
                # 特徴量名とSHAP値を紐付け、上位3つを抽出
                feat_contrib = list(zip(features, contribs))
                feat_contrib_sorted = sorted(feat_contrib, key=lambda x: x[1], reverse=True)
                top3 = feat_contrib_sorted[:3]
                # 特徴量名を日本語ラベルに変換
                feat_label = {
                    '近5走_中央値スピード指数': '近5走のスピード指数(中央値)',
                    '近5走_最高スピード指数': '過去最高スピード指数',
                    '上昇度_スピード指数': 'スピード指数の上昇度',
                    '前走_スピード指数': '前走スピード指数',
                    'コース適性_着順パーセント': 'このコースの適性',
                    '前走着順パーセント': '前走の相対着順',
                    '直近3走着順パーセント': '直近3走の安定感',
                    '前走距離補正タイム差': '前走のタイム差',
                    '前走上り偏差': '前走の末脚の切れ味',
                    '休養日数': '休養明けの上積み',
                    '乗り替わりフラグ': '騎手強化の乗り替わり',
                    '穴馬_実力馬の巻き返し': '前走凡走からの巻き返し',
                    '穴馬_勝負の乗り替わり': '勝負の乗り替わり',
                }
                reasons = [feat_label.get(f, f) for f, _ in top3]
                shap_reason = ("\n\n🤖 **AIの推し理由 (SHAP分析)**\n"
                               f"◎ **{best_horse_name}** が本命の最大の根拠は **「{reasons[0]}」** です。"
                               f"次いで **「{reasons[1]}」「{reasons[2]}」** が高評価でした。")
        except Exception as shap_e:
            shap_reason = f"（SHAP分析エラー: {shap_e}）"
        reco = reco + shap_reason if shap_reason else reco

        return df_test, topics_list, reco, pace_text, confidence_text, track_type, place, distance, error_log

    except Exception as e:
        tb = traceback.format_exc()
        error_log.append(f"❌ 予測AI内部で致命的なエラーが発生:\n{tb}")
        # エラーログが空でも必ず何か入るよう保証
        if not error_log:
            error_log = [f"❌ 不明なエラー: {str(e)}"]
        return None,None,None,None,None,None,None,None,error_log

# ==========================================
# 4. メインUI構成
# ==========================================
st.sidebar.markdown("## 🕹️ keiba-ebye メニュー")

# ── Free / Pro 認証 ─────────────────────────────────────────
_PRO_PASSWORD = os.environ.get("PRO_PASSWORD", "")  # HuggingFace Secrets に設定

if _PRO_PASSWORD:
    _is_pro = st.session_state.get('is_pro', False)

    # ── オーナー自動ログイン ─────────────────────────────────
    # OWNER_KEY Secret を設定しておくと、URLパラメータ ?key=<OWNER_KEY> で
    # パスワード入力なしに自動ログインできる
    _OWNER_KEY = os.environ.get("OWNER_KEY", "")
    if not _is_pro and _OWNER_KEY:
        _url_key = st.query_params.get("key", "")
        if _url_key == _OWNER_KEY:
            st.session_state['is_pro'] = True
            st.session_state['is_owner'] = True
            _is_pro = True

    if not _is_pro:
        _pw_input = st.sidebar.text_input("🔑 Pro アクセスコード", type="password",
                                           placeholder="Proメンバーはここに入力")
        if _pw_input == _PRO_PASSWORD:
            st.session_state['is_pro'] = True
            st.rerun()
        st.sidebar.caption("アクセスコードなしでも「次のレースを予想」は無料でご利用いただけます。")
    else:
        _is_owner = st.session_state.get('is_owner', False)
        if _is_owner:
            st.sidebar.success("👑 オーナーログイン中")
        else:
            st.sidebar.success("✅ Pro メンバー")
            if st.sidebar.button("ログアウト", key="logout"):
                st.session_state['is_pro'] = False
                st.session_state['is_owner'] = False
                st.rerun()
else:
    _is_pro = True  # PRO_PASSWORD未設定 = 全機能開放（開発時・移行期）

# Proのみのメニュー項目
_PRO_ACTIONS = [
    "📅 今週末の全レース予想",
    "🔍 レースを指定して予想",
    "📝 1日の振り返り (答え合わせ)",
    "🧪 性能試験 (バックテスト)",
    "📈 長期成績分析",
    "📊 モデル検証 (ウォークフォワード)",
    "🏇 騎手・調教師フォーム分析",
    "📝 馬券メモ管理",
    "🐴 愛馬の成長記録",
]
_FREE_ACTIONS = ["⏩ 次のレースを予想"]
_ALL_ACTIONS = _FREE_ACTIONS + _PRO_ACTIONS

action = st.sidebar.radio(
    "機能を選択",
    _FREE_ACTIONS if not _is_pro else _ALL_ACTIONS
)

# Pro機能に直接アクセスしようとした場合のガード
if action in _PRO_ACTIONS and not _is_pro:
    st.warning("この機能はProメンバー限定です。サイドバーにアクセスコードを入力してください。")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.markdown("### 💰 軍資金シミュレーター")
sim_budget     = st.sidebar.number_input("軍資金 (円)", 1000, 500000, 30000, 1000,
                   help="1日の総予算。ケリー基準でここから各レースに配分します。")
sim_ev_filter  = st.sidebar.slider("購入する期待値の下限", 1.0, 3.0, 1.2, 0.1,
                   help="この期待値以上の馬だけを買います。高いほど厳選。")
sim_kelly_frac = st.sidebar.slider("ケリー係数", 0.1, 1.0, 0.25, 0.05,
                   help="1.0=フルケリー(高リスク) 0.25=推奨(安定)")
sim_max_per_race = st.sidebar.slider("1レース最大投資額 (軍資金の%)", 5, 40, 20, 5,
                   help="1レースに軍資金の何%まで使うか上限を設定します。") / 100

# ── モデル管理 (Pro + HF Hubが設定済みの場合のみ表示) ─────
if _is_pro and _hub_available:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ モデル管理")
    st.sidebar.caption(f"HF Hub: `{_HF_REPO_ID}`")
    if st.sidebar.button("🔄 強制再学習 & Hub更新", help="データが更新された際に手動で再学習してHubにアップロードします"):
        st.cache_resource.clear()
        with st.spinner("再学習中... (数分かかります)"):
            (model, features, cat_features, num_features, cat_categories_dict,
             latest_horse_data, horse_course_dict, ped_dict,
             known_jockeys, known_trainers, te_dicts, global_mean, recent_return_rate) = prepare_model_and_data(force_retrain=True)
        st.sidebar.success("✅ 再学習完了・Hubにアップロードしました")
        st.rerun()

tokyo_tz = pytz.timezone('Asia/Tokyo')
now = datetime.datetime.now(tokyo_tz)

def display_error_log(err_log):
    st.error("⚠️ 予想データまたは結果の取得に失敗しました。")
    with st.expander("🔍 エラー解析ログを見る (デバッグ用)", expanded=True):
        if not err_log:
            st.write("（エラーログなし: ネットワーク接続またはサイト側の問題の可能性があります）")
        for log in err_log:
            st.code(log, language=None)

def display_result(df_res, topics, reco, pace_text, confidence_text, show_change_table=True):
    tab1, tab2, tab3, tab4 = st.tabs(["📊 予想一覧", "💡 展開・買い目", "🔍 性能詳細", "🎰 複合馬券EV"])

    with tab1:
        if "鉄板" in confidence_text: st.success(confidence_text)
        elif "波乱" in confidence_text: st.error(confidence_text)
        else: st.info(confidence_text)

        # ── 軍資金シミュレーター ─────────────────────────────
        def calc_kelly_sim(p_raw, odds_raw):
            if "見送り" in confidence_text: return 0
            try:
                p = float(str(p_raw).replace('%',''))/100 if '%' in str(p_raw) else float(p_raw)
                b = float(odds_raw) - 1.0
            except: return 0
            if b <= 0: return 0
            if p * float(odds_raw) < sim_ev_filter: return 0
            f_star = p - (1.0 - p) / b
            if f_star <= 0: return 0
            bet = int(min(f_star * sim_kelly_frac * sim_budget, sim_budget * sim_max_per_race) / 100) * 100
            return max(0, bet)

        # メモ読み込み
        memo_horses = st.session_state.get('memo_horses', [])
        all_memos = {}
        if memo_horses and os.path.exists("horse_memos.json"):
            try:
                with open("horse_memos.json", encoding="utf-8") as _mf:
                    all_memos = json.load(_mf)
            except: pass

        # ── 🏇 馬柱UI ─────────────────────────────────────────
        import altair as alt
        st.markdown("#### 🏇 出走馬 AI予想")
        total_bet = 0

        # ── 馬柱テーブル（上段: 印付きリスト）─────────────────
        bets = []
        for _, row in df_res.iterrows():
            bet = calc_kelly_sim(row['勝率(AI予測)'], row['単勝オッズ'])
            bets.append(f"¥{bet:,}" if bet > 0 else "見送り")
            if bet > 0: total_bet += bet

        show_df = df_res[['印','枠番','馬番','馬名','脚質カテゴリ','単勝オッズ','勝率(AI予測)','複勝率(AI予測)','期待値']].copy()
        show_df = show_df.rename(columns={'勝率(AI予測)':'勝率','複勝率(AI予測)':'複勝率','単勝オッズ':'オッズ','脚質カテゴリ':'脚質'})
        show_df['💰推奨'] = bets
        show_df['勝率']   = (show_df['勝率'] * 100).map('{:.1f}%'.format)
        show_df['複勝率'] = (show_df['複勝率'] * 100).map('{:.1f}%'.format)

        if total_bet > 0:
            st.caption(f"💰 推奨投資合計: **¥{total_bet:,}** / 軍資金¥{sim_budget:,}の {total_bet/sim_budget*100:.1f}%")

        # メモあり馬のメッセージ
        for _, row in df_res.iterrows():
            hname = row['馬名']
            if hname in all_memos:
                latest = sorted(all_memos[hname], key=lambda x: x["日付"], reverse=True)[0]
                st.info(f"📝 **{hname}** ({row['印']}) にメモあり: {latest['タグ']} {latest['日付']} — {latest['メモ'] or '(内容なし)'}")

        def highlight_row(row):
            horse_name = row.get('馬名', '')
            bet_str    = row.get('💰推奨', '見送り')
            try:
                ev = float(row.get('期待値', 0) or 0)
            except (ValueError, TypeError):
                ev = 0.0
            if horse_name in memo_horses:
                return ['border-left:3px solid #9B59B6;background:rgba(155,89,182,0.08)'] * len(row)
            # EV1.5以上は常に赤ハイライト
            if ev >= 1.5:
                return ['background-color:rgba(255,75,75,0.18)'] * len(row)
            if bet_str != '見送り':
                return ['background-color:rgba(255,200,0,0.10)'] * len(row)
            return [''] * len(row)

        st.dataframe(
            show_df.style.apply(highlight_row, axis=1)
                   .format({'期待値':'{:.2f}','オッズ':'{:.1f}','枠番':'{:.0f}','馬番':'{:.0f}'}),
            use_container_width=True, hide_index=True
        )

        # ── リアルタイム勝率バー（下段: グラフ）────────────────
        st.markdown("---")
        bar_df = df_res[['馬番','馬名','印','勝率(AI予測)','期待値','単勝オッズ']].copy()
        bar_df['勝率%'] = (bar_df['勝率(AI予測)'] * 100).round(1)
        bar_df['label'] = bar_df['印'] + ' ' + bar_df['馬名']
        bar_df['color'] = bar_df['期待値'].apply(
            lambda v: '#FF4B4B' if v >= 1.5 else '#4B8BFF'
        )
        bar_chart = alt.Chart(bar_df).mark_bar(cornerRadiusEnd=4).encode(
            y=alt.Y('label:N', sort=list(bar_df['label']), title='', axis=alt.Axis(labelFontSize=12)),
            x=alt.X('勝率%:Q', title='AI勝率 (%)', scale=alt.Scale(domain=[0, bar_df['勝率%'].max()*1.15])),
            color=alt.Color('color:N', scale=None, legend=None),
            tooltip=['馬名', '勝率%', '期待値', '単勝オッズ']
        ).properties(height=max(200, len(bar_df) * 32))
        text_chart = alt.Chart(bar_df).mark_text(align='left', dx=4, fontSize=11).encode(
            y=alt.Y('label:N', sort=list(bar_df['label'])),
            x=alt.X('勝率%:Q'),
            text=alt.Text('勝率%:Q', format='.1f'),
            color=alt.value('var(--color-text-secondary)')
        )
        st.altair_chart((bar_chart + text_chart).configure_view(strokeWidth=0), use_container_width=True)
        st.caption("赤バー = EV1.5以上の注目馬 / 青バー = 通常")

    with tab2:
        st.info(f"**🏇 展開予想:**\n{pace_text}")
        ev_horses = df_res[(df_res.index < 5) & (df_res['期待値'] >= sim_ev_filter)]
        if not ev_horses.empty:
            st.error(f"💰 **【期待値レーダー発動】** {', '.join(ev_horses['馬名'].tolist())} に妙味あり！")
        if topics: st.warning("**📝 要注目トピック馬:**\n\n" + "\n".join(topics))
        st.success(f"**🤖 AI推奨買い目:**\n\n{reco}")

    with tab3:
        import altair as alt

        # 見方ガイド
        with st.expander("📖 指標の見方・解説 (クリックで開く)", expanded=False):
            guide_data = {
                "指標": ["地力(中央値)", "最高ポテンシャル", "上昇度",
                          "コース適性", "位置取り変化", "近3走安定度",
                          "休養日数", "騎手変化", "馬場変化", "距離変化"],
                "見方・ポイント": [
                    "近5走スピード指数の中央値。50が平均、高いほど強い。最も信頼できる実力値",
                    "近5走の最高値。地力との差が大きい馬は条件次第で爆発力がある",
                    "前走指数-近5走中央値。+2以上=上昇中(赤強調)、-2以下=下降気味(グレー)",
                    "このコースでの過去着順パーセント。0に近いほどコースが得意(緑強調)",
                    "今回予想コーナー順位-前走。マイナス=前に行きそう、プラス=後退しそう",
                    "直近3走の着順パーセント平均。0.3以下=安定(緑)、0.6以上=不安定",
                    "前走からの間隔日数。14日以内=超短期、56日以上=休み明けで状態要確認",
                    "前走と今回の騎手比較。格上騎手への乗り替わりは要注目",
                    "芝/ダートの変更。初芝・初ダートは過去実績との乖離リスクあり",
                    "距離の変更。距離延長/短縮で得意不得意が変わる。適性距離の確認推奨",
                ],
            }
            st.dataframe(pd.DataFrame(guide_data), use_container_width=True, hide_index=True)

        # スコアテーブル
        st.markdown("#### 📐 AI評価スコア詳細")
        score_cols_map = {
            '近5走_中央値スピード指数': '地力(中央値)',
            '近5走_最高スピード指数':   '最高ポテンシャル',
            '上昇度_スピード指数':       '上昇度',
            'コース適性_着順パーセント': 'コース適性',
            '位置取りショック':          '位置取り変化',
            '直近3走着順パーセント':     '近3走安定度',
        }
        avail_s = {k: v for k, v in score_cols_map.items() if k in df_res.columns}
        score_df = df_res[['馬番', '馬名', '脚質カテゴリ'] + list(avail_s.keys())].copy()
        score_df = score_df.rename(columns=avail_s)

        def highlight_score(row):
            styles = [''] * len(row)
            cols = list(row.index)
            for col, fn in [
                ('上昇度', lambda v: 'color:#FF4B4B;font-weight:bold' if v >= 2.0
                           else ('color:#888888' if v <= -2.0 else '')),
                ('地力(中央値)', lambda v: 'color:#4B8BFF;font-weight:bold' if v >= 55 else ''),
                ('コース適性', lambda v: 'color:#22AA22;font-weight:bold' if v <= 0.2 else ''),
                ('近3走安定度', lambda v: 'color:#22AA22;font-weight:bold' if v <= 0.3
                              else ('color:#888888' if v >= 0.6 else '')),
            ]:
                if col in cols:
                    try:
                        styles[cols.index(col)] = fn(float(row[col]))
                    except: pass
            return styles

        fmt_s = {v: '{:.2f}' for v in avail_s.values()}
        st.dataframe(
            score_df.style.apply(highlight_score, axis=1).format(fmt_s, na_rep='不明'),
            use_container_width=True, hide_index=True
        )
        st.caption("赤=上昇度+2以上 / 青=地力55以上 / 緑=コース適性0.2以下 or 安定度0.3以下")

        # 前走比較テーブル（バックテスト・振り返りではshow_change_table=Falseで非表示）
        if show_change_table:
            st.markdown("#### 🔄 前走との変化点チェック")

            def fmt_kyuyo(days):
                try:
                    v = float(days)
                    if np.isnan(v) or v <= 0: return '不明'
                    d = int(v)
                    if d <= 14:  return f'[超短]{d}日'
                    if d <= 28:  return f'[中2週]{d}日'
                    if d <= 56:  return f'[中3-7週]{d}日'
                    return f'[休み明け]{d}日'
                except: return '不明'

            change_rows = []
            for _, row in df_res.iterrows():
                flag_j  = int(row.get('乗り替わりフラグ', 0) or 0)
                prev_j  = str(row['_前走騎手']) if '_前走騎手' in df_res.columns and not pd.isna(row.get('_前走騎手')) else '不明'
                now_j   = str(row.get('騎手', ''))
                jockey_str = f"変更:{prev_j}->{now_j}" if flag_j == 1 and prev_j not in ('不明', '', 'nan') else '変化なし'

                flag_s  = int(row.get('馬場替わりフラグ', 0) or 0)
                prev_s  = str(row['_前走馬場']) if '_前走馬場' in df_res.columns and not pd.isna(row.get('_前走馬場')) else '不明'
                now_s   = str(row.get('芝/ダート', ''))
                surf_str = f"変更:{prev_s}->{now_s}" if flag_s == 1 else now_s

                flag_d  = int(row.get('距離変更フラグ', 0) or 0)
                prev_d  = row.get('_前走距離') if '_前走距離' in df_res.columns else np.nan
                now_d   = row.get('距離', np.nan)
                if flag_d == 1 and not pd.isna(prev_d):
                    diff = int(float(now_d)) - int(float(prev_d))
                    dist_str = f"変更:{int(float(prev_d))}m->{int(float(now_d))}m({'+' if diff>0 else ''}{diff}m)"
                else:
                    try:    dist_str = f"{int(float(now_d))}m"
                    except: dist_str = '不明'

                change_rows.append({
                    '馬番':     row['馬番'],
                    '馬名':     row['馬名'],
                    '休養日数': fmt_kyuyo(row.get('休養日数', np.nan)),
                    '騎手変化': jockey_str,
                    '馬場変化': surf_str,
                    '距離変化': dist_str,
                })
            change_df = pd.DataFrame(change_rows)

            def highlight_change(row):
                styles = [''] * len(row)
                cols = list(row.index)
                for c in ['騎手変化', '馬場変化', '距離変化']:
                    if c in cols and '変更:' in str(row.get(c, '')):
                        styles[cols.index(c)] = 'color:#FFA500;font-weight:bold'
                if '休養日数' in cols:
                    v = str(row.get('休養日数', ''))
                    if '休み明け' in v:  styles[cols.index('休養日数')] = 'color:#888888'
                    elif '超短' in v:    styles[cols.index('休養日数')] = 'color:#FF6666'
                return styles

            st.dataframe(change_df.style.apply(highlight_change, axis=1), use_container_width=True, hide_index=True)
            st.caption("オレンジ=変化あり / [超短]=14日以内 / [休み明け]=56日以上")

        # スピード指数バーチャート
        if '近5走_中央値スピード指数' in df_res.columns:
            st.markdown("#### 📊 地力比較チャート")
            chart_data = df_res[['馬名', '近5走_中央値スピード指数']].copy()
            if '近5走_最高スピード指数' in df_res.columns:
                chart_data['近5走_最高スピード指数'] = df_res['近5走_最高スピード指数']
            chart_data = chart_data.dropna(subset=['近5走_中央値スピード指数'])
            chart_data = chart_data.sort_values('近5走_中央値スピード指数', ascending=False).head(12)
            base = alt.Chart(chart_data).encode(y=alt.Y('馬名:N', sort='-x', title=''))
            bar  = base.mark_bar(color='#4B8BFF', opacity=0.75).encode(
                x=alt.X('近5走_中央値スピード指数:Q', title='スピード指数'),
                tooltip=['馬名', '近5走_中央値スピード指数']
            )
            layer = bar
            if '近5走_最高スピード指数' in chart_data.columns:
                tick = base.mark_tick(color='#FF4B4B', thickness=2, size=15).encode(
                    x='近5走_最高スピード指数:Q', tooltip=['馬名', '近5走_最高スピード指数']
                )
                layer = bar + tick
            st.altair_chart(layer.properties(height=max(220, len(chart_data)*30)), use_container_width=True)
            st.caption("青バー=近5走中央値(地力) / 赤ティック=近5走最高値(ポテンシャル)")

    with tab4:
        st.markdown("AI勝率から計算した複合馬券の理論期待値です。**1.0以上**が購入検討ライン。")
        probs = df_res['勝率(AI予測)'].values
        odds_list = df_res['単勝オッズ'].values
        names = df_res['馬名'].values
        nums = df_res['馬番'].values

        # 複勝率（近似）
        fukusho_probs = np.clip(probs * 2.8, 0, 0.99)

        # 馬連・ワイドの期待値（上位5頭の組み合わせ）
        umaren_rows, wide_rows = [], []
        for a in range(min(5, len(probs))):
            for b in range(a+1, min(7, len(probs))):
                # 馬連: aかbが1着でもう一方が2着
                p_umaren = probs[a]*fukusho_probs[b] + probs[b]*fukusho_probs[a]
                p_umaren = min(p_umaren, 0.99)
                # ワイド: 両馬が3着以内
                p_wide = fukusho_probs[a] * fukusho_probs[b] * 1.5  # 相関補正
                p_wide = min(p_wide, 0.99)
                # 単勝オッズから馬連オッズを推定（簡易モデル: 単勝の積÷0.7）
                est_umaren_odds = (odds_list[a] * odds_list[b]) / 8.0
                est_wide_odds   = (odds_list[a] * odds_list[b]) / 20.0
                ev_umaren = p_umaren * est_umaren_odds
                ev_wide   = p_wide   * est_wide_odds
                umaren_rows.append({'組合せ': f'{nums[a]}-{nums[b]}', '馬名': f'{names[a]} - {names[b]}',
                                    '推定EV': round(ev_umaren, 2), '理論的中率': f'{p_umaren*100:.1f}%',
                                    '推定オッズ': f'{est_umaren_odds:.1f}倍'})
                wide_rows.append({'組合せ': f'{nums[a]}-{nums[b]}', '馬名': f'{names[a]} - {names[b]}',
                                  '推定EV': round(ev_wide, 2), '理論的中率': f'{p_wide*100:.1f}%',
                                  '推定オッズ': f'{est_wide_odds:.1f}倍'})

        # 3連複（上位3頭固定）
        sanrenpuku_rows = []
        for a in range(min(4, len(probs))):
            for b in range(a+1, min(5, len(probs))):
                for c in range(b+1, min(6, len(probs))):
                    p3 = fukusho_probs[a] * fukusho_probs[b] * fukusho_probs[c] * 3.0
                    p3 = min(p3, 0.99)
                    est_odds3 = (odds_list[a] * odds_list[b] * odds_list[c]) / 20.0
                    ev3 = p3 * est_odds3
                    sanrenpuku_rows.append({'組合せ': f'{nums[a]}-{nums[b]}-{nums[c]}',
                                            '推定EV': round(ev3, 2), '理論的中率': f'{p3*100:.1f}%',
                                            '推定オッズ': f'{est_odds3:.0f}倍'})

        def color_ev(val):
            if isinstance(val, float):
                if val >= 1.5: return 'color:#FF4B4B; font-weight:bold'
                if val >= 1.0: return 'color:#FFA500; font-weight:bold'
            return ''

        sub1, sub2, sub3 = st.tabs(["馬連", "ワイド", "3連複"])
        with sub1:
            st.caption("※ オッズは単勝オッズから推定した理論値です。実際のオッズとは異なります。")
            df_uma = pd.DataFrame(umaren_rows).sort_values('推定EV', ascending=False)
            st.dataframe(df_uma.style.applymap(color_ev, subset=['推定EV']).format({'推定EV': '{:.2f}'}), use_container_width=True, hide_index=True)
        with sub2:
            df_wid = pd.DataFrame(wide_rows).sort_values('推定EV', ascending=False)
            st.dataframe(df_wid.style.applymap(color_ev, subset=['推定EV']).format({'推定EV': '{:.2f}'}), use_container_width=True, hide_index=True)
        with sub3:
            df_san = pd.DataFrame(sanrenpuku_rows).sort_values('推定EV', ascending=False).head(10)
            st.dataframe(df_san.style.applymap(color_ev, subset=['推定EV']).format({'推定EV': '{:.2f}'}), use_container_width=True, hide_index=True)


if action in ["⏩ 次のレースを予想", "🔍 レースを指定して予想"]:
    todays_races = get_todays_races()
    if not todays_races: st.warning(f"本日 ({now.strftime('%Y/%m/%d')}) はJRAのレースが開催されていません。")
    else:
        if action == "⏩ 次のレースを予想":
            st.subheader("🕒 まもなく出走するレース")
            races_sorted_by_time = sorted(todays_races, key=lambda x: x['time'])
            next_race = next((r for r in races_sorted_by_time if r['time'] > now), None)

            if next_race:
                mins_left = int((next_race['time'] - now).total_seconds() / 60)
                st.info(f"👉 **{next_race['place']} {next_race['num']}R** 「{next_race['title']}」 (あと **{mins_left}** 分)")

                # ── オッズ自動再取得 ─────────────────────────────────
                # 発走5分前になったら自動で予想を更新するオプション
                auto_refresh = st.checkbox(
                    "🔄 発走5分前になったら自動でオッズを再取得して予想を更新する",
                    value=False,
                    help="チェックを入れると発走5分前に自動で予想を再実行します"
                )
                col_btn1, col_btn2 = st.columns([2, 1])
                with col_btn1:
                    manual_run = st.button("🚀 keiba-ebye 予想起動！", type="primary")
                with col_btn2:
                    force_refresh = st.button("🔄 オッズ再取得して更新", help="最新オッズで予想を再実行します")

                # 自動更新ロジック: 10分前になったら自動実行
                auto_triggered = False
                if auto_refresh and 0 <= mins_left <= 5:
                    last_auto = st.session_state.get(f'last_auto_{next_race["id"]}', 0)
                    if time.time() - last_auto > 300:  # 5分以上経過したら再実行
                        auto_triggered = True
                        st.session_state[f'last_auto_{next_race["id"]}'] = time.time()
                        st.warning(f"⚡ 発走{mins_left}分前！最新オッズで予想を更新します...")

                if manual_run or force_refresh or auto_triggered:
                    with st.spinner('AIが推論中（最新オッズ取得含む）...'):
                        res_df, topics, reco, pace_text, conf_text, _, _, _, err_log = run_real_prediction(next_race['id'], now.strftime('%Y-%m-%d'))
                        if res_df is not None:
                            display_result(res_df, topics, reco, pace_text, conf_text)
                            if force_refresh or auto_triggered:
                                st.success("✅ オッズを再取得して予想を更新しました")
                        else: display_error_log(err_log)

                # 自動更新チェック用ページ再読み込み
                if auto_refresh and mins_left > 5:
                    st.caption(f"次の自動確認まで... 発走{mins_left}分前（5分前になると自動更新）")
                    # 1分ごとに再確認
                    time.sleep(0)  # Streamlitの再実行をトリガーしない（ボタン待ち）
            else:
                st.success("🏁 本日の全レースは終了しました。")
            
        elif action == "🔍 レースを指定して予想":
            options = [f"{r['place']} {r['num']}R - {r['title']}" for r in todays_races]
            selected = st.selectbox("レースを選んでください", options)
            target_race = todays_races[options.index(selected)]
            if st.button("🚀 予想開始", type="primary"):
                with st.spinner('推論中...'):
                    res_df, topics, reco, pace_text, conf_text, _, _, _, err_log = run_real_prediction(target_race['id'], now.strftime('%Y-%m-%d'))
                    if res_df is not None: display_result(res_df, topics, reco, pace_text, conf_text)
                    else: display_error_log(err_log)

elif action == "📅 今週末の全レース予想":
    st.subheader("📅 今週末 (土・日) の先取り予想")
    sat_str, sun_str = get_weekend_dates()
    col1, col2 = st.columns(2)
    with col1: run_sat = st.button(f"🚀 土曜日 ({sat_str[4:6]}/{sat_str[6:]}) の予想", type="primary")
    with col2: run_sun = st.button(f"🚀 日曜日 ({sun_str[4:6]}/{sun_str[6:]}) の予想", type="primary")
    target_date = sat_str if run_sat else sun_str if run_sun else None
    
    if target_date:
        with st.spinner(f'出馬表を収集中...'):
            target_races = get_todays_races(target_date)
        if not target_races: st.error("出馬表が未発表です。")
        else:
            my_bar = st.progress(0, text="推論中...")
            results_for_txt = []
            for i, r in enumerate(target_races):
                with st.expander(f"🏁 {r['place']} {r['num']}R"):
                    res_df, topics, reco, pace_text, conf_text, track_type, place, dist, err_log = run_real_prediction(r['id'], f"{target_date[:4]}-{target_date[4:6]}-{target_date[6:]}")
                    # 枠番未確定・オッズ暫定の警告を表示
                    waku_warn = [e for e in (err_log or []) if '枠順未確定' in e]
                    if waku_warn:
                        st.warning(waku_warn[0])
                    if res_df is not None:
                        display_result(res_df, topics, reco, pace_text, conf_text)
                        results_for_txt.append({'date': f"{target_date[:4]}年{target_date[4:6]}月{target_date[6:]}日", 'place': place, 'num': r['num'], 'track': track_type, 'dist': dist, 'pace': pace_text, 'confidence': conf_text, 'df': res_df, 'topics': topics, 'reco': reco})
                    else: display_error_log(err_log)
                time.sleep(1.0)
                my_bar.progress((i + 1) / len(target_races))
            if results_for_txt:
                _dw1, _dw2 = st.columns(2)
                _dw1.download_button(f"📥 {target_date[4:6]}/{target_date[6:]} 予想レポート(.txt)", data=generate_txt_report(results_for_txt), file_name=f"keiba_weekend_{target_date}.txt", mime="text/plain")
                _html_w = generate_pdf_report(results_for_txt)
                if _html_w:
                    _dw2.download_button(f"🌐 {target_date[4:6]}/{target_date[6:]} 予想レポート(.html)", data=_html_w, file_name=f"keiba_weekend_{target_date}.html", mime="text/html")

elif action == "📝 1日の振り返り (答え合わせ)":
    st.subheader("📝 1日のレース結果とAI予想の答え合わせ")
    target_date = st.date_input("振り返りたい日付を選択", datetime.date.today() - datetime.timedelta(days=1))

    if st.button("🚀 振り返り実行！", type="primary"):
        with st.spinner(f'{target_date.strftime("%Y/%m/%d")} のレースデータと結果を取得・集計中...'):
            races = get_todays_races(target_date.strftime('%Y%m%d'))
            if not races:
                st.error("指定した日付のレースが見つかりません。")
            else:
                my_bar = st.progress(0, text="集計中...")

                stats = {
                    'honmei_races': 0, 'honmei_tan_hits': 0, 'honmei_tan_return': 0,
                    'honmei_fuku_hits': 0, 'honmei_fuku_return': 0,
                    'umaren_races': 0, 'umaren_invest': 0, 'umaren_hits': 0, 'umaren_return': 0,
                    'wide_ana_races': 0, 'wide_ana_invest': 0, 'wide_ana_hits': 0, 'wide_ana_return': 0,
                    'ev_invest': 0, 'ev_tan_hits': 0, 'ev_tan_return': 0, 'ev_fuku_hits': 0, 'ev_fuku_return': 0,
                    'shiba_races': 0, 'shiba_return': 0, 'dart_races': 0, 'dart_return': 0,
                    'exp_races': 0, 'exp_return': 0, 'new_races': 0, 'new_return': 0,
                }

                for i, r in enumerate(races):
                    res_df, topics, reco, pace_text, conf_text, track_type, place, dist, err_log = run_real_prediction(r['id'], target_date.strftime('%Y-%m-%d'), skip_live_scrape=True)
                    payouts = get_all_payouts(r['id'])

                    # =========================================================
                    # レースごとの予想を expander で表示（★追加）
                    # =========================================================
                    honmei_name = res_df.iloc[0]['馬名'] if res_df is not None else "不明"
                    honmei_num  = res_df.iloc[0]['馬番'] if res_df is not None else "-"
                    tan_pay = payouts['tansho'].get(honmei_num, 0) if res_df is not None else 0
                    hit_icon = "✅" if tan_pay > 0 else ("❌" if res_df is not None and payouts['tansho'] else "⚠️")
                    expander_label = f"{hit_icon} {r['place']} {r['num']}R  ◎{honmei_num}番 {honmei_name}"
                    if tan_pay > 0:
                        expander_label += f"  → 単勝 {tan_pay/100:.1f}倍 的中！"

                    with st.expander(expander_label, expanded=False):
                        if res_df is not None:
                            display_result(res_df, topics, reco, pace_text, conf_text, show_change_table=False)
                            # 払い戻し結果を表示
                            if payouts['tansho']:
                                st.markdown("##### 📋 払い戻し結果")
                                result_rows = []
                                for rank_i, row in res_df.iterrows():
                                    uma = row['馬番']
                                    tan = payouts['tansho'].get(uma, 0)
                                    fuku = payouts['fukusho'].get(uma, 0)
                                    if rank_i < 5 or tan > 0 or fuku > 0:
                                        result_rows.append({
                                            '印': row['印'],
                                            '馬番': uma,
                                            '馬名': row['馬名'],
                                            'AI勝率': f"{row['勝率(AI予測)']*100:.1f}%",
                                            '単勝払戻': f"¥{tan:,}" if tan > 0 else '-',
                                            '複勝払戻': f"¥{fuku:,}" if fuku > 0 else '-',
                                        })
                                if result_rows:
                                    st.dataframe(pd.DataFrame(result_rows), use_container_width=True, hide_index=True)
                            else:
                                st.warning("払い戻しデータが取得できませんでした")
                        else:
                            display_error_log(err_log)

                    # =========================================================
                    # 集計処理（従来通り）
                    # =========================================================
                    if res_df is not None and payouts['tansho']:
                        honmei = res_df.iloc[0]['馬番']
                        has_unraced = ('新馬' in r['title']) or ('未出走' in r['title']) or (res_df['前走_着順'].isna().any() if '前走_着順' in res_df.columns else False)

                        stats['honmei_races'] += 1
                        if track_type == "芝": stats['shiba_races'] += 1
                        elif track_type == "ダート": stats['dart_races'] += 1
                        if has_unraced: stats['new_races'] += 1
                        else: stats['exp_races'] += 1

                        if honmei in payouts['tansho']:
                            stats['honmei_tan_hits'] += 1
                            stats['honmei_tan_return'] += payouts['tansho'][honmei]
                            if track_type == "芝": stats['shiba_return'] += payouts['tansho'][honmei]
                            elif track_type == "ダート": stats['dart_return'] += payouts['tansho'][honmei]
                            if has_unraced: stats['new_return'] += payouts['tansho'][honmei]
                            else: stats['exp_return'] += payouts['tansho'][honmei]

                        if honmei in payouts['fukusho']:
                            stats['honmei_fuku_hits'] += 1
                            stats['honmei_fuku_return'] += payouts['fukusho'][honmei]

                        if len(res_df) >= 5:
                            himo_list = res_df.iloc[1:5]['馬番'].tolist()
                            stats['umaren_races'] += 1
                            stats['umaren_invest'] += len(himo_list) * 100
                            for himo in himo_list:
                                key = tuple(sorted([honmei, himo]))
                                if key in payouts['umaren']:
                                    stats['umaren_hits'] += 1
                                    stats['umaren_return'] += payouts['umaren'][key]

                        ana_list = res_df[(res_df.index >= 4) & (res_df['期待値'] >= 1.5)]['馬番'].tolist()
                        if ana_list:
                            stats['wide_ana_races'] += 1
                            stats['wide_ana_invest'] += len(ana_list) * 100
                            for ana in ana_list:
                                key = tuple(sorted([honmei, ana]))
                                if key in payouts['wide']:
                                    stats['wide_ana_hits'] += 1
                                    stats['wide_ana_return'] += payouts['wide'][key]

                        ev_list = res_df[(res_df.index < 5) & (res_df['期待値'] >= 1.5)]['馬番'].tolist()
                        if ev_list:
                            stats['ev_invest'] += len(ev_list) * 100
                            for ev in ev_list:
                                if ev in payouts['tansho']:
                                    stats['ev_tan_hits'] += 1
                                    stats['ev_tan_return'] += payouts['tansho'][ev]
                                if ev in payouts['fukusho']:
                                    stats['ev_fuku_hits'] += 1
                                    stats['ev_fuku_return'] += payouts['fukusho'][ev]

                    time.sleep(0.5)
                    my_bar.progress((i + 1) / len(races))
                
                # 計算
                tan_rate = (stats['honmei_tan_return'] / (stats['honmei_races'] * 100) * 100) if stats['honmei_races'] > 0 else 0
                fuku_rate = (stats['honmei_fuku_return'] / (stats['honmei_races'] * 100) * 100) if stats['honmei_races'] > 0 else 0
                uma_rate = (stats['umaren_return'] / stats['umaren_invest'] * 100) if stats['umaren_invest'] > 0 else 0
                wide_rate = (stats['wide_ana_return'] / stats['wide_ana_invest'] * 100) if stats['wide_ana_invest'] > 0 else 0
                ev_tan_rate = (stats['ev_tan_return'] / stats['ev_invest'] * 100) if stats['ev_invest'] > 0 else 0
                ev_fuku_rate = (stats['ev_fuku_return'] / stats['ev_invest'] * 100) if stats['ev_invest'] > 0 else 0
                shiba_rate = (stats['shiba_return'] / (stats['shiba_races'] * 100) * 100) if stats['shiba_races'] > 0 else 0
                dart_rate = (stats['dart_return'] / (stats['dart_races'] * 100) * 100) if stats['dart_races'] > 0 else 0
                exp_rate = (stats['exp_return'] / (stats['exp_races'] * 100) * 100) if stats['exp_races'] > 0 else 0
                new_rate = (stats['new_return'] / (stats['new_races'] * 100) * 100) if stats['new_races'] > 0 else 0

                # CSVセーブ
                csv_file = "ai_daily_history.csv"
                daily_data = pd.DataFrame([{
                    '日付': target_date.strftime('%Y/%m/%d'),
                    '本命単勝回収率': round(tan_rate, 1),
                    '本命複勝回収率': round(fuku_rate, 1),
                    '穴馬単勝回収率': round(ev_tan_rate, 1),
                    '穴馬複勝回収率': round(ev_fuku_rate, 1)
                }])
                if os.path.exists(csv_file):
                    existing_df = pd.read_csv(csv_file)
                    for col in ['本命単勝回収率', '本命複勝回収率', '穴馬単勝回収率', '穴馬複勝回収率']:
                        if col not in existing_df.columns: existing_df[col] = 0.0
                    existing_df = existing_df[existing_df['日付'] != target_date.strftime('%Y/%m/%d')]
                    updated_df = pd.concat([existing_df, daily_data])
                    updated_df.to_csv(csv_file, index=False)
                else: daily_data.to_csv(csv_file, index=False)

                # HF Dataset Hubにai_daily_history.csvを保存（再起動対策）
                if _HF_TOKEN and _HF_REPO_ID:
                    try:
                        from huggingface_hub import HfApi as _HfApi
                        _api = _HfApi(token=_HF_TOKEN)
                        _api.upload_file(
                            path_or_fileobj=csv_file,
                            path_in_repo="ai_daily_history.csv",
                            repo_id=_HF_REPO_ID,
                            repo_type="dataset",
                            commit_message=f"成績履歴更新 {target_date.strftime('%Y-%m-%d')}",
                            token=_HF_TOKEN,
                        )
                    except Exception: pass

                st.markdown("---")
                st.markdown(f"### 🏆 {target_date.strftime('%Y/%m/%d')} レース振り返りレポート")
                st.markdown(f"**対象レース数: {stats['honmei_races']} レース**")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.success("🎯 【本命(◎) 単勝・複勝成績】")
                    st.write(f"- **単勝 的中率**: {(stats['honmei_tan_hits'] / stats['honmei_races'] * 100):.1f}% ({stats['honmei_tan_hits']}R)")
                    st.write(f"- **単勝 回収率**: **{tan_rate:.1f}%**")
                    st.write(f"- **複勝 的中率**: {(stats['honmei_fuku_hits'] / stats['honmei_races'] * 100):.1f}% ({stats['honmei_fuku_hits']}R)")
                    st.write(f"- **複勝 回収率**: **{fuku_rate:.1f}%**")
                    st.markdown("---")
                    st.write(f"🌱 **芝** 回収率: {shiba_rate:.1f}% ({stats['shiba_races']}R)")
                    st.write(f"🏜️ **ダート** 回収率: {dart_rate:.1f}% ({stats['dart_races']}R)")
                    st.markdown("---")
                    st.write(f"📚 **既走馬のみ** 回収率: **{exp_rate:.1f}%** ({stats['exp_races']}R)")
                    st.write(f"🔰 **未出走混在** 回収率: **{new_rate:.1f}%** ({stats['new_races']}R)")
                    
                with col2:
                    st.info("🔗 【馬券シミュレーション】")
                    st.write(f"- **馬連流し (◎ → 2〜5番手へ4点)**")
                    st.write(f"  投資: ¥{stats['umaren_invest']:,} / 回収率: **{uma_rate:.1f}%** (的中 {stats['umaren_hits']}R)")
                    st.write(f"- **穴馬ワイド (◎ → 期待値特大の穴馬へ)**")
                    st.write(f"  該当: {stats['wide_ana_races']}R / 回収率: **{wide_rate:.1f}%** (的中 {stats['wide_ana_hits']}回)")
                    st.markdown("---")
                    st.warning("🔥 【上位5頭内 期待値1.5以上馬 ベタ買い】")
                    st.write(f"- 該当数: {int(stats['ev_invest']/100)} 頭")
                    st.write(f"- **単勝 回収率**: **{ev_tan_rate:.1f}%** (的中 {stats['ev_tan_hits']}頭)")
                    st.write(f"- **複勝 回収率**: **{ev_fuku_rate:.1f}%** (的中 {stats['ev_fuku_hits']}頭)")

elif action == "📈 長期成績分析":
    st.subheader("📈 長期成績分析 & 回収率ダッシュボード")
    import altair as alt
    csv_file = "ai_daily_history.csv"
    # ダッシュボード設定
    db_col1, db_col2 = st.columns([3,1])
    with db_col2:
        target_rate = st.number_input("目標回収率 (%)", 80, 200, 100, 5,
            help="このラインをグラフに表示します。損益分岐点=100%")
        focus_col = st.selectbox("重点分析指標",
            ['本命単勝回収率','本命複勝回収率','穴馬単勝回収率','穴馬複勝回収率'],
            help="累積損益グラフで比較する主指標")

    # 起動時にGitHubからai_daily_history.csvを取得（再起動でリセット防止）
    if not os.path.exists(csv_file) and _HF_TOKEN and _HF_REPO_ID:
        try:
            from huggingface_hub import hf_hub_download
            _hist_path = hf_hub_download(
                repo_id=_HF_REPO_ID, filename="ai_daily_history.csv",
                repo_type="dataset", token=_HF_TOKEN, cache_dir="/tmp/hf_cache"
            )
            import shutil
            shutil.copy(_hist_path, csv_file)
        except Exception: pass

    if not os.path.exists(csv_file):
        st.info("まだデータがありません。「1日の振り返り」を実行するとここにデータが蓄積されます。")
    else:
        history_df = pd.read_csv(csv_file)
        for col in ['本命単勝回収率','本命複勝回収率','穴馬単勝回収率','穴馬複勝回収率']:
            if col not in history_df.columns: history_df[col] = 0.0
        history_df['日付'] = pd.to_datetime(history_df['日付'], errors='coerce')
        history_df = history_df.dropna(subset=['日付']).sort_values('日付').reset_index(drop=True)

        if len(history_df) == 0:
            st.warning("有効なデータがありません。")
        else:
            # ── KPI サマリー ─────────────────────────────────
            n  = len(history_df)
            avg_tan  = history_df['本命単勝回収率'].mean()
            avg_fuku = history_df['本命複勝回収率'].mean()
            avg_ana  = history_df['穴馬単勝回収率'].mean()
            over100_tan  = (history_df['本命単勝回収率'] >= 100).sum()
            over100_fuku = (history_df['本命複勝回収率'] >= 100).sum()

            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("📅 集計日数",     f"{n}日")
            k2.metric("📈 本命単勝 平均", f"{avg_tan:.1f}%",
                      delta=f"{avg_tan-100:+.1f}%", delta_color="normal")
            k3.metric("📊 本命複勝 平均", f"{avg_fuku:.1f}%",
                      delta=f"{avg_fuku-100:+.1f}%", delta_color="normal")
            k4.metric("🔥 穴馬単勝 平均", f"{avg_ana:.1f}%",
                      delta=f"{avg_ana-100:+.1f}%", delta_color="normal")
            k5.metric("✅ 単勝100%超え日", f"{over100_tan}日 / {n}日",
                      f"{over100_tan/n*100:.0f}%")

            st.markdown("---")

            # ── 移動平均オプション ────────────────────────────
            if n <= 2:
                ma_window = 1
                if n == 2:
                    st.caption("データが2件のため移動平均は適用されません（3件以上で有効）。")
                else:
                    st.caption("データが1件のため移動平均は適用されません。")
            else:
                _ma_max = min(10, n - 1)          # 最大: min(10, n-1)、n>=3なら最低2
                _ma_def = min(3, _ma_max)          # default: _ma_maxを超えない
                ma_window = st.slider("移動平均ウィンドウ (日)", 1, _ma_max, _ma_def, 1)

            # ── 折れ線グラフ ─────────────────────────────────
            plot_cols = ['本命単勝回収率','本命複勝回収率','穴馬単勝回収率','穴馬複勝回収率']
            history_df['日付_str'] = history_df['日付'].dt.strftime('%Y/%m/%d')

            # 移動平均を計算
            for col in plot_cols:
                history_df[f'{col}_MA'] = history_df[col].rolling(ma_window, min_periods=1).mean()

            melted = history_df.melt(
                '日付_str',
                value_vars=[f'{c}_MA' for c in plot_cols],
                var_name='指標', value_name='回収率(%)'
            )
            melted['指標'] = melted['指標'].str.replace('_MA','')

            rule100 = alt.Chart(pd.DataFrame({'y':[100]})).mark_rule(
                color='gray', strokeDash=[4,4], opacity=0.6
            ).encode(y='y:Q')

            line = alt.Chart(melted).mark_line(point=True).encode(
                x=alt.X('日付_str:N', sort=None, title='日付'),
                y=alt.Y('回収率(%):Q', title='回収率 (%)'),
                color=alt.Color('指標:N', legend=alt.Legend(orient='bottom')),
                tooltip=['日付_str','指標','回収率(%)']
            ).properties(height=320)
            st.altair_chart(line + rule100, use_container_width=True)
            st.caption(f"灰色破線 = 損益分岐点 / {ma_window}日移動平均を表示中 / 目標: {target_rate}%")

            st.markdown("---")
            st.markdown("#### 📋 日別詳細テーブル")

            # 値でセルを色付け
            def color_rate(val):
                try:
                    v = float(val)
                    if v >= 120: return 'background-color:rgba(255,75,75,0.2);color:#c00;font-weight:bold'
                    if v >= 100: return 'background-color:rgba(255,165,0,0.15);color:#a60'
                    if v < 70:  return 'background-color:rgba(100,100,100,0.08);color:#999'
                except: pass
                return ''

            show_table = history_df[['日付_str'] + plot_cols].copy()
            show_table = show_table.rename(columns={'日付_str':'日付'}).sort_values('日付', ascending=False)
            st.dataframe(
                show_table.style.applymap(color_rate, subset=plot_cols)
                          .format({c:'{:.1f}%' for c in plot_cols}),
                use_container_width=True, hide_index=True
            )

            # ── 月別集計 ─────────────────────────────────────
            if len(history_df) >= 2:
                st.markdown("#### 📅 月別集計")
                history_df['年月'] = history_df['日付'].dt.to_period('M').astype(str)
                monthly = history_df.groupby('年月')[plot_cols].mean().round(1)
                monthly['対象日数'] = history_df.groupby('年月').size()
                st.dataframe(
                    monthly.style.applymap(color_rate, subset=plot_cols)
                           .format({c:'{:.1f}%' for c in plot_cols}),
                    use_container_width=True
                )

            # ── 累積損益シミュレーション ────────────────────
            st.markdown("---")
            st.markdown("#### 📈 累積損益シミュレーション (4戦略比較)")
            sim_unit = st.number_input("1レースあたりの賭け金 (円)", 100, 10000, 100, 100, key="sim_unit_hist")
            for col in plot_cols:
                history_df[f'損益_{col}'] = (history_df[col] - 100) / 100 * sim_unit
                history_df[f'累積_{col}'] = history_df[f'損益_{col}'].cumsum()
            history_df['損益']   = history_df[f'損益_{focus_col}']
            history_df['累積損益'] = history_df[f'累積_{focus_col}']
            history_df['日付_str2'] = history_df['日付'].dt.strftime('%m/%d')

            cum_chart = alt.Chart(history_df).mark_line(
                color='#4B8BFF', strokeWidth=2, point=alt.OverlayMarkDef(color='#4B8BFF', size=50)
            ).encode(
                x=alt.X('日付_str2:N', sort=None, title='日付', axis=alt.Axis(labelAngle=-45)),
                y=alt.Y('累積損益:Q', title='累積損益 (円)'),
                tooltip=['日付_str2', alt.Tooltip('累積損益:Q', format=',')]
            ).properties(height=220)
            # 目標回収率ライン（損益分岐点 + ユーザー設定ライン）
            rule_data = pd.DataFrame({'y': [0], 'label': ['損益0円']})
            zero_rule = alt.Chart(rule_data).mark_rule(
                color='gray', strokeDash=[4,4], opacity=0.5
            ).encode(y='y:Q')
            st.altair_chart(cum_chart + zero_rule, use_container_width=True)

            total_profit  = history_df['累積損益'].iloc[-1] if len(history_df) > 0 else 0
            total_invest_h = len(history_df) * sim_unit
            cp1, cp2, cp3 = st.columns(3)
            cp1.metric("総投資額",   f"¥{total_invest_h:,.0f}")
            cp2.metric("累積損益",   f"¥{total_profit:,.0f}",
                       delta=f"{'+' if total_profit>=0 else ''}{total_profit:.0f}円")
            cp3.metric("総合回収率", f"{(total_profit/total_invest_h+1)*100:.1f}%" if total_invest_h>0 else "N/A")

            # ── 条件別回収率ヒートマップ ────────────────────
            if len(history_df) >= 3:
                st.markdown("---")
                st.markdown("#### 🗓️ 週別回収率ヒートマップ (曜日×週)")
                history_df['曜日'] = history_df['日付'].dt.day_name().map({
                    'Monday':'月','Tuesday':'火','Wednesday':'水','Thursday':'木',
                    'Friday':'金','Saturday':'土','Sunday':'日'
                })
                history_df['週'] = history_df['日付'].dt.strftime('%m/%d週')
                hm_data = history_df.groupby(['週','曜日'])['本命単勝回収率'].mean().reset_index()
                hm_data.columns = ['週','曜日','回収率']
                heat = alt.Chart(hm_data).mark_rect(
                    cornerRadius=3, stroke='white', strokeWidth=1
                ).encode(
                    x=alt.X('週:N', title=''),
                    y=alt.Y('曜日:N', sort=['月','火','水','木','金','土','日'], title=''),
                    color=alt.Color('回収率:Q',
                        scale=alt.Scale(domain=[50,150], scheme='redblue'),
                        legend=alt.Legend(title='回収率(%)')),
                    tooltip=['週','曜日', alt.Tooltip('回収率:Q', format='.1f')]
                ).properties(height=180)
                st.altair_chart(heat, use_container_width=True)
                st.caption("赤=高回収率 / 青=低回収率。開催日(主に土日)のみ反映")

# ==========================================
# 🌟 性能試験 (バックテスト) 機能
# ==========================================
elif action == "🧪 性能試験 (バックテスト)":
    st.subheader("🧪 性能試験 (バックテスト)")
    with st.expander("ℹ️ バックテスト精度について（必読）", expanded=False):
        st.info("""**バックテスト結果の見方**

過去2週間以内のバックテストは信頼性が高いです。それより古い日付は参考値として扱ってください。

モデルは「学習データの最終日」までの全期間で学習されています。
古い日付でのバックテストでは、直近2週間より前の検証で一部の馬情報（コース適性・直近成績）に
「その時点では知り得なかった未来情報」が混入する場合があります。

今回のバージョンからリーク防止処理を追加済みです（最新_日付がバックテスト日以降の馬は前走情報をNaNマスク）。
それでも学習モデル自体は全期間データで学習されているため、完全な隔離ではありません。""")

    # ── 設定エリア ────────────────────────────────────────
    col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
    with col_cfg1:
        test_date = st.date_input("テストする日付", datetime.date.today() - datetime.timedelta(days=3))
    with col_cfg2:
        ev_threshold = st.slider("期待値フィルター", 1.0, 3.0, 1.5, 0.1,
                                  help="この値以上の期待値の馬だけをベット対象にします")
    with col_cfg3:
        bet_unit = st.number_input("1点あたりの賭け金 (円)", 100, 10000, 100, 100)

    if st.button("🔥 バックテスト実行！", type="primary"):
        with st.spinner(f'全レースを推論・集計中...'):
            test_races = get_todays_races(test_date.strftime('%Y%m%d'))
            if not test_races:
                st.error("レースが見つかりません。")
            else:
                my_bar = st.progress(0, text="集計中...")
                results_for_txt = []
                analysis_records = []  # レースごとの詳細記録

                for i, r in enumerate(test_races):
                    with st.expander(f"🏁 {r['place']} {r['num']}R"):
                        res_df, topics, reco, pace_text, conf_text, track_type, place, dist, err_log = run_real_prediction(r['id'], test_date.strftime('%Y-%m-%d'), skip_live_scrape=True)
                        t_dict, f_dict = get_payouts(r['id'])

                        if res_df is not None:
                            display_result(res_df, topics, reco, pace_text, conf_text, show_change_table=False)
                            results_for_txt.append({'date': test_date.strftime('%Y年%m月%d日'), 'place': place, 'num': r['num'], 'track': track_type, 'dist': dist, 'pace': pace_text, 'confidence': conf_text, 'df': res_df, 'topics': topics, 'reco': reco})

                            if not t_dict:
                                st.warning("⚠️ 払い戻しデータが取得できませんでした（予想は表示済み）")
                            else:
                                try:
                                    d = int(dist)
                                    if d <= 1400: d_cat = "短距離(〜1400m)"
                                    elif d <= 1600: d_cat = "マイル(1600m)"
                                    elif d <= 2200: d_cat = "中距離(1800〜2200m)"
                                    else: d_cat = "長距離(2400m〜)"
                                except: d_cat = "不明"

                                honmei = res_df.iloc[0]['馬番']
                                honmei_tan = t_dict.get(honmei, 0)
                                honmei_fuku = f_dict.get(honmei, 0)

                                ev_targets = res_df[(res_df.index < 5) & (res_df['期待値'] >= ev_threshold)]
                                for _, horse in ev_targets.iterrows():
                                    ret_t = t_dict.get(horse['馬番'], 0)
                                    ret_f = f_dict.get(horse['馬番'], 0)
                                    analysis_records.append({
                                        'レース': f"{place}{r['num']}R",
                                        '競馬場': place,
                                        '芝/ダート': track_type,
                                        '距離帯': d_cat,
                                        '馬名': horse['馬名'],
                                        '印': horse['印'],
                                        'AI勝率': horse['勝率(AI予測)'],
                                        '期待値': horse['期待値'],
                                        '単勝オッズ': horse['単勝オッズ'],
                                        '投資額': bet_unit,
                                        '単勝回収': ret_t * bet_unit // 100,
                                        '複勝回収': ret_f * bet_unit // 100,
                                        '本命単勝払戻': honmei_tan,
                                        '本命複勝払戻': honmei_fuku,
                                    })
                        else:
                            if err_log: display_error_log(err_log)
                            else: st.warning(f"⚠️ {r['place']} {r['num']}R: 取得失敗")
                    time.sleep(1.0)
                    my_bar.progress((i + 1) / len(test_races))

                # ── 集計レポート ─────────────────────────────────────
                st.markdown("---")
                st.markdown(f"### 🏆 {test_date.strftime('%Y/%m/%d')} バックテスト集計レポート")

                if not analysis_records:
                    st.warning("期待値フィルターに合致する馬がいませんでした。フィルター値を下げてみてください。")
                else:
                    import altair as alt
                    df_ana = pd.DataFrame(analysis_records)
                    total_invest   = df_ana['投資額'].sum()
                    total_tan_ret  = df_ana['単勝回収'].sum()
                    total_fuku_ret = df_ana['複勝回収'].sum()
                    tan_hits  = (df_ana['単勝回収'] > 0).sum()
                    fuku_hits = (df_ana['複勝回収'] > 0).sum()
                    tan_rate  = total_tan_ret  / total_invest * 100 if total_invest > 0 else 0
                    fuku_rate = total_fuku_ret / total_invest * 100 if total_invest > 0 else 0

                    # KPIカード
                    k1, k2, k3, k4, k5 = st.columns(5)
                    k1.metric("🎯 対象ベット数", f"{len(df_ana)}件",
                              help=f"期待値{ev_threshold}以上 × 上位5頭以内")
                    k2.metric("💰 総投資額", f"¥{total_invest:,}")
                    k3.metric("📈 単勝回収率",
                              f"{tan_rate:.1f}%",
                              f"{tan_rate-100:+.1f}%",
                              delta_color="normal")
                    k4.metric("📊 複勝回収率",
                              f"{fuku_rate:.1f}%",
                              f"{fuku_rate-100:+.1f}%",
                              delta_color="normal")
                    k5.metric("✅ 的中数",
                              f"単:{tan_hits} / 複:{fuku_hits}",
                              f"的中率 {tan_hits/len(df_ana)*100:.0f}% / {fuku_hits/len(df_ana)*100:.0f}%")

                    st.markdown("---")

                    # 損益推移グラフ
                    df_ana['損益(単)']  = df_ana['単勝回収'] - df_ana['投資額']
                    df_ana['損益(複)']  = df_ana['複勝回収'] - df_ana['投資額']
                    df_ana['累計損益(単)'] = df_ana['損益(単)'].cumsum()
                    df_ana['累計損益(複)'] = df_ana['損益(複)'].cumsum()
                    df_ana['番号'] = range(1, len(df_ana)+1)

                    st.markdown("#### 📈 累積損益推移")
                    melted = df_ana.melt('番号', value_vars=['累計損益(単)','累計損益(複)'], var_name='戦略', value_name='累計損益')
                    rule0 = alt.Chart(pd.DataFrame({'y':[0]})).mark_rule(color='gray', strokeDash=[4,4]).encode(y='y:Q')
                    line = alt.Chart(melted).mark_line(point=True).encode(
                        x=alt.X('番号:Q', title='ベット番号'),
                        y=alt.Y('累計損益:Q', title='累計損益 (円)'),
                        color='戦略:N',
                        tooltip=['番号','戦略','累計損益']
                    ).properties(height=250)
                    st.altair_chart(line + rule0, use_container_width=True)

                    st.markdown("#### 🔍 条件別成績")

                    def make_seg(df, col):
                        g = df.groupby(col).agg(
                            件数=('投資額','count'),
                            投資=('投資額','sum'),
                            単勝回収=('単勝回収','sum'),
                            複勝回収=('複勝回収','sum'),
                        ).reset_index()
                        g['単勝回収率(%)'] = (g['単勝回収']/g['投資']*100).round(1)
                        g['複勝回収率(%)'] = (g['複勝回収']/g['投資']*100).round(1)
                        g['単勝損益']=g['単勝回収']-g['投資']
                        return g[[col,'件数','投資','単勝回収率(%)','複勝回収率(%)','単勝損益']].sort_values('単勝回収率(%)',ascending=False)

                    def style_seg(df):
                        def color_row(row):
                            if row['単勝回収率(%)'] >= 120: return ['background-color:rgba(255,75,75,0.15)']*len(row)
                            if row['単勝回収率(%)'] >= 100: return ['background-color:rgba(255,165,0,0.1)']*len(row)
                            return ['']*len(row)
                        return df.style.apply(color_row,axis=1).format({'単勝回収率(%)':'{}%','複勝回収率(%)':'{}%','投資':'¥{:,}','単勝損益':'¥{:,}'})

                    bt1, bt2, bt3, bt4 = st.tabs(["⛰️ 芝/ダート", "🏟️ 競馬場", "📏 距離帯", "📋 全ベット一覧"])
                    with bt1: st.dataframe(style_seg(make_seg(df_ana,'芝/ダート')), use_container_width=True, hide_index=True)
                    with bt2: st.dataframe(style_seg(make_seg(df_ana,'競馬場')), use_container_width=True, hide_index=True)
                    with bt3:
                        sort_order = ["短距離(〜1400m)","マイル(1600m)","中距離(1800〜2200m)","長距離(2400m〜)","不明"]
                        df_d = make_seg(df_ana,'距離帯')
                        df_d = df_d.set_index('距離帯').reindex([x for x in sort_order if x in df_d['距離帯'].values]).reset_index()
                        st.dataframe(style_seg(df_d), use_container_width=True, hide_index=True)
                    with bt4:
                        show_detail = df_ana[['レース','印','馬名','AI勝率','期待値','単勝オッズ','投資額','単勝回収','複勝回収']].copy()
                        show_detail['AI勝率'] = (show_detail['AI勝率']*100).round(1).astype(str)+'%'
                        show_detail['期待値'] = show_detail['期待値'].round(2)
                        show_detail['結果'] = show_detail['単勝回収'].apply(lambda x: '✅ 的中' if x>0 else '❌')
                        def color_result(row):
                            if row['単勝回収'] > 0: return ['background-color:rgba(75,255,75,0.1)']*len(row)
                            return ['']*len(row)
                        st.dataframe(show_detail.style.apply(color_result,axis=1)
                                     .format({'期待値':'{:.2f}','単勝オッズ':'{:.1f}','投資額':'¥{:,}','単勝回収':'¥{:,}','複勝回収':'¥{:,}'}),
                                     use_container_width=True, hide_index=True)

                if results_for_txt:
                    _db1, _db2 = st.columns(2)
                    _db1.download_button("📥 バックテスト結果 (.txt)", data=generate_txt_report(results_for_txt), file_name=f"keiba_backtest_{test_date.strftime('%Y%m%d')}.txt", mime="text/plain")
                    _html_b = generate_pdf_report(results_for_txt)
                    if _html_b: _db2.download_button("🌐 バックテスト結果 (.html)", data=_html_b, file_name=f"keiba_backtest_{test_date.strftime('%Y%m%d')}.html", mime="text/html")

# 🌟 新機能: 一口馬主・推し馬向け 成長記録グラフ

# ==========================================
# ② ウォークフォワード検証 (モデル精度の安定性確認)
# ==========================================
elif action == "📊 モデル検証 (ウォークフォワード)":
    st.subheader("📊 モデル検証 - ウォークフォワード分析")
    st.info("学習データを時系列に3分割し、各期間でのAI精度を検証します。精度が安定していれば過学習していない証拠です。")

    if st.button("🔬 ウォークフォワード検証を実行", type="primary"):
        with st.spinner("3期間分の検証を実行中... (数分かかります)"):
            try:
                df_wf = pd.read_csv('learning_data_perfect_tier.zip', compression='zip', dtype=str)
                df_wf['日付'] = pd.to_datetime(df_wf['日付'], format='mixed', errors='coerce')
                df_wf = df_wf.dropna(subset=['日付'])
                for col in ['着順', '単勝', '人気', '斤量', '距離', '上り', '枠番', '馬番']:
                    df_wf[col] = pd.to_numeric(df_wf[col], errors='coerce')
                df_wf['馬券内'] = (df_wf['着順'] <= 3).astype(int)
                df_wf = df_wf.dropna(subset=['着順', '単勝']).sort_values('日付').reset_index(drop=True)

                min_date = df_wf['日付'].min()
                max_date = df_wf['日付'].max()
                total_days = (max_date - min_date).days
                fold_days = total_days // 3

                wf_results = []
                import altair as alt

                for fold in range(3):
                    fold_start = min_date + pd.Timedelta(days=fold * fold_days)
                    fold_mid   = fold_start + pd.Timedelta(days=int(fold_days * 0.7))
                    fold_end   = fold_start + pd.Timedelta(days=fold_days) if fold < 2 else max_date

                    tr = df_wf[(df_wf['日付'] >= fold_start) & (df_wf['日付'] < fold_mid)].copy()
                    te = df_wf[(df_wf['日付'] >= fold_mid) & (df_wf['日付'] < fold_end)].copy()

                    if len(tr) < 100 or len(te) < 10:
                        continue

                    # TE計算
                    gm = tr['馬券内'].mean()
                    for col in ['騎手', '調教師', '父']:
                        if col in tr.columns:
                            ted = tr.groupby(col)['馬券内'].mean().to_dict()
                            tr[f'{col}_TE'] = tr[col].map(ted).fillna(gm)
                            te[f'{col}_TE'] = te[col].map(ted).fillna(gm)

                    use_cols = [c for c in ['枠番', '馬番', '距離', '斤量', '人気',
                                            '騎手_TE', '調教師_TE', '父_TE'] if c in tr.columns]
                    if not use_cols: continue

                    for col in use_cols:
                        tr[col] = pd.to_numeric(tr[col], errors='coerce')
                        te[col] = pd.to_numeric(te[col], errors='coerce')

                    tr = tr.dropna(subset=use_cols)
                    te = te.dropna(subset=use_cols)

                    tr_groups = tr.groupby('レースID', sort=False).size().values if 'レースID' in tr.columns else np.ones(len(tr), dtype=int)
                    te_groups = te.groupby('レースID', sort=False).size().values if 'レースID' in te.columns else np.ones(len(te), dtype=int)

                    m_wf = lgb.LGBMRanker(n_estimators=200, learning_rate=0.05,
                                          num_leaves=31, random_state=42)
                    m_wf.fit(tr[use_cols], tr['馬券内'], group=tr_groups)

                    te['score'] = m_wf.predict(te[use_cols])
                    te['exp_s'] = np.exp(te['score'] - te.groupby('レースID')['score'].transform('max')) if 'レースID' in te.columns else np.exp(te['score'])
                    te['ai_win'] = te['exp_s'] / te.groupby('レースID')['exp_s'].transform('sum') if 'レースID' in te.columns else te['exp_s']

                    top1 = te.sort_values(['レースID', 'ai_win'], ascending=[True, False]).groupby('レースID').head(1) if 'レースID' in te.columns else te.sort_values('ai_win', ascending=False).head(len(te)//10)
                    hits = top1[pd.to_numeric(top1['着順'], errors='coerce') == 1]
                    invest = len(top1) * 100
                    ret = (pd.to_numeric(hits['単勝'], errors='coerce') * 100).sum()
                    rr = (ret / invest * 100) if invest > 0 else 0
                    hit_rate = len(hits) / len(top1) * 100 if len(top1) > 0 else 0

                    wf_results.append({
                        '期間': f"Fold {fold+1}",
                        '学習期間': f"{fold_start.strftime('%Y/%m')} 〜 {fold_mid.strftime('%Y/%m')}",
                        '検証期間': f"{fold_mid.strftime('%Y/%m')} 〜 {fold_end.strftime('%Y/%m')}",
                        '検証レース数': len(top1),
                        '本命的中率(%)': round(hit_rate, 1),
                        '単勝回収率(%)': round(rr, 1),
                    })

                if wf_results:
                    df_wfr = pd.DataFrame(wf_results)
                    st.markdown("#### 検証結果")

                    c1, c2, c3 = st.columns(3)
                    for i, row in df_wfr.iterrows():
                        col = [c1, c2, c3][i]
                        color = "🟢" if row['単勝回収率(%)'] >= 100 else "🔴"
                        col.metric(f"{color} {row['期間']}", f"回収率 {row['単勝回収率(%)']}%",
                                   f"的中率 {row['本命的中率(%)']}%")

                    st.dataframe(df_wfr, use_container_width=True, hide_index=True)

                    chart_data = df_wfr[['期間', '単勝回収率(%)', '本命的中率(%)']].melt('期間', var_name='指標', value_name='値')
                    rule = alt.Chart(pd.DataFrame({'y': [100]})).mark_rule(color='red', strokeDash=[5,5]).encode(y='y:Q')
                    bars = alt.Chart(chart_data).mark_bar(opacity=0.8).encode(
                        x=alt.X('期間:N'),
                        y=alt.Y('値:Q'),
                        color='指標:N',
                        tooltip=['期間', '指標', '値']
                    )
                    st.altair_chart((bars + rule).properties(height=300), use_container_width=True)

                    avg_rr = df_wfr['単勝回収率(%)'].mean()
                    std_rr = df_wfr['単勝回収率(%)'].std()
                    if std_rr < 20:
                        st.success(f"✅ 安定性: 良好 (3期間の回収率標準偏差 = {std_rr:.1f}%)")
                    else:
                        st.warning(f"⚠️ 安定性: やや不安定 (標準偏差 = {std_rr:.1f}%、特定期間に偏りあり)")
                    st.metric("3期間平均 単勝回収率", f"{avg_rr:.1f}%")
                else:
                    st.error("検証データが不足しています。")

            except Exception as e:
                st.error(f"ウォークフォワード検証エラー: {e}")
                import traceback
                st.code(traceback.format_exc())

# ==========================================
# ⑤ 騎手・調教師フォーム分析
# ==========================================
elif action == "🏇 騎手・調教師フォーム分析":
    st.subheader("🏇 騎手・調教師 近況フォーム分析")
    st.info("学習データから直近の騎手・調教師の好調/不調を分析します。")

    try:
        df_form = pd.read_csv('learning_data_perfect_tier.zip', compression='zip', dtype=str)
        df_form['日付'] = pd.to_datetime(df_form['日付'], format='mixed', errors='coerce')
        df_form = df_form.dropna(subset=['日付'])
        df_form['着順'] = pd.to_numeric(df_form['着順'], errors='coerce')
        df_form['単勝'] = pd.to_numeric(df_form['単勝'], errors='coerce')
        df_form['人気'] = pd.to_numeric(df_form['人気'], errors='coerce')

        max_dt = df_form['日付'].max()
        period_days = st.slider("分析期間 (日)", 30, 180, 90, 30)
        since_dt = max_dt - pd.Timedelta(days=period_days)
        df_recent = df_form[df_form['日付'] >= since_dt].copy()
        df_recent['勝ち'] = (df_recent['着順'] == 1).astype(int)
        df_recent['複勝'] = (df_recent['着順'] <= 3).astype(int)
        df_recent['人気馬逃げ'] = ((df_recent['人気'] <= 3) & (df_recent['着順'] > 5)).astype(int)
        df_recent['穴馬激走'] = ((df_recent['人気'] >= 7) & (df_recent['着順'] <= 3)).astype(int)

        top_n = st.slider("表示件数", 10, 50, 20, 5)

        tab_j, tab_t = st.tabs(["🏅 騎手", "🏠 調教師"])

        def build_form_df(df, col):
            g = df.groupby(col).agg(
                出走数=('着順', 'count'),
                勝利数=('勝ち', 'sum'),
                複勝数=('複勝', 'sum'),
                人気馬逃げ数=('人気馬逃げ', 'sum'),
                穴馬激走数=('穴馬激走', 'sum'),
                単勝回収額=('単勝', lambda x: (df.loc[x.index][df.loc[x.index]['勝ち']==1]['単勝'] * 100).sum()),
            ).reset_index()
            g = g[g['出走数'] >= 10]
            g['勝率(%)']   = (g['勝利数']  / g['出走数'] * 100).round(1)
            g['複勝率(%)'] = (g['複勝数']  / g['出走数'] * 100).round(1)
            g['単勝回収率(%)'] = (g['単勝回収額'] / (g['出走数'] * 100) * 100).round(1)
            g['フォームスコア'] = (g['勝率(%)'] * 2 + g['複勝率(%)'] + g['単勝回収率(%)'] / 10).round(1)
            return g.sort_values('フォームスコア', ascending=False).head(top_n)

        def style_form(df):
            def color_row(row):
                if row['単勝回収率(%)'] >= 120:
                    return ['background-color: rgba(255,75,75,0.15)'] * len(row)
                elif row['単勝回収率(%)'] >= 100:
                    return ['background-color: rgba(255,165,0,0.1)'] * len(row)
                return [''] * len(row)
            return df.style.apply(color_row, axis=1).format({
                '勝率(%)': '{:.1f}%', '複勝率(%)': '{:.1f}%',
                '単勝回収率(%)': '{:.1f}%', 'フォームスコア': '{:.1f}'
            })

        with tab_j:
            if '騎手' in df_recent.columns:
                df_j = build_form_df(df_recent, '騎手')
                st.caption(f"直近{period_days}日の成績（出走10回以上、フォームスコア順）赤＝回収率120%超、橙＝100%超")
                st.dataframe(style_form(df_j), use_container_width=True, hide_index=True)
            else:
                st.warning("騎手データが見つかりません")

        with tab_t:
            if '調教師' in df_recent.columns:
                df_t = build_form_df(df_recent, '調教師')
                st.caption(f"直近{period_days}日の成績（出走10回以上、フォームスコア順）赤＝回収率120%超、橙＝100%超")
                st.dataframe(style_form(df_t), use_container_width=True, hide_index=True)
            else:
                st.warning("調教師データが見つかりません")

    except Exception as e:
        st.error(f"フォーム分析エラー: {e}")


# ==========================================
# 馬券メモ管理
# ==========================================
elif action == "📝 馬券メモ管理":
    st.subheader("📝 馬券メモ管理")
    st.caption("馬に紐づいたメモを記録します。次回出走時にハイライト表示されます。")

    MEMO_FILE = "horse_memos.json"

    def load_memos():
        # 1. まずローカルファイルを確認
        if os.path.exists(MEMO_FILE):
            try:
                with open(MEMO_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except: pass
        # 2. GitHubから取得を試みる（ローカルにない場合）
        gh_token = os.environ.get("GITHUB_TOKEN", "")
        gh_repo  = os.environ.get("GITHUB_REPO", "")   # 例: "username/keiba-ebye"
        if gh_token and gh_repo:
            try:
                api_url = f"https://api.github.com/repos/{gh_repo}/contents/{MEMO_FILE}"
                resp = requests.get(api_url,
                    headers={"Authorization": f"token {gh_token}", "Accept": "application/vnd.github.v3+json"},
                    timeout=5)
                if resp.status_code == 200:
                    import base64
                    data = json.loads(base64.b64decode(resp.json()["content"]).decode("utf-8"))
                    # ローカルにキャッシュ
                    with open(MEMO_FILE, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                    return data
            except: pass
        return {}

    def save_memos(memos):
        # ローカル保存
        with open(MEMO_FILE, "w", encoding="utf-8") as f:
            json.dump(memos, f, ensure_ascii=False, indent=2)
        # GitHub自動コミット（Secrets設定がある場合）
        gh_token = os.environ.get("GITHUB_TOKEN", "")
        gh_repo  = os.environ.get("GITHUB_REPO", "")
        if not gh_token or not gh_repo:
            return  # Secrets未設定なら無視
        try:
            import base64
            api_url  = f"https://api.github.com/repos/{gh_repo}/contents/{MEMO_FILE}"
            headers  = {"Authorization": f"token {gh_token}", "Accept": "application/vnd.github.v3+json"}
            # 既存ファイルのSHAを取得（更新時に必要）
            get_resp = requests.get(api_url, headers=headers, timeout=5)
            sha = get_resp.json().get("sha", "") if get_resp.status_code == 200 else ""
            content_b64 = base64.b64encode(
                json.dumps(memos, ensure_ascii=False, indent=2).encode("utf-8")
            ).decode("utf-8")
            payload = {
                "message": f"📝 馬券メモ更新 {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
                "content": content_b64,
            }
            if sha: payload["sha"] = sha
            put_resp = requests.put(api_url, headers=headers, json=payload, timeout=8)
            if put_resp.status_code in (200, 201):
                st.toast("✅ GitHubにメモを自動保存しました", icon="💾")
            else:
                st.toast(f"⚠️ GitHub保存失敗(status={put_resp.status_code})", icon="⚠️")
        except Exception as _e:
            st.toast(f"⚠️ GitHub自動コミットエラー: {_e}", icon="⚠️")

    memos = load_memos()

    col_l, col_r = st.columns([2, 1])

    with col_r:
        st.markdown("#### ✏️ メモを追加・編集")
        memo_horse = st.text_input("馬名", placeholder="例: ドウデュース")
        memo_tag = st.selectbox("タグ", [
            "🔴 出遅れ", "🟠 不利あり", "🟡 内有利で損",
            "🟢 好走の手応え", "🔵 距離が長かった", "⚫ 馬場が合わなかった",
            "⚪ その他",
        ])
        memo_text = st.text_area("メモ内容", placeholder="例: 3角で大外に張られ位置取り悪化。次走は巻き返し期待", height=100)
        memo_date = st.date_input("レース日", datetime.date.today())

        if st.button("💾 メモを保存", type="primary") and memo_horse.strip():
            if memo_horse not in memos:
                memos[memo_horse] = []
            memos[memo_horse].append({
                "日付": memo_date.strftime("%Y/%m/%d"),
                "タグ": memo_tag,
                "メモ": memo_text.strip(),
            })
            save_memos(memos)
            st.success(f"✅ {memo_horse} のメモを保存しました！")
            st.rerun()

    with col_l:
        st.markdown("#### 📋 登録済みメモ一覧")
        if not memos:
            st.info("まだメモがありません。右の欄からメモを追加してください。")
        else:
            search_horse = st.text_input("🔍 馬名で検索", placeholder="馬名を入力...")
            display_memos = {
                k: v for k, v in memos.items()
                if not search_horse or search_horse.lower() in k.lower()
            }
            if not display_memos:
                st.warning(f"'{search_horse}' に一致する馬のメモはありません。")
            else:
                for horse_name, memo_list in sorted(display_memos.items()):
                    with st.expander(f"🐴 {horse_name}  ({len(memo_list)}件)", expanded=False):
                        for i, memo in enumerate(sorted(memo_list, key=lambda x: x["日付"], reverse=True)):
                            tag_color = {
                                "🔴": "rgba(255,75,75,0.1)",
                                "🟠": "rgba(255,165,0,0.1)",
                                "🟡": "rgba(255,230,0,0.1)",
                                "🟢": "rgba(75,200,75,0.1)",
                                "🔵": "rgba(75,75,255,0.1)",
                                "⚫": "rgba(100,100,100,0.1)",
                            }.get(memo["タグ"][0], "rgba(200,200,200,0.05)")
                            st.markdown(
                                f'<div style="padding:8px;margin:4px 0;border-radius:6px;'
                                f'background:{tag_color};border-left:3px solid #ccc;">'
                                f'<b>{memo["日付"]}</b> {memo["タグ"]}<br>'
                                f'{memo["メモ"] or "(メモなし)"}</div>',
                                unsafe_allow_html=True
                            )
                            col_del = st.columns([4, 1])[1]
                            if col_del.button("🗑️", key=f"del_{horse_name}_{i}", help="削除"):
                                memos[horse_name].pop(i)
                                if not memos[horse_name]:
                                    del memos[horse_name]
                                save_memos(memos)
                                st.rerun()

    # 次回出走時ハイライト用: メモ対象馬名リストをセッションに保存
    if memos:
        st.session_state['memo_horses'] = list(memos.keys())

elif action == "🐴 愛馬の成長記録":
    st.subheader("🐴 愛馬の成長記録 & AIスカウティング")
    import altair as alt

    col_inp1, col_inp2 = st.columns([3, 1])
    with col_inp1:
        horse_name = st.text_input("🔍 馬名を入力", placeholder="例: ドウデュース、リバティアイランド")
    with col_inp2:
        show_n = st.number_input("最近N戦を表示", 5, 50, 20, 5)

    if st.button("📊 成長記録を表示", type="primary") and horse_name:
        with st.spinner(f"{horse_name} のデータを検索中..."):
            try:
                data_file = 'learning_data_perfect_tier.zip'
                if not os.path.exists(data_file):
                    st.error(f"データベースファイル ({data_file}) が見つかりません。")
                else:
                    df_hist = pd.read_csv(data_file, compression='zip', dtype=str)
                    df_hist['日付'] = pd.to_datetime(df_hist['日付'], format='mixed', errors='coerce')
                    df_horse = df_hist[df_hist['馬名'] == horse_name].copy()

                    if df_horse.empty:
                        # 部分一致検索
                        candidates = df_hist[df_hist['馬名'].str.contains(horse_name, na=False)]['馬名'].unique()
                        if len(candidates) > 0:
                            st.warning(f"「{horse_name}」は見つかりませんでした。似た名前: {', '.join(candidates[:5])}")
                        else:
                            st.warning(f"「{horse_name}」は見つかりませんでした。")
                    else:
                        df_horse = df_horse.sort_values('日付').dropna(subset=['日付'])
                        df_horse = df_horse.tail(show_n).copy()

                        # 数値変換
                        for col in ['着順','人気','単勝','上り','当日馬体重','馬体重増減',
                                    '補正タイム偏差','タイム差','距離','斤量','枠番','馬番']:
                            if col in df_horse.columns:
                                df_horse[col] = pd.to_numeric(df_horse[col], errors='coerce')

                        # タイム指数計算
                        if '補正タイム偏差' in df_horse.columns:
                            df_horse['タイム指数'] = (50 - df_horse['補正タイム偏差'] * 10).round(1)

                        df_horse['日付_str'] = df_horse['日付'].dt.strftime('%Y/%m/%d')

                        st.success(f"✅ {len(df_horse)}戦分のデータ（直近{show_n}戦）を表示中")

                        # ── サマリーカード ──────────────────────────────
                        total_n = len(df_horse)
                        wins    = (df_horse['着順'] == 1).sum()
                        top3    = (df_horse['着順'] <= 3).sum()
                        avg_pop = df_horse['人気'].mean()
                        avg_idx = df_horse['タイム指数'].mean() if 'タイム指数' in df_horse.columns else None
                        best_idx = df_horse['タイム指数'].max() if 'タイム指数' in df_horse.columns else None

                        k1,k2,k3,k4,k5 = st.columns(5)
                        k1.metric("🏅 成績",     f"{wins}勝/{total_n}戦",
                                  f"複勝 {top3}/{total_n}")
                        k2.metric("🎯 勝率",      f"{wins/total_n*100:.1f}%",
                                  f"複勝率 {top3/total_n*100:.1f}%")
                        k3.metric("👥 平均人気", f"{avg_pop:.1f}番人気")
                        if avg_idx: k4.metric("📊 平均タイム指数", f"{avg_idx:.1f}")
                        if best_idx: k5.metric("🚀 最高タイム指数", f"{best_idx:.1f}")

                        st.markdown("---")

                        # ── タブ構成 ───────────────────────────────────
                        tab_idx, tab_agari, tab_weight, tab_rank, tab_table = st.tabs([
                            "📈 タイム指数", "💨 末脚・タイム差", "⚖️ 馬体重", "👑 着順・人気", "📋 詳細テーブル"
                        ])

                        def make_line(data, x, y, color='#4B8BFF', title='', reverse_y=False, zero_line=False):
                            if data.empty: return None
                            scale = alt.Scale(reverse=reverse_y)
                            chart = alt.Chart(data).mark_line(point=True, color=color).encode(
                                x=alt.X(f'{x}:N', sort=None, title='日付'),
                                y=alt.Y(f'{y}:Q', scale=scale, title=title),
                                tooltip=[f'{x}:N', f'{y}:Q']
                            ).interactive()
                            if zero_line:
                                rule = alt.Chart(pd.DataFrame({'y':[0]})).mark_rule(
                                    color='gray', strokeDash=[4,4]).encode(y='y:Q')
                                return chart + rule
                            return chart

                        with tab_idx:
                            st.caption("数値が高いほど優秀なパフォーマンスです。")
                            if 'タイム指数' in df_horse.columns:
                                d = df_horse[['日付_str','タイム指数','競馬場','芝/ダート','距離','着順']].dropna(subset=['タイム指数'])
                                if not d.empty:
                                    chart = alt.Chart(d).mark_line(point=True).encode(
                                        x=alt.X('日付_str:N', sort=None, title=''),
                                        y=alt.Y('タイム指数:Q', title='タイム指数'),
                                        color=alt.condition(
                                            alt.datum['着順'] == 1,
                                            alt.value('#FF4B4B'),
                                            alt.value('#4B8BFF')
                                        ),
                                        tooltip=['日付_str','タイム指数','競馬場','芝/ダート','距離','着順']
                                    ).interactive().properties(height=280)
                                    st.altair_chart(chart, use_container_width=True)
                                    st.caption("赤点 = 1着")
                            else:
                                st.info("タイム指数データがありません（補正タイム偏差列が必要）")

                        with tab_agari:
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.caption("上がり3F (低いほど末脚切れる)")
                                if '上り' in df_horse.columns:
                                    d = df_horse[['日付_str','上り']].dropna(subset=['上り'])
                                    if not d.empty:
                                        c = make_line(d,'日付_str','上り','#FFA500','上がり3F(秒)',reverse_y=True)
                                        if c: st.altair_chart(c, use_container_width=True)
                            with col_b:
                                st.caption("1着タイム差 (0.0=1着)")
                                if 'タイム差' in df_horse.columns:
                                    d = df_horse[['日付_str','タイム差']].dropna(subset=['タイム差'])
                                    if not d.empty:
                                        c = make_line(d,'日付_str','タイム差','#FF4B4B','タイム差(秒)',reverse_y=True)
                                        if c: st.altair_chart(c, use_container_width=True)

                        with tab_weight:
                            col_w1, col_w2 = st.columns(2)
                            wt_col = '当日馬体重' if '当日馬体重' in df_horse.columns else None
                            with col_w1:
                                st.caption("馬体重推移 (kg)")
                                if wt_col:
                                    d = df_horse[['日付_str', wt_col]].replace(0, np.nan).dropna(subset=[wt_col])
                                    if not d.empty:
                                        c = make_line(d,'日付_str', wt_col,'#4BFF8B','馬体重(kg)')
                                        if c: st.altair_chart(c, use_container_width=True)
                                else:
                                    st.info("馬体重データなし")
                            with col_w2:
                                st.caption("馬体重増減 (前走比)")
                                if '馬体重増減' in df_horse.columns:
                                    d = df_horse[['日付_str','馬体重増減']].dropna(subset=['馬体重増減'])
                                    if not d.empty:
                                        c = make_line(d,'日付_str','馬体重増減','#888','増減(kg)',zero_line=True)
                                        if c: st.altair_chart(c, use_container_width=True)

                        with tab_rank:
                            st.caption("数値が低い（1位に近い）ほど上位。Y軸反転。")
                            rank_d = df_horse[['日付_str','着順','人気']].dropna(subset=['着順'])
                            if not rank_d.empty:
                                melted = rank_d.melt('日付_str', value_vars=['着順','人気'],
                                                     var_name='項目', value_name='順位')
                                max_v = max(18, rank_d[['着順','人気']].max().max() + 1)
                                cr = alt.Chart(melted).mark_line(point=True).encode(
                                    x=alt.X('日付_str:N', sort=None, title=''),
                                    y=alt.Y('順位:Q', scale=alt.Scale(domain=[1, max_v], reverse=True)),
                                    color=alt.Color('項目:N'),
                                    tooltip=['日付_str','項目','順位']
                                ).interactive().properties(height=250)
                                st.altair_chart(cr, use_container_width=True)

                                # 人気 vs 着順の散布図（人気より走った/凡走した分析）
                                st.markdown("##### 📊 人気 vs 着順（左下 = 人気通り勝ち / 右上 = 人気負け）")
                                sc_d = df_horse[['日付_str','人気','着順','競馬場','レース名']].dropna(subset=['人気','着順'])
                                if not sc_d.empty:
                                    sc_d['超過率'] = sc_d['着順'] - sc_d['人気']
                                    sc = alt.Chart(sc_d).mark_circle(size=100).encode(
                                        x=alt.X('人気:Q', title='人気', scale=alt.Scale(domain=[1,18])),
                                        y=alt.Y('着順:Q', title='着順', scale=alt.Scale(domain=[1,18], reverse=True)),
                                        color=alt.Color('超過率:Q',
                                            scale=alt.Scale(scheme='redblue', domain=[-8,8]),
                                            legend=alt.Legend(title='着順-人気')),
                                        tooltip=['日付_str','人気','着順','競馬場','レース名']
                                    ).properties(height=250).interactive()
                                    # 対角線 (人気=着順の線)
                                    diag_data = pd.DataFrame({'x':range(1,19),'y':range(1,19)})
                                    diag = alt.Chart(diag_data).mark_line(color='gray',strokeDash=[3,3],opacity=0.5).encode(
                                        x='x:Q', y=alt.Y('y:Q', scale=alt.Scale(reverse=True)))
                                    st.altair_chart(sc + diag, use_container_width=True)
                                    st.caption("青 = 人気より上の着順（激走）/ 赤 = 人気を下回る着順（凡走）/ 灰点線 = 人気通り")

                        with tab_table:
                            disp_cols = [c for c in [
                                '日付_str','競馬場','芝/ダート','距離','馬場','着順','人気',
                                '騎手','斤量','当日馬体重','馬体重増減','上り','タイム差','タイム指数','単勝'
                            ] if c in df_horse.columns]
                            show = df_horse[disp_cols].copy().sort_values('日付_str', ascending=False)
                            show = show.rename(columns={'日付_str':'日付'})

                            def color_rank(row):
                                try:
                                    r = int(row['着順'])
                                    if r == 1: return ['background-color:rgba(255,215,0,0.2)']*len(row)
                                    if r <= 3: return ['background-color:rgba(192,192,192,0.15)']*len(row)
                                except: pass
                                return ['']*len(row)

                            fmt = {}
                            for c in ['上り','タイム差','タイム指数','単勝']:
                                if c in show.columns: fmt[c] = '{:.1f}'
                            if '馬体重増減' in show.columns: fmt['馬体重増減'] = '{:+.0f}'
                            st.dataframe(
                                show.style.apply(color_rank, axis=1).format(fmt),
                                use_container_width=True, hide_index=True
                            )

            except Exception as e:
                import traceback
                st.error(f"データの読み込み中にエラーが発生しました: {e}")
                with st.expander("詳細エラーログ"):
                    st.code(traceback.format_exc())
