# ============================================================
# features_engine.py
# (ebi × AI 共同開発：純粋な整理整頓用モジュール)
# ============================================================

# 🌟 絶対に変更してはいけない「勝負の特徴量」リスト
# （現状のapp.pyの学習・推論で使っているものと完全一致させます）
NUM_FEATURES = [
    '枠番', '馬番', '年齢', '距離', '斤量', '出走頭数', '馬体重_num', '馬体重増減', 
    '斤量差', '休養日数', '前走_着順', '2走前_着順', '3走前_着順', '過去3走平均着順', 
    '前走着順パーセント', '直近3走着順パーセント', '前走_スピード指数', '2走前_スピード指数', 
    '3走前_スピード指数', '過去3走平均スピード指数', '近5走_中央値スピード指数', 
    '近5走_最高スピード指数', '上昇度_スピード指数', '前走距離補正タイム差', '前走上り偏差', 
    '位置取りショック', '同レース逃げ馬頭数', '同レース先行馬頭数', 'コース適性_着順パーセント', 
    '乗り替わりフラグ', '馬場替わりフラグ', '距離変更フラグ', '前走失速フラグ', '前走大敗フラグ', 
    '穴馬_距離変更一変', '穴馬_馬場替わり一変', '穴馬_勝負の乗り替わり', '穴馬_実力馬の巻き返し'
]

CAT_FEATURES = [
    '競馬場', '芝/ダート', '天候', '馬場', '父系', '母系', '母父系', 
    '前走芝ダート', '回り', 'コース地形', '脚質カテゴリ', '騎手_競馬場', '騎手_距離'
]

TE_COLS = ['調教師', '父', '母父', '騎手']


# 🌟 重複を防ぐための共通関数
def classify_style(pos):
    """
    最終コーナー通過順位から脚質カテゴリを判定する（学習・推論で完全共通化）
    """
    import pandas as pd
    if pd.isna(pos):
        return '不明'
    pos = float(pos)
    if pos <= 2: return '逃げ'
    elif pos <= 5: return '先行'
    elif pos <= 10: return '差し'
    else: return '追込'

def create_features(df, te_dicts=None):
    import pandas as pd
    import numpy as np
    import re
    from src.utils import VENUE_MAWARI, VENUE_CHIKEI, TRACK_CONDITION_MAP, classify_race_class
    
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

    # ── コース統計（リーク防止版: expanding window で過去データのみ使用）────
    # 従来はdf全体で mean/std を計算してからマージしていたため、
    # 未来レースの走破タイムが「コース基準値」に混入するリークがあった。
    # shift(1).expanding() = 「現レース時点より前の実績」のみで基準値を計算 → リーク消滅
    for _c in ['競馬場', '芝/ダート', '距離']:
        df[_c] = df[_c].astype(str).str.strip()
    df = df.sort_values('日付').reset_index(drop=True)

    df['コース平均'] = (
        df.groupby(['競馬場', '芝/ダート', '距離'])['走破タイム秒']
        .transform(lambda x: x.shift(1).expanding(min_periods=3).mean())
    )
    df['コース標準偏差'] = (
        df.groupby(['競馬場', '芝/ダート', '距離'])['走破タイム秒']
        .transform(lambda x: x.shift(1).expanding(min_periods=3).std())
    )

    # 過去実績3件未満（新設コース・距離）の場合は同コース全体平均でフォールバック
    _fb_mean = df.groupby(['競馬場', '芝/ダート', '距離'])['走破タイム秒'].transform('mean')
    _fb_std  = df.groupby(['競馬場', '芝/ダート', '距離'])['走破タイム秒'].transform('std')
    df['コース平均']       = df['コース平均'].fillna(_fb_mean)
    df['コース標準偏差'] = df['コース標準偏差'].fillna(_fb_std).fillna(1.0)

    # 距離を数値型に戻す（後続処理用）
    df['距離'] = pd.to_numeric(df['距離'], errors='coerce')

    df['スピード指数'] = np.where(
        df['コース標準偏差'] > 0,
        50 - ((df['走破タイム秒'] - df['コース平均']) / df['コース標準偏差']) * 10,
        50
    )
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

    return df, te_dicts
