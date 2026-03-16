# ============================================================
# features_engine.py
# keiba-ebye 特徴量エンジン v2.0
#
# 役割:
#   - classify_style() を唯一の定義場所に集約 (③バグ修正)
#   - 特徴量リストを一元管理 (③責務分離)
#   - CSVに存在するが未使用だった特徴量を追加 (②特徴量追加)
#   - 学習時・推論時で共通のFE処理を提供 (③責務分離)
#
# app.py への統合方法:
#   from features_engine import (
#       classify_style,
#       NUM_FEATURES, CAT_FEATURES, TE_COLS,
#       LATEST_HORSE_COLS_RENAME,
#       LATEST_HORSE_COLS_KEEP,
#       build_train_features,
#       apply_te,
#       build_predict_features,
#   )
# ============================================================

import re
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


# ============================================================
# ① 脚質分類 (唯一の定義: prepare_model_and_data と run_real_prediction
#             両方で同じロジックを使うことを保証)
# ============================================================
def classify_style(pos):
    """最終コーナー順位から脚質カテゴリに変換する。

    Parameters
    ----------
    pos : float or None
        最終コーナー通過順位

    Returns
    -------
    str
        '逃げ' | '先行' | '差し' | '追込' | '不明'
    """
    if pd.isna(pos):
        return '不明'
    if pos <= 2.5:
        return '逃げ'
    elif pos <= 5.5:
        return '先行'
    elif pos <= 9.5:
        return '差し'
    else:
        return '追込'


# ============================================================
# ② 特徴量リスト (追加分にコメントで理由を明記)
# ============================================================

# ---- 数値特徴量 ----
NUM_FEATURES = [
    # --- 基本情報 ---
    '枠番', '馬番', '年齢', '距離', '斤量', '出走頭数',

    # --- 馬体重 ---
    '馬体重_num',           # 当日馬体重
    '馬体重増減',           # ★追加: 前走比体重変化 (CSV列: 馬体重増減)
    '斤量差',               # ★追加: レース平均斤量との差 (CSV列: 斤量差)

    # --- 間隔・ローテ ---
    '休養日数',             # 前走からの日数

    # --- 着順系 ---
    '前走_着順', '2走前_着順', '3走前_着順', '過去3走平均着順',
    '前走着順パーセント',    # ★追加: 前走の相対的着順 (CSV列: 前走着順パーセント)
    '直近3走着順パーセント', # ★追加: 近3走の着順パーセント平均 (CSV列: 直近3走着順パーセント)

    # --- スピード指数系 ---
    '前走_スピード指数', '2走前_スピード指数', '3走前_スピード指数',
    '過去3走平均スピード指数', '近5走_中央値スピード指数',
    '近5走_最高スピード指数', '上昇度_スピード指数',

    # --- タイム差系 ---
    '前走距離補正タイム差',  # ★追加: 前走の1着比補正済みタイム差 (CSV列: 前走距離補正タイム差)

    # --- 上がり系 ---
    '前走上り偏差',          # ★追加: 前走の上がり3Fのレース内偏差 (CSV列: 前走上り偏差)

    # --- コーナー・ペース ---
    '位置取りショック',
    '同レース逃げ馬頭数', '同レース先行馬頭数',
    'コース適性_着順パーセント',

    # --- フラグ系 (0/1) ---
    '乗り替わりフラグ',      # ★追加: 前走から騎手変更 (CSV列: 乗り替わりフラグ)
    '馬場替わりフラグ',      # ★追加: 前走から芝/ダート変更 (CSV列: 馬場替わりフラグ)
    '距離変更フラグ',        # ★追加: 前走から距離変更 (CSV列: 距離変更フラグ)
    '前走失速フラグ',        # ★追加: 前走で末脚が失速したか (CSV列: 前走失速フラグ)
    '前走大敗フラグ',        # ★追加: 前走で大敗したか (CSV列: 前走大敗フラグ)

    # --- 穴馬複合フラグ (0/1) ---
    '穴馬_距離変更一変',     # ★追加: 距離変更で一変期待の穴馬フラグ
    '穴馬_馬場替わり一変',   # ★追加: 馬場変更で一変期待の穴馬フラグ
    '穴馬_勝負の乗り替わり', # ★追加: 強化騎手乗り替わりの穴馬フラグ
    '穴馬_実力馬の巻き返し', # ★追加: 前走凡走からの巻き返し期待フラグ
]

# Target Encoding 対象列 (TEした結果を num_features に追加)
TE_COLS = [
    '騎手',
    '調教師',
    '父',
    '調教師_騎手',
    '騎手_競馬場',  # ★追加: 騎手×競馬場の複合TE (CSV列: 騎手_競馬場)
    '騎手_距離',    # ★追加: 騎手×距離の複合TE  (CSV列: 騎手_距離)
]

# ---- カテゴリ特徴量 ----
CAT_FEATURES = [
    '競馬場', '馬場', '芝/ダート',
    '天候',          # ★追加: 天候 (CSV列: 天候) ※雨/晴で馬場状態と相互作用
    '回り',          # ★追加: 右回り/左回り (CSV列: 回り)
    'コース地形',    # ★追加: 急坂/平坦 等 (CSV列: コース地形)
    '性別', '脚質カテゴリ',
    '父', '父系', '母', '母系', '母父', '母父系',
    '騎手', '調教師', '調教師_騎手',
    '騎手_競馬場',   # ★追加 (TE用だが catboost 的にもカテゴリとして有効)
    '騎手_距離',     # ★追加
]

# ============================================================
# ③ latest_horse_data のリネームと保持カラム定義
#    (prepare_model_and_data と run_real_prediction で一致させる)
# ============================================================
LATEST_HORSE_COLS_RENAME = {
    # 既存
    '着順':         '最新_着順',
    'スピード指数': '最新_スピード指数',
    '人気':         '最新_人気',
    '上り':         '最新_上り',
    '距離':         '最新_距離',
    '斤量':         '最新_斤量',
    '馬体重_num':   '最新_馬体重',
    '日付':         '最新_日付',
    '通過':         '最新_通過',
    # ★追加: 推論時に「前走○○」として使うために保存
    '騎手':             '最新_騎手',           # 乗り替わりフラグ計算用
    '芝/ダート':        '最新_芝ダート',        # 馬場替わりフラグ計算用
    '着順パーセント':   '最新_着順パーセント',  # → 前走着順パーセント
    '失速フラグ':       '最新_失速フラグ',      # → 前走失速フラグ
    '距離補正タイム差': '最新_距離補正タイム差',# → 前走距離補正タイム差
    '上り偏差':         '最新_上り偏差',        # → 前走上り偏差
    '直近3走着順パーセント': '最新_直近3走着順パーセント',  # そのまま使用
    '馬体重増減':       '最新_馬体重増減',      # 参考用
}

LATEST_HORSE_COLS_KEEP = [
    '馬ID', '父', '父系', '母', '母系', '母父', '母父系',
    # 最新レース情報
    '最新_着順', '最新_スピード指数', '最新_人気', '最新_上り',
    '最新_距離', '最新_斤量', '最新_馬体重', '最新_日付', '最新_通過',
    # 乗り替わり・馬場替わりフラグ計算用
    '最新_騎手', '最新_芝ダート',
    # → 前走○○ として使うための前走記録
    '最新_着順パーセント', '最新_失速フラグ',
    '最新_距離補正タイム差', '最新_上り偏差',
    '最新_直近3走着順パーセント',
    # 過去走スピード指数
    '前走_スピード指数', '2走前_スピード指数', '3走前_スピード指数',
    '4走前_スピード指数', '5走前_スピード指数',
    # 過去走着順
    '前走_着順', '2走前_着順', '3走前_着順',
    # コーナー
    '前走_通過', '2走前_通過', '前走_最終コーナー', '2走前_最終コーナー',
]


# ============================================================
# ④ 学習データ用 特徴量エンジニアリング
#    (prepare_model_and_data から分離)
# ============================================================
def build_train_features(df: pd.DataFrame) -> pd.DataFrame:
    """学習用DataFrameに対して特徴量エンジニアリングを適用する。

    CSVに存在するが app.py v1 で未使用だったカラムをそのまま活用し、
    必要な変換・派生特徴量を追加して返す。

    Parameters
    ----------
    df : pd.DataFrame
        load_csv 直後の生データ (dtype=str から数値変換済みを想定)

    Returns
    -------
    pd.DataFrame
        特徴量エンジニアリング適用済み DataFrame
    """
    df = df.copy()

    # --- 基本数値変換 ---
    for col in ['着順', '単勝', '人気', '斤量', '距離', '上り', '枠番', '馬番']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['性別'] = df['性齢'].astype(str).str.extract(r'([牡牝セ])')[0]
    df['年齢'] = pd.to_numeric(
        df['性齢'].astype(str).str.extract(r'(\d+)')[0], errors='coerce'
    )
    df['馬体重_num'] = pd.to_numeric(
        df['馬体重'].astype(str).str.extract(r'(\d+)')[0], errors='coerce'
    )

    # 当日馬体重 (CSV列: 当日馬体重) の数値化
    if '当日馬体重' in df.columns:
        df['馬体重_num'] = pd.to_numeric(df['当日馬体重'], errors='coerce').fillna(df['馬体重_num'])

    # --- ★追加: 体重増減・斤量差 ---
    for col in ['馬体重増減', '斤量差']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            df[col] = np.nan

    # --- タイム変換 ---
    def time_to_seconds(t):
        try:
            m = re.match(r'(\d+):(\d+\.\d+)', str(t))
            return float(m.group(1)) * 60 + float(m.group(2)) if m else float(t)
        except Exception:
            return np.nan

    df['走破タイム秒'] = df['タイム'].apply(time_to_seconds)
    df['日付'] = pd.to_datetime(df['日付'], format='mixed', errors='coerce')
    df = df.dropna(subset=['日付'])

    # --- スピード指数 ---
    df['出走頭数'] = df.groupby('レースID')['馬ID'].transform('count')
    df['着順パーセント'] = (df['着順'] - 1) / (df['出走頭数'] - 1).replace(0, 1)

    course_stats = (
        df.groupby(['競馬場', '芝/ダート', '距離'])['走破タイム秒']
        .agg(['mean', 'std'])
        .reset_index()
    )
    course_stats.columns = ['競馬場', '芝/ダート', '距離', 'コース平均', 'コース標準偏差']
    df = pd.merge(df, course_stats, on=['競馬場', '芝/ダート', '距離'], how='left')
    df['スピード指数'] = np.where(
        df['コース標準偏差'] > 0,
        50 - ((df['走破タイム秒'] - df['コース平均']) / df['コース標準偏差']) * 10,
        50
    )

    # --- 調教師×騎手 複合キー ---
    df['調教師_騎手'] = df['調教師'].astype(str) + '_' + df['騎手'].astype(str)

    # --- ★追加: 騎手×競馬場・距離 複合キー (CSV既存列を優先, なければ生成) ---
    if '騎手_競馬場' not in df.columns:
        df['騎手_競馬場'] = df['騎手ID'].astype(str) + '_' + df['競馬場'].astype(str)
    if '騎手_距離' not in df.columns:
        df['騎手_距離'] = df['騎手ID'].astype(str) + '_' + df['距離'].astype(str)

    # --- 過去走特徴量 (shift) ---
    df = df.sort_values(['馬ID', '日付']).reset_index(drop=True)

    df['前走_着順']  = df.groupby('馬ID')['着順'].shift(1)
    df['2走前_着順'] = df.groupby('馬ID')['着順'].shift(2)
    df['3走前_着順'] = df.groupby('馬ID')['着順'].shift(3)
    df['過去3走平均着順'] = df[['前走_着順', '2走前_着順', '3走前_着順']].mean(axis=1)

    df['前走_スピード指数']  = df.groupby('馬ID')['スピード指数'].shift(1)
    df['2走前_スピード指数'] = df.groupby('馬ID')['スピード指数'].shift(2)
    df['3走前_スピード指数'] = df.groupby('馬ID')['スピード指数'].shift(3)
    df['4走前_スピード指数'] = df.groupby('馬ID')['スピード指数'].shift(4)
    df['5走前_スピード指数'] = df.groupby('馬ID')['スピード指数'].shift(5)

    df['過去3走平均スピード指数'] = df[['前走_スピード指数', '2走前_スピード指数', '3走前_スピード指数']].mean(axis=1)
    df['近5走_中央値スピード指数'] = df[['前走_スピード指数', '2走前_スピード指数', '3走前_スピード指数', '4走前_スピード指数', '5走前_スピード指数']].median(axis=1)
    df['近5走_最高スピード指数']   = df[['前走_スピード指数', '2走前_スピード指数', '3走前_スピード指数', '4走前_スピード指数', '5走前_スピード指数']].max(axis=1)
    df['上昇度_スピード指数'] = df['前走_スピード指数'] - df['近5走_中央値スピード指数']

    # --- コーナー・脚質 ---
    df['前走_通過']  = df.groupby('馬ID')['通過'].shift(1)
    df['2走前_通過'] = df.groupby('馬ID')['通過'].shift(2)

    def parse_last_corner(x):
        s = str(x)
        if '-' in s:
            last = s.split('-')[-1]
            return float(last) if last.isdigit() else np.nan
        return float(s) if s.isdigit() else np.nan

    df['前走_最終コーナー']  = pd.to_numeric(
        df['前走_通過'].fillna('').astype(str).apply(parse_last_corner), errors='coerce'
    )
    df['2走前_最終コーナー'] = pd.to_numeric(
        df['2走前_通過'].fillna('').astype(str).apply(parse_last_corner), errors='coerce'
    )

    # classify_style は モジュールレベル関数を使用 (重複定義なし)
    df['脚質カテゴリ'] = df['前走_最終コーナー'].apply(classify_style)

    df['前走逃げフラグ']  = (df['前走_最終コーナー'] <= 2).astype(int)
    df['前走先行フラグ']  = ((df['前走_最終コーナー'] > 2) & (df['前走_最終コーナー'] <= 5)).astype(int)
    df['同レース逃げ馬頭数'] = df.groupby('レースID')['前走逃げフラグ'].transform('sum')
    df['同レース先行馬頭数'] = df.groupby('レースID')['前走先行フラグ'].transform('sum')

    df['コース適性_着順パーセント'] = (
        df.groupby(['馬ID', '競馬場', '芝/ダート'])['着順パーセント']
        .transform(lambda x: x.shift(1).expanding().mean())
        .fillna(0.5)
    )
    df['位置取りショック'] = df['前走_最終コーナー'] - df['2走前_最終コーナー']

    df['前走_日付'] = df.groupby('馬ID')['日付'].shift(1)
    df['休養日数']  = (df['日付'] - df['前走_日付']).dt.days

    # --- ★追加: CSV既存列の数値化 (未変換の場合に備える) ---
    csv_num_cols = [
        '乗り替わりフラグ', '馬場替わりフラグ', '距離変更フラグ',
        '前走失速フラグ', '前走大敗フラグ', '前走上り偏差',
        '前走着順パーセント', '直近3走着順パーセント', '前走距離補正タイム差',
        '穴馬_距離変更一変', '穴馬_馬場替わり一変',
        '穴馬_勝負の乗り替わり', '穴馬_実力馬の巻き返し',
        '馬体重増減', '斤量差',
    ]
    for col in csv_num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            df[col] = np.nan
            logger.warning(f"特徴量 '{col}' がCSVに存在しません。NaNで補完します。")

    # --- ★追加: 天候・回り・コース地形 の文字列正規化 ---
    for col in ['天候', '回り', 'コース地形']:
        if col in df.columns:
            df[col] = df[col].fillna('不明').astype(str)
        else:
            df[col] = '不明'
            logger.warning(f"カテゴリ特徴量 '{col}' がCSVに存在しません。'不明'で補完します。")

    # 騎手_競馬場 / 騎手_距離 の文字列正規化
    for col in ['騎手_競馬場', '騎手_距離']:
        if col in df.columns:
            df[col] = df[col].fillna('不明').astype(str)
        else:
            df[col] = '不明'

    logger.info(f"build_train_features 完了: {len(df)} 行, {len(df.columns)} 列")
    return df


# ============================================================
# ⑤ Target Encoding
# ============================================================
def apply_te(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    te_cols: list,
    target_col: str = '馬券内',
    global_mean: float = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, float]:
    """訓練 → テスト順序でターゲットエンコーディングを適用する。

    Parameters
    ----------
    train_df, test_df : pd.DataFrame
    te_cols : list of str
    target_col : str
    global_mean : float or None

    Returns
    -------
    train_df, test_df (TE列追加済み), te_dicts, global_mean
    """
    if global_mean is None:
        global_mean = train_df[target_col].mean()

    te_dicts = {}
    for col in te_cols:
        if col not in train_df.columns:
            logger.warning(f"TE対象列 '{col}' が訓練データに存在しません。スキップします。")
            continue
        te_dicts[col] = train_df.groupby(col)[target_col].mean().to_dict()
        train_df[f'{col}_TE'] = train_df[col].map(te_dicts[col]).fillna(global_mean)
        test_df[f'{col}_TE']  = test_df[col].map(te_dicts[col]).fillna(global_mean)

    return train_df, test_df, te_dicts, global_mean


# ============================================================
# ⑥ 推論時 特徴量構築
#    (run_real_prediction 内の df_test に対する変換)
#    latest_horse_data とのマージ後に呼び出す
# ============================================================
def build_predict_features(
    df_test: pd.DataFrame,
    horse_course_dict: dict,
    race_date_str: str,
    te_dicts: dict,
    global_mean: float,
    race_track_type: str,
    race_distance: float,
    race_venue: str,
) -> pd.DataFrame:
    """推論用 DataFrame に対して特徴量エンジニアリングを適用する。

    Parameters
    ----------
    df_test : pd.DataFrame
        scraping 結果から作成し latest_horse_data をマージ済みのDF
    horse_course_dict : dict
        馬ID × 競馬場 × 芝/ダート → コース適性着順パーセント
    race_date_str : str
        予想レースの日付 (例: '2025-01-05')
    te_dicts : dict
        学習時に作成した target encoding 辞書
    global_mean : float
        TE の欠損補完用グローバル平均
    race_track_type : str
        '芝' | 'ダート' | '障害'
    race_distance : float
        レース距離 (m)
    race_venue : str
        競馬場名

    Returns
    -------
    pd.DataFrame
        推論直前の特徴量構築済み df_test
    """
    df = df_test.copy()

    # --- 基本派生 ---
    df['性別'] = df['性齢'].astype(str).str.extract(r'([牡牝セ])')[0]
    df['年齢'] = pd.to_numeric(
        df['性齢'].astype(str).str.extract(r'(\d+)')[0], errors='coerce'
    )
    df['調教師_騎手'] = df['調教師'].astype(str) + '_' + df['騎手'].astype(str)

    # --- 過去走着順シフト ---
    df['3走前_着順'] = df.get('2走前_着順', np.nan)
    df['2走前_着順'] = df.get('前走_着順', np.nan)
    df['前走_着順']  = df.get('最新_着順', np.nan)
    df['過去3走平均着順'] = df[['前走_着順', '2走前_着順', '3走前_着順']].mean(axis=1)

    # --- 過去走スピード指数シフト ---
    df['5走前_スピード指数'] = df.get('4走前_スピード指数', np.nan)
    df['4走前_スピード指数'] = df.get('3走前_スピード指数', np.nan)
    df['3走前_スピード指数'] = df.get('2走前_スピード指数', np.nan)
    df['2走前_スピード指数'] = df.get('前走_スピード指数', np.nan)
    df['前走_スピード指数']  = df.get('最新_スピード指数', np.nan)

    df['過去3走平均スピード指数'] = df[['前走_スピード指数', '2走前_スピード指数', '3走前_スピード指数']].mean(axis=1)
    df['近5走_中央値スピード指数'] = df[['前走_スピード指数', '2走前_スピード指数', '3走前_スピード指数', '4走前_スピード指数', '5走前_スピード指数']].median(axis=1)
    df['近5走_最高スピード指数']   = df[['前走_スピード指数', '2走前_スピード指数', '3走前_スピード指数', '4走前_スピード指数', '5走前_スピード指数']].max(axis=1)
    df['上昇度_スピード指数'] = df['前走_スピード指数'] - df['近5走_中央値スピード指数']

    # --- コーナー・脚質 (classify_style は共通関数を使用) ---
    df['前走_通過']  = df.get('最新_通過', np.nan)

    def parse_last_corner(x):
        s = str(x)
        if '-' in s:
            last = s.split('-')[-1]
            return float(last) if last.isdigit() else np.nan
        return float(s) if s.isdigit() else np.nan

    df['前走_最終コーナー'] = pd.to_numeric(
        df['前走_通過'].fillna('').astype(str).apply(parse_last_corner), errors='coerce'
    )
    df['脚質カテゴリ'] = df['前走_最終コーナー'].apply(classify_style)  # ★共通関数使用

    df['前走逃げフラグ']  = (df['前走_最終コーナー'] <= 2).astype(int)
    df['前走先行フラグ']  = ((df['前走_最終コーナー'] > 2) & (df['前走_最終コーナー'] <= 5)).astype(int)
    df['同レース逃げ馬頭数'] = df['前走逃げフラグ'].sum()
    df['同レース先行馬頭数'] = df['前走先行フラグ'].sum()

    df['コース適性_着順パーセント'] = (
        df.set_index(['馬ID', '競馬場', '芝/ダート']).index
        .map(horse_course_dict).fillna(0.5)
    )
    df['位置取りショック'] = df['前走_最終コーナー'] - df.get('2走前_最終コーナー', np.nan)

    race_date_obj = pd.to_datetime(race_date_str)
    df['休養日数'] = (
        (race_date_obj - pd.to_datetime(df['最新_日付'])).dt.days
        if '最新_日付' in df.columns else np.nan
    )

    # --- ★追加: 新特徴量を推論時に構築 ---

    # 乗り替わりフラグ: スクレイプした騎手名 vs 前走騎手名
    if '最新_騎手' in df.columns:
        df['乗り替わりフラグ'] = (df['騎手'] != df['最新_騎手']).astype(int)
    else:
        df['乗り替わりフラグ'] = 0

    # 馬場替わりフラグ: 今回芝/ダート vs 前走
    if '最新_芝ダート' in df.columns:
        df['馬場替わりフラグ'] = (df['芝/ダート'] != df['最新_芝ダート']).astype(int)
    else:
        df['馬場替わりフラグ'] = 0

    # 距離変更フラグ: 今回距離 vs 最新_距離
    if '最新_距離' in df.columns:
        df['距離変更フラグ'] = (df['距離'] != df['最新_距離']).astype(int)
    else:
        df['距離変更フラグ'] = 0

    # 前走失速フラグ: latest_horse_data に保存した最新_失速フラグを使用
    df['前走失速フラグ'] = pd.to_numeric(df.get('最新_失速フラグ', 0), errors='coerce').fillna(0)

    # 前走大敗フラグ: 前走着順が出走頭数の70%以下なら大敗 (または saved flag)
    if '最新_着順' in df.columns and '出走頭数' in df.columns:
        n = df['出走頭数'].replace(0, 1)
        df['前走大敗フラグ'] = ((df['最新_着順'] / n) > 0.7).astype(int)
    else:
        df['前走大敗フラグ'] = 0

    # 前走上り偏差: latest_horse_data の最新_上り偏差
    df['前走上り偏差'] = pd.to_numeric(df.get('最新_上り偏差', np.nan), errors='coerce')

    # 前走着順パーセント
    df['前走着順パーセント'] = pd.to_numeric(df.get('最新_着順パーセント', np.nan), errors='coerce')

    # 直近3走着順パーセント
    df['直近3走着順パーセント'] = pd.to_numeric(df.get('最新_直近3走着順パーセント', np.nan), errors='coerce').fillna(0.5)

    # 前走距離補正タイム差
    df['前走距離補正タイム差'] = pd.to_numeric(df.get('最新_距離補正タイム差', np.nan), errors='coerce')

    # 馬体重増減: 今日の体重 - 最新_馬体重
    if '馬体重_num' in df.columns and '最新_馬体重' in df.columns:
        df['馬体重増減'] = df['馬体重_num'] - pd.to_numeric(df['最新_馬体重'], errors='coerce')
    else:
        df['馬体重増減'] = pd.to_numeric(df.get('最新_馬体重増減', 0), errors='coerce').fillna(0)

    # 斤量差: この馬の斤量 - レース平均斤量
    if '斤量' in df.columns:
        avg_kinryo = df['斤量'].mean()
        df['斤量差'] = df['斤量'] - avg_kinryo
    else:
        df['斤量差'] = 0.0

    # 穴馬複合フラグ
    df['穴馬_距離変更一変']     = ((df['距離変更フラグ'] == 1) & (df['直近3走着順パーセント'] < 0.4)).astype(int)
    df['穴馬_馬場替わり一変']   = ((df['馬場替わりフラグ'] == 1) & (df['直近3走着順パーセント'] < 0.4)).astype(int)
    df['穴馬_勝負の乗り替わり'] = ((df['乗り替わりフラグ'] == 1) & (df['直近3走着順パーセント'] < 0.5)).astype(int)
    df['穴馬_実力馬の巻き返し'] = ((df['前走大敗フラグ'] == 1) & (df['近5走_最高スピード指数'] >= 55)).astype(int)

    # 天候・回り・コース地形 (スクレイプデータから設定 or デフォルト)
    if '天候' not in df.columns:
        df['天候'] = '不明'
    if '回り' not in df.columns:
        # 競馬場から推定 (簡易版)
        venue_mawari = {
            '札幌': '右回り', '函館': '右回り', '福島': '右回り', '新潟': '左回り',
            '東京': '左回り', '中山': '右回り', '中京': '左回り', '京都': '右回り',
            '阪神': '右回り', '小倉': '右回り',
        }
        df['回り'] = df['競馬場'].map(venue_mawari).fillna('不明')
    if 'コース地形' not in df.columns:
        venue_chikei = {
            '札幌': '平坦', '函館': '平坦', '福島': '急坂', '新潟': '平坦',
            '東京': '急坂', '中山': '急坂', '中京': '急坂', '京都': '緩坂',
            '阪神': '急坂', '小倉': '平坦',
        }
        df['コース地形'] = df['競馬場'].map(venue_chikei).fillna('不明')

    # 騎手_競馬場 / 騎手_距離 の複合キー
    if '騎手_競馬場' not in df.columns:
        df['騎手_競馬場'] = df['騎手'].astype(str) + '_' + df['競馬場'].astype(str)
    if '騎手_距離' not in df.columns:
        df['騎手_距離'] = df['騎手'].astype(str) + '_' + df['距離'].astype(str)

    # --- Target Encoding 適用 ---
    for col in TE_COLS:
        if col in te_dicts:
            df[f'{col}_TE'] = df[col].map(te_dicts[col]).fillna(global_mean)
        else:
            df[f'{col}_TE'] = global_mean

    return df


# ============================================================
# ⑦ latest_horse_data 構築ヘルパー
#    (prepare_model_and_data 内で使用)
# ============================================================
def build_latest_horse_data(df: pd.DataFrame) -> pd.DataFrame:
    """各馬の最新レース情報をまとめた DataFrame を構築する。

    Parameters
    ----------
    df : pd.DataFrame
        build_train_features 適用済みのフルデータ

    Returns
    -------
    pd.DataFrame
        馬ID をキーとする最新レース情報テーブル
    """
    df_latest = df.groupby('馬ID').tail(1).copy()
    df_latest = df_latest.rename(columns=LATEST_HORSE_COLS_RENAME)

    keep = [c for c in LATEST_HORSE_COLS_KEEP if c in df_latest.columns]
    return df_latest[keep].copy()
