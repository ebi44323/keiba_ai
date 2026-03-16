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
import joblib
import logging

# ── features_engine から共通定義をインポート ──────────────────────────────
from features_engine import (
    classify_style,
    NUM_FEATURES        as FE_NUM_FEATURES,
    CAT_FEATURES        as FE_CAT_FEATURES,
    TE_COLS             as FE_TE_COLS,
    LATEST_HORSE_COLS_RENAME,
    LATEST_HORSE_COLS_KEEP,
    build_train_features,
    apply_te,
    build_predict_features,
    build_latest_horse_data,
)

# ── ロギング設定 ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('keiba_ebye.log', encoding='utf-8'),
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger('keiba_ebye')

MODEL_CACHE_PATH = 'keiba_ebye_model_cache.joblib'

st.set_page_config(page_title="keiba-ebye 予測ダッシュボード", page_icon="🐴", layout="wide")
st.title("🐴 keiba-ebye 予測ダッシュボード")
st.markdown("えーびーあい (ebi × AI × Eye) が、極限まで高められた精度でお宝馬を暴き出すかも。。。。")

# ==========================================
# 🌟 【ebiさん考案】最強の「名寄せ（フルネーム復元）」関数
# ==========================================
def resolve_name(short_name, known_names):
    if pd.isna(short_name) or short_name == '不明': return '不明'
    clean_name = re.sub(r'[☆▲△◇★\n\s　]', '', str(short_name))
    clean_name = re.sub(r'\[[東西地外]\]', '', clean_name)
    clean_name = re.sub(r'(栗東|美浦)', '', clean_name)
    if not clean_name: return '不明'
    aliases = {
        "鮫島駿": "鮫島克駿", "鮫島良": "鮫島良太",
        "吉田隼": "吉田隼人", "武幸": "武幸四郎", "菅原明": "菅原明良"
    }
    if clean_name in aliases: clean_name = aliases[clean_name]
    normalized_dict = {}
    for kn in known_names:
        if pd.isna(kn): continue
        norm_kn = re.sub(r'[☆▲△◇★\n\s　]', '', str(kn))
        norm_kn = re.sub(r'\[[東西地外]\]', '', norm_kn)
        norm_kn = re.sub(r'(栗東|美浦)', '', norm_kn)
        if norm_kn not in normalized_dict: normalized_dict[norm_kn] = []
        normalized_dict[norm_kn].append(kn)
    if clean_name in normalized_dict: return sorted(normalized_dict[clean_name], key=len)[0]
    forward_matches = []
    for norm_kn, orig_names in normalized_dict.items():
        if norm_kn.startswith(clean_name): forward_matches.extend(orig_names)
    if forward_matches: return sorted(forward_matches, key=len)[0]
    partial_matches = []
    for norm_kn, orig_names in normalized_dict.items():
        if clean_name in norm_kn: partial_matches.extend(orig_names)
    if partial_matches: return sorted(partial_matches, key=len)[0]
    return clean_name

# ==========================================
# 1. AIエンジン ── アンサンブル学習 + joblib キャッシュ
# ==========================================
@st.cache_resource
def prepare_model_and_data():
    """
    【③ 責務分離】4ステップに分割
    【① アンサンブル】異なるシード/パラメータの LGBMRanker 3本を平均
    【  joblib】特徴量数が変わらなければ再起動時はキャッシュから即ロード
    """
    # ── STEP1: データ読み込み & FE ────────────────────────────────────────
    logger.info("STEP1: データ読み込み開始")
    try:
        df_raw = pd.read_csv('learning_data_perfect_tier.zip', compression='zip', dtype=str)
    except FileNotFoundError:
        df_raw = pd.read_csv('learning_data_perfect_tier.csv', dtype=str)
    logger.info(f"  → {len(df_raw)} 行 読み込み完了")

    df = build_train_features(df_raw)

    horse_course_dict = (
        df.groupby(['馬ID', '競馬場', '芝/ダート'])['着順パーセント']
        .mean().to_dict()
    )
    latest_horse_data = build_latest_horse_data(df)
    logger.info(f"  → latest_horse_data: {len(latest_horse_data)} 頭")

    # ── STEP2: 学習データ準備 ─────────────────────────────────────────────
    logger.info("STEP2: 学習データ準備")
    df_valid = df.dropna(subset=['着順', '単勝']).copy()
    df_valid['馬券内'] = (pd.to_numeric(df_valid['着順'], errors='coerce') <= 3).astype(int)

    num_features = list(FE_NUM_FEATURES)
    cat_features = list(FE_CAT_FEATURES)

    cat_categories_dict = {}
    for col in cat_features:
        if col not in df_valid.columns: df_valid[col] = '不明'
        df_valid[col] = df_valid[col].fillna('不明').astype('category')
        cats = list(df_valid[col].cat.categories)
        if '不明' not in cats: cats.append('不明')
        cat_categories_dict[col] = cats

    known_jockeys  = df_valid['騎手'].dropna().unique().tolist()
    known_trainers = df_valid['調教師'].dropna().unique().tolist()

    df_valid = df_valid.sort_values(['レースID', '馬番'])

    max_date   = df_valid['日付'].max()
    test_start = max_date - pd.Timedelta(days=30)
    train_df = df_valid[df_valid['日付'] < test_start].copy()
    test_df  = df_valid[df_valid['日付'] >= test_start].copy()

    train_df, test_df, te_dicts, global_mean = apply_te(train_df, test_df, FE_TE_COLS)
    for col in FE_TE_COLS:
        te_col = f'{col}_TE'
        if te_col not in num_features and te_col in train_df.columns:
            num_features.append(te_col)

    features = [f for f in (cat_features + num_features) if f in train_df.columns]
    logger.info(f"  → 特徴量数: {len(features)} (cat={len(cat_features)}, num={len(num_features)})")

    train_groups = train_df.groupby('レースID', sort=False).size().values
    test_groups  = test_df.groupby('レースID', sort=False).size().values

    # ── STEP3: アンサンブル学習 or キャッシュ読み込み ──────────────────────
    models = None
    if os.path.exists(MODEL_CACHE_PATH):
        try:
            cache = joblib.load(MODEL_CACHE_PATH)
            if cache.get('n_features') == len(features):
                models = cache['models']
                logger.info(f"  → joblib キャッシュから {len(models)} モデルを読み込みました")
            else:
                logger.info("  → 特徴量数変化のためキャッシュ破棄して再学習")
        except Exception as e:
            logger.warning(f"  → キャッシュ読み込み失敗: {e} → 再学習")

    if models is None:
        logger.info("STEP3: アンサンブル学習開始 (LGBMRanker × 3)")
        # 【① アンサンブル】3モデルの設定（シード・サブサンプル比を変える）
        ensemble_configs = [
            dict(n_estimators=500, learning_rate=0.01, num_leaves=63,
                 colsample_bytree=0.7, subsample=0.8, random_state=42),
            dict(n_estimators=500, learning_rate=0.01, num_leaves=50,
                 colsample_bytree=0.6, subsample=0.9, random_state=123),
            dict(n_estimators=600, learning_rate=0.008, num_leaves=80,
                 colsample_bytree=0.8, subsample=0.7, random_state=777),
        ]
        models = []
        cat_cols_used = [f for f in cat_features if f in features]
        for i, cfg in enumerate(ensemble_configs):
            m = lgb.LGBMRanker(max_bin=255, cat_smooth=10,
                                importance_type='gain', **cfg)
            m.fit(
                train_df[features], train_df['馬券内'],
                group=train_groups,
                categorical_feature=cat_cols_used,
                eval_set=[(test_df[features], test_df['馬券内'])],
                eval_group=[test_groups],
            )
            models.append(m)
            logger.info(f"  → モデル {i+1}/3 学習完了")

        joblib.dump({'models': models, 'n_features': len(features)}, MODEL_CACHE_PATH)
        logger.info(f"  → アンサンブルモデルを保存 ({MODEL_CACHE_PATH})")

    # ── STEP4: バリデーション回収率 ──────────────────────────────────────
    logger.info("STEP4: 回収率評価")
    raw_scores_list = [m.predict(test_df[features]) for m in models]
    ensemble_scores = np.mean(raw_scores_list, axis=0)
    test_df['予測スコア'] = ensemble_scores
    test_df['exp_score'] = np.exp(
        test_df['予測スコア'] - test_df.groupby('レースID')['予測スコア'].transform('max')
    )
    test_df['AI勝率'] = test_df['exp_score'] / test_df.groupby('レースID')['exp_score'].transform('sum')

    top_preds = (
        test_df.sort_values(['レースID', 'AI勝率'], ascending=[True, False])
        .groupby('レースID').head(1)
    )
    win_hits      = top_preds[pd.to_numeric(top_preds['着順'], errors='coerce') == 1]
    invest_amount = len(top_preds) * 100
    win_return    = (pd.to_numeric(win_hits['単勝'], errors='coerce') * 100).sum()
    recent_return_rate = (win_return / invest_amount * 100) if invest_amount > 0 else 0.0
    logger.info(f"  → 直近30日 単勝回収率: {recent_return_rate:.1f}%")

    # ── STEP5: 血統辞書 ───────────────────────────────────────────────────
    try:
        ped_df = pd.read_csv('pedigree_master_all.csv', dtype=str)
        ped_df['馬ID'] = ped_df['馬ID'].astype(str).str.zfill(10)
        ped_dict = (
            ped_df.set_index('馬ID')[['父', '父系', '母', '母系', '母父', '母父系']]
            .to_dict('index')
        )
    except Exception:
        ped_dict = {}
        logger.warning("pedigree_master_all.csv が見つかりません。血統データなしで動作します。")

    return (
        models, features, cat_features, num_features,
        cat_categories_dict, latest_horse_data, horse_course_dict,
        ped_dict, known_jockeys, known_trainers,
        te_dicts, global_mean, recent_return_rate,
    )


with st.spinner('keiba-ebye フルパワーAIエンジンを起動・学習中... (初回のみ数分かかります)'):
    (models, features, cat_features, num_features,
     cat_categories_dict, latest_horse_data, horse_course_dict,
     ped_dict, known_jockeys, known_trainers,
     te_dicts, global_mean, recent_return_rate) = prepare_model_and_data()

headers = {"User-Agent": "Mozilla/5.0"}

# ==========================================
# 2. スクレイピング ＆ アナリティクス関数群
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
            res = requests.get(url, headers=headers, timeout=10)
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
            res = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(res.content, 'html.parser')
            ids = set(re.findall(r'/race/(\d{12})', res.text))
            for r_id in ids:
                if not (1 <= int(r_id[4:6]) <= 10): continue
                place_dict = {'01':'札幌','02':'函館','03':'福島','04':'新潟','05':'東京','06':'中山','07':'中京','08':'京都','09':'阪神','10':'小倉'}
                place = place_dict.get(r_id[4:6], '不明')
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
            res = requests.get(url, headers=headers, timeout=10); res.encoding = 'euc-jp'
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
    def parse_td(td_element):
        if not td_element: return []
        html = str(td_element).replace('<br>', '\n').replace('<br/>', '\n').replace('<br />', '\n')
        html = re.sub(r'</?(div|li|ul|p|span|strong)[^>]*>', '\n', html, flags=re.I)
        lines = BeautifulSoup(html, 'html.parser').get_text().split('\n')
        return [line.strip() for line in lines if line.strip()]

    for url in [f"https://race.netkeiba.com/race/result.html?race_id={race_id}", f"https://db.netkeiba.com/race/{race_id}/"]:
        try:
            res = requests.get(url, headers=headers, timeout=10)
            html_text = res.content.decode('euc-jp', errors='ignore')
            soup = BeautifulSoup(html_text, 'html.parser')
            tables = soup.find_all('table', class_=re.compile(r'Pay_Table_01|pay_table_01', re.I))
            if not tables: tables = soup.find_all('table', summary='払い戻し')
            for tbl in tables:
                current_kind = None
                for tr in tbl.find_all('tr'):
                    th = tr.find('th')
                    if th:
                        th_text = re.sub(r'\s+', '', th.text)
                        th_class = " ".join(th.get('class', [])).lower()
                        if 'tansho' in th_class or '単勝' in th_text: current_kind = '単勝'
                        elif 'fukusho' in th_class or '複勝' in th_text: current_kind = '複勝'
                        elif 'umaren' in th_class or '馬連' in th_text: current_kind = '馬連'
                        elif 'wide' in th_class or 'ワイド' in th_text: current_kind = 'ワイド'
                        else: current_kind = None
                    if not current_kind: continue
                    tds = tr.find_all('td')
                    if not tds: continue
                    res_td = tr.find('td', class_=re.compile(r'Result', re.I))
                    pay_td = tr.find('td', class_=re.compile(r'Payout', re.I))
                    if not res_td and len(tds) >= 2: res_td = tds[0]; pay_td = tds[1]
                    if not res_td or not pay_td: continue
                    r_lines = parse_td(res_td); p_lines = parse_td(pay_td)
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

    try:
        yahoo_id = str(race_id)[2:]
        url_yh = f"https://sports.yahoo.co.jp/keiba/race/result/{yahoo_id}/"
        res_y = requests.get(url_yh, headers=headers, timeout=10)
        soup_y = BeautifulSoup(res_y.text, 'html.parser')
        current_kind = None
        for tr in soup_y.find_all('tr'):
            th = tr.find('th')
            if th:
                th_text = th.text.strip()
                if th_text in ['単勝', '複勝', '馬連', 'ワイド']: current_kind = th_text
                else: current_kind = None
            if not current_kind: continue
            tds = tr.find_all('td')
            if len(tds) < 2: continue
            r_lines = parse_td(tds[0]); p_lines = parse_td(tds[1])
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

def generate_txt_report(results_list):
    txt = "=== 🏇 keiba-ebye 予想レポート ===\n\n"
    for r in results_list:
        txt += "="*50 + "\n"
        txt += f"■ {r['date']} | {r['place']} {r['num']}R ({r['track']}{r['dist']}m) ■\n"
        txt += f"🐎 【展開予想】\n{r['pace']}\n"
        txt += f"🔮 【AI自信度】\n{r['confidence']}\n"
        txt += "-"*50 + "\n"
        for rank, row in r['df'].iterrows():
            ev_str = f" 📈期待値:{row['期待値']:.2f}" if row['期待値'] >= 1.5 else ""
            txt += f" {row['印']} {rank+1}位: [{row['枠番']}枠{row['馬番']}番] {row['馬名']} ({row['脚質カテゴリ']}) - 勝率 {row['勝率(AI予測)']*100:.1f}% / 複勝率 {row['複勝率(AI予測)']*100:.1f}% (オッズ {row['単勝オッズ']}倍){ev_str}\n"
        txt += "-"*50 + "\n"
        if r['topics']:
            txt += "📝 要注目トピック馬:\n"
            for t in r['topics']: txt += f"  {t}\n"
            txt += "-"*50 + "\n"
        txt += f"🤖 AI推奨買い目:\n  {r['reco']}\n"
        txt += "="*50 + "\n\n"
    return txt

# ==========================================
# 3. 本格AI予測関数
# ==========================================
def run_real_prediction(race_id, race_date_str):
    error_log = []
    odds_dict = {}
    html_text = ""

    # オッズ取得 アプローチ1: netkeiba 隠しAPI
    try:
        odds_api_url = f'https://race.netkeiba.com/api/api_get_jra_odds.html?type=1&action=init&race_id={race_id}'
        api_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Referer": f"https://race.netkeiba.com/odds/index.html?type=b1&race_id={race_id}",
            "X-Requested-With": "XMLHttpRequest"
        }
        r_api = requests.get(odds_api_url, headers=api_headers, timeout=5)
        api_data = json.loads(r_api.text)
        if 'data' in api_data and 'odds' in api_data['data'] and '1' in api_data['data']['odds']:
            for uma_num, odds_list in api_data['data']['odds']['1'].items():
                if str(uma_num).isdigit():
                    odds_dict[int(uma_num)] = float(odds_list[0])
    except Exception as e:
        error_log.append(f"netkeiba APIからのオッズ取得に失敗: {e}")

    # オッズ取得 アプローチ2: Yahoo!競馬
    if not odds_dict:
        try:
            yahoo_race_id = str(race_id)[2:]
            yahoo_url = f"https://sports.yahoo.co.jp/keiba/race/odds/tfw/{yahoo_race_id}/"
            r_yahoo = requests.get(yahoo_url, headers=headers, timeout=5)
            soup_y = BeautifulSoup(r_yahoo.text, 'html.parser')
            for tr in soup_y.find_all('tr'):
                tds = tr.find_all('td')
                if len(tds) >= 4:
                    uma_td = tds[1]
                    odds_span = tr.find('span', class_='fB')
                    if uma_td and odds_span:
                        u_m = re.search(r'^\s*(\d+)\s*$', uma_td.text)
                        o_m = re.search(r'\d{1,4}\.\d+', odds_span.text)
                        if u_m and o_m:
                            odds_dict[int(u_m.group(1))] = float(o_m.group(0))
        except Exception as e:
            error_log.append(f"Yahoo競馬からのオッズ取得に失敗: {e}")

    # 出馬表取得
    for fetch_url in [
        f'https://race.netkeiba.com/race/shutuba.html?race_id={race_id}',
        f'https://race.netkeiba.com/race/result.html?race_id={race_id}',
        f'https://db.netkeiba.com/race/{race_id}/'
    ]:
        try:
            r = requests.get(fetch_url, headers=headers, timeout=10); r.encoding = 'euc-jp'
            soup = BeautifulSoup(r.text, 'html.parser')
            if soup.select_one('.Shutuba_Table') or soup.select_one('.RaceTable01') or soup.select_one('.race_table_01') or soup.select_one('#All_Result_Table'):
                html_text = r.text
                break
        except: pass

    if not html_text: return None, None, None, None, None, None, None, None, ["❌ 出馬表が取得できませんでした。"]
    soup = BeautifulSoup(html_text, 'html.parser')

    race_data_box = soup.find('div', class_='RaceData01') or soup.find('dl', class_='racedata')
    if not race_data_box: return None, None, None, None, None, None, None, None, ["❌ レース条件が見つかりません。"]

    race_text = race_data_box.text.replace('\n', '')
    baba_match = re.search(r'馬場:([良稍重不良]+)', race_text)
    todays_baba = baba_match.group(1) if baba_match else '良'
    track_dist_match = re.search(r'(芝|ダ|障|障害).*?(\d+)m', race_text)
    track_type = "芝" if track_dist_match and track_dist_match.group(1) == "芝" else "ダート" if track_dist_match and "ダ" in track_dist_match.group(1) else "障害"
    distance = float(track_dist_match.group(2)) if track_dist_match else 1600.0
    place = {'01':'札幌','02':'函館','03':'福島','04':'新潟','05':'東京','06':'中山','07':'中京','08':'京都','09':'阪神','10':'小倉'}.get(str(race_id)[4:6], '東京')

    # 天候取得
    weather_match = re.search(r'天候:([晴曇雨小雪]+)', race_text)
    todays_tenki = weather_match.group(1) if weather_match else '晴'

    horses = []
    table = soup.select_one('.Shutuba_Table') or soup.select_one('.RaceTable01') or soup.select_one('.race_table_01') or soup.select_one('#All_Result_Table')
    if not table: return None, None, None, None, None, None, None, None, ["❌ 出走馬の一覧表が見つかりません。"]

    ths = table.find_all('th')
    headers_text = [th.text.strip().replace('\n', '') for th in ths]
    def get_idx(keywords):
        for i, h in enumerate(headers_text):
            for kw in keywords:
                if kw in h: return i
        return -1

    waku_idx    = get_idx(['枠'])
    uma_idx     = get_idx(['馬番'])
    kinryo_idx  = get_idx(['斤量'])
    weight_idx  = get_idx(['馬体重'])
    odds_idx    = get_idx(['単勝', 'オッズ', '予想', '人気'])
    sex_age_idx = get_idx(['性齢'])
    jockey_idx  = get_idx(['騎手'])
    trainer_idx = get_idx(['調教師', '厩舎'])

    for tr in table.find_all('tr')[1:]:
        tds = tr.find_all('td')
        if len(tds) < 5: continue
        try:
            umaban = int(re.search(r'\d+', tds[uma_idx].text).group(0)) if uma_idx != -1 and len(tds) > uma_idx and re.search(r'\d+', tds[uma_idx].text) else len(horses) + 1
            waku   = int(re.search(r'\d+', tds[waku_idx].text).group(0)) if waku_idx != -1 and len(tds) > waku_idx and re.search(r'\d+', tds[waku_idx].text) else 0
            horse_a = tr.find('a', href=re.compile(r'/horse/'))
            if not horse_a: continue
            horse_id = re.search(r'\d+', horse_a['href']).group(0)
            jockey_raw  = tds[jockey_idx].text.strip() if jockey_idx != -1 and len(tds) > jockey_idx else "不明"
            jockey_name = resolve_name(jockey_raw, known_jockeys)
            trainer_raw  = tds[trainer_idx].text.strip() if trainer_idx != -1 and len(tds) > trainer_idx else "不明"
            trainer_name = resolve_name(trainer_raw, known_trainers)
            kinryo_text = tds[kinryo_idx].text if kinryo_idx != -1 and len(tds) > kinryo_idx else "55.0"
            kinryo_match = re.search(r'\d+(\.\d+)?', kinryo_text)
            kinryo = float(kinryo_match.group(0)) if kinryo_match else 55.0
            weight_text = tds[weight_idx].text if weight_idx != -1 and len(tds) > weight_idx else ""
            weight_match = re.search(r'^(\d{3})', weight_text.strip())
            weight_val = float(weight_match.group(1)) if weight_match else np.nan
            odds_val = odds_dict.get(umaban, 0.0)
            if odds_val == 0.0 and odds_idx != -1 and len(tds) > odds_idx:
                odds_match = re.search(r'\d{1,4}\.\d+', tds[odds_idx].text)
                if odds_match: odds_val = float(odds_match.group(0))
            if odds_val == 0.0:
                for td in tds:
                    class_list = td.get('class', [])
                    if any(c in ['Odds', 'Popular', 'txt_c'] for c in class_list):
                        om = re.search(r'\d{1,4}\.\d+', td.text)
                        if om: odds_val = float(om.group(0)); break
            if odds_val == 0.0: odds_val = 10.0
            sex_age = tds[sex_age_idx].text.strip() if sex_age_idx != -1 and len(tds) > sex_age_idx else "牡3"
            horses.append({
                '枠番': waku, '馬番': umaban, '馬名': horse_a.text.strip(), '馬ID': horse_id,
                '性齢': sex_age, '斤量': kinryo, '騎手': jockey_name, '調教師': trainer_name,
                '距離': distance, '競馬場': place, '芝/ダート': track_type, '馬場': todays_baba,
                '天候': todays_tenki, '馬体重_num': weight_val, '単勝オッズ': odds_val
            })
        except: pass

    if not horses: return None, None, None, None, None, None, None, None, ["❌ 出走馬データの読み取りに失敗しました。"]

    try:
        df_test = pd.DataFrame(horses)
        df_test['出走頭数'] = len(df_test)
        df_test = pd.merge(df_test, latest_horse_data, on='馬ID', how='left')

        for col in ['父', '父系', '母', '母系', '母父', '母父系']:
            if col not in df_test.columns: df_test[col] = np.nan

        for i, row in df_test.iterrows():
            hid = row['馬ID']
            if pd.isna(row.get('父')) or row.get('父') == '不明':
                ped = ped_dict.get(hid, {})
                for col in ['父', '父系', '母', '母系', '母父', '母父系']:
                    df_test.at[i, col] = ped.get(col, '不明')

        # ★ 特徴量構築は features_engine に委譲 (classify_style 重複排除)
        df_test = build_predict_features(
            df_test,
            horse_course_dict=horse_course_dict,
            race_date_str=race_date_str,
            te_dicts=te_dicts,
            global_mean=global_mean,
            race_track_type=track_type,
            race_distance=distance,
            race_venue=place,
        )

        # 数値・カテゴリ型変換
        for col in num_features:
            df_test[col] = pd.to_numeric(df_test[col], errors='coerce')
        for col in cat_features:
            if col not in df_test.columns: df_test[col] = '不明'
            cats = cat_categories_dict.get(col, ['不明'])
            if '不明' not in cats: cats.append('不明')
            df_test[col] = pd.Categorical(df_test[col].fillna('不明'), categories=cats)

        nige_count   = int(df_test['同レース逃げ馬頭数'].iloc[0]) if not df_test.empty else 0
        senko_count  = int(df_test['同レース先行馬頭数'].iloc[0]) if not df_test.empty else 0

        if nige_count >= 3: pace_text = f"🔥 【ハイペース濃厚】 前走逃げた馬が{nige_count}頭もおり先行争いが激化。差し・追込馬の台頭に警戒！"
        elif nige_count == 0: pace_text = f"🐌 【スローペース濃厚】 確たる逃げ馬が不在。先行馬({senko_count}頭)の押し切り、前残りに注意。"
        else: pace_text = f"🐎 【ミドルペース】 逃げ馬{nige_count}頭、先行馬{senko_count}頭。平均的なペースで実力が反映されやすい展開。"

        # ★ アンサンブル推論 (3モデルの平均)
        raw_scores_list = [m.predict(df_test[features]) for m in models]
        raw_scores = np.mean(raw_scores_list, axis=0)

        exp_scores = np.exp(raw_scores - np.max(raw_scores))
        win_probs  = exp_scores / np.sum(exp_scores)

        df_test['勝率(AI予測)'] = win_probs
        df_test['複勝率(AI予測)'] = np.clip(win_probs * 2.8, 0, 0.99)
        df_test['期待値'] = df_test['勝率(AI予測)'] * df_test['単勝オッズ']
        df_test = df_test.sort_values('勝率(AI予測)', ascending=False).reset_index(drop=True)

        marks = ['◎', '〇', '▲', '△', '☆'] + [''] * (len(df_test) - 5)
        df_test['印'] = marks[:len(df_test)]

        p1, p2 = df_test.loc[0, '勝率(AI予測)'], df_test.loc[1, '勝率(AI予測)']
        score_diff = p1 - p2
        top1_umaban = df_test.loc[0, '馬番']
        himo_umabans = df_test.loc[1:4, '馬番'].astype(str).tolist() if len(df_test) >= 5 else df_test.loc[1:, '馬番'].astype(str).tolist()
        himo_str = "・".join(himo_umabans)

        has_unraced = ('新馬' in race_text) or ('未出走' in race_text) or df_test['前走_着順'].isna().any()

        ana_horse_nums = []
        topics_list = []
        for rank, row in df_test.iterrows():
            if not has_unraced and rank >= 4 and row['期待値'] >= 1.5:
                topics_list.append(f"📌 {row['馬名']} (期待値特大の穴馬！)")
                if f"{row['馬番']}番" not in ana_horse_nums: ana_horse_nums.append(f"{row['馬番']}番")
        ana_str = "・".join(str(n) for n in ana_horse_nums[:3]) if ana_horse_nums else ""

        if has_unraced:
            confidence_text = "🛑 【見送り推奨・未出走混在】 過去データのない馬が含まれており、AIの予測精度が担保できません。"
            reco = f"⚠️ **購入見送り** (データ不足によるリスク大)\n※観戦に留めるか、どうしても買う場合は◎ {top1_umaban}番 の単複を少額で。"
            df_test['期待値'] = 0.0
        elif p1 >= 0.25 and score_diff >= 0.10:
            confidence_text = f"💎 【鉄板レース】 ◎が抜けた存在({p1*100:.1f}%)！ 軸は不動です。"
            reco = f"🎯 【本命・単勝勝負】 ◎ {top1_umaban}番 の単勝。\n  🔗 馬単・3連単: {top1_umaban}着固定 → 相手: {himo_str}"
            if ana_str: reco += f"\n  💣 余裕があれば穴馬({ana_str}番)へのヒモ流しも推奨。"
        elif score_diff <= 0.03 and p1 < 0.20:
            confidence_text = "🌪️ 【波乱レース】 上位の実力が拮抗の大混戦！ 穴馬からのヒモ荒れに警戒してください。"
            reco = f"⚠️ 【ボックス推奨】 上位陣 ({top1_umaban}・{himo_str}番) の馬連・3連複ボックス。"
            if ana_str: reco += f"\n  💣 大穴狙い: 穴馬({ana_str}番)を絡めたワイドや3連複が面白いです。"
        else:
            confidence_text = "⚖️ 【中穴狙いレース】 上位はまとまっていますが、展開次第で伏兵の台頭もあります。"
            reco = f"🎯 【馬連・ワイド】 ◎ {top1_umaban}番 から相手 ({himo_str}番) への流し。"
            if ana_str: reco += f"\n  💣 妙味狙い: {top1_umaban}番から穴馬({ana_str}番)へのワイドで高配当！"

        return df_test, topics_list, reco, pace_text, confidence_text, track_type, place, distance, error_log

    except Exception as e:
        error_log.append(f"❌ 予測AI内部で致命的なエラーが発生: {traceback.format_exc()}")
        return None, None, None, None, None, None, None, None, error_log

# ==========================================
# 4. メインUI構成
# ==========================================
st.sidebar.markdown("## 🕹️ keiba-ebye メニュー")
action = st.sidebar.radio("機能を選択", [
    "⏩ 次のレースを予想",
    "📜 本日の全レース予想",
    "📅 今週末の全レース予想",
    "🔍 レースを指定して予想",
    "📝 1日の振り返り (答え合わせ)",
    "🧪 性能試験 (バックテスト)",
    "📈 長期成績分析",
    "🐴 愛馬の成長記録",
    "🔬 特徴量重要度",
])

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ 設定")

# 【④ フラクショナルケリー】係数をサイドバーから調整可能に
kelly_fraction = st.sidebar.slider(
    "ケリー係数 (リスク調整)", 0.1, 1.0, 0.25, 0.05,
    help="1.0=フルケリー(高リスク) / 0.25=推奨(安定) / 0.1=超保守"
)
budget = st.sidebar.number_input("予算 (円)", 5000, 100000, 10000, 1000)

# モデル再学習ボタン
if st.sidebar.button("🔄 モデルを再学習 (キャッシュクリア)"):
    if os.path.exists(MODEL_CACHE_PATH):
        os.remove(MODEL_CACHE_PATH)
        st.cache_resource.clear()
        st.sidebar.success("キャッシュを削除しました。ページをリロードすると再学習します。")
    else:
        st.sidebar.info("キャッシュは存在しません。")

st.sidebar.markdown("---")
st.sidebar.metric("📊 直近30日 単勝回収率", f"{recent_return_rate:.1f}%",
                  delta="基準100%" if recent_return_rate >= 100 else None)

tokyo_tz = pytz.timezone('Asia/Tokyo')
now = datetime.datetime.now(tokyo_tz)

def display_error_log(err_log):
    st.error("⚠️ 予想データまたは結果の取得に失敗しました。")
    with st.expander("🔍 エラー解析ログを見る (デバッグ用)"):
        for log in err_log: st.write(f"- {log}")

def display_result(df_res, topics, reco, pace_text, confidence_text):
    tab1, tab2, tab3 = st.tabs(["📊 予想一覧", "💡 展開・買い目", "🔍 性能詳細"])

    with tab1:
        if "鉄板" in confidence_text: st.success(confidence_text)
        elif "波乱" in confidence_text: st.error(confidence_text)
        else: st.info(confidence_text)

        def highlight_ev(row): return ['background-color: rgba(255, 99, 71, 0.3)' if row['期待値'] >= 1.5 else '' for _ in row]
        show_df = df_res[['印', '馬番', '馬名', '脚質カテゴリ', '単勝オッズ', '勝率(AI予測)', '複勝率(AI予測)', '期待値']].copy()
        show_df = show_df.rename(columns={'勝率(AI予測)': '勝率', '複勝率(AI予測)': '複勝率', '単勝オッズ': 'オッズ', '脚質カテゴリ': '脚質'})

        # 【④ フラクショナルケリー】係数をサイドバー値で適用
        def calc_kelly_bet(row):
            if row['期待値'] < 1.0: return 0
            p = row['勝率']
            b = row['オッズ'] - 1.0
            if b <= 0: return 0
            f_star = p - ((1.0 - p) / b)
            bet = int(max(0, f_star * kelly_fraction) * budget / 100) * 100
            return min(bet, int(budget * 0.3))  # 1レース1頭の上限: 予算の30%

        show_df['推奨ベット'] = show_df.apply(calc_kelly_bet, axis=1).astype(str) + "円"
        show_df.loc[show_df['推奨ベット'] == "0円", '推奨ベット'] = "見送り"
        show_df['勝率']  = (show_df['勝率'] * 100).map('{:.1f}%'.format)
        show_df['複勝率'] = (show_df['複勝率'] * 100).map('{:.1f}%'.format)

        st.dataframe(
            show_df.style.apply(highlight_ev, axis=1)
                   .format({'期待値': '{:.2f}', 'オッズ': '{:.1f}'}),
            use_container_width=True, hide_index=True
        )

    with tab2:
        st.info(f"**🏇 展開予想:**\n{pace_text}")
        ev_horses = df_res[(df_res.index < 5) & (df_res['期待値'] >= 1.5)]
        if not ev_horses.empty: st.error(f"💰 **【期待値レーダー発動】** {', '.join(ev_horses['馬名'].tolist())} に強烈なオッズ妙味あり！")
        if topics: st.warning("**📝 要注目トピック馬:**\n\n" + "\n".join(topics))
        st.success(f"**🤖 AI推奨買い目:**\n\n{reco}")

        # 穴馬フラグ表示
        flag_cols = [c for c in ['乗り替わりフラグ', '馬場替わりフラグ', '距離変更フラグ',
                                  '穴馬_距離変更一変', '穴馬_馬場替わり一変', '穴馬_勝負の乗り替わり', '穴馬_実力馬の巻き返し']
                     if c in df_res.columns]
        if flag_cols:
            flag_df = df_res[['馬番', '馬名'] + flag_cols].copy()
            flag_df = flag_df[flag_df[flag_cols].sum(axis=1) > 0]
            if not flag_df.empty:
                st.markdown("**🚨 一変・変化アラート馬:**")
                label_map = {
                    '乗り替わりフラグ': '🔄乗替', '馬場替わりフラグ': '🌱馬場変',
                    '距離変更フラグ': '📏距離変', '穴馬_距離変更一変': '💥距離一変',
                    '穴馬_馬場替わり一変': '💥馬場一変', '穴馬_勝負の乗り替わり': '💥勝負乗替',
                    '穴馬_実力馬の巻き返し': '🔥巻き返し'
                }
                for _, row in flag_df.iterrows():
                    active = [label_map.get(c, c) for c in flag_cols if row[c] == 1]
                    st.caption(f"  {row['馬番']}番 {row['馬名']}: {' / '.join(active)}")

    with tab3:
        detail_cols = ['馬番', '馬名', '騎手', '調教師',
                       '近5走_中央値スピード指数', '上昇度_スピード指数',
                       'コース適性_着順パーセント', '位置取りショック']
        detail_df = df_res[[c for c in detail_cols if c in df_res.columns]].copy()
        detail_df = detail_df.rename(columns={
            'コース適性_着順パーセント': 'コース適性(%)',
            '近5走_中央値スピード指数': '本来の指数(中央値)',
            '上昇度_スピード指数': '成長度・復調度'
        })
        st.markdown("※『コース適性(%)』は数字が低い（0に近い）ほどそのコースが得意なことを示します。")
        fmt = {}
        if '本来の指数(中央値)' in detail_df.columns: fmt['本来の指数(中央値)'] = '{:.1f}'
        if '成長度・復調度' in detail_df.columns: fmt['成長度・復調度'] = '{:.1f}'
        if 'コース適性(%)' in detail_df.columns: fmt['コース適性(%)'] = '{:.2f}'
        if '位置取りショック' in detail_df.columns: fmt['位置取りショック'] = '{:.1f}'
        st.dataframe(detail_df.style.format(fmt), use_container_width=True, hide_index=True)


# ==========================================
# 5. アクション分岐
# ==========================================
if action in ["⏩ 次のレースを予想", "📜 本日の全レース予想", "🔍 レースを指定して予想"]:
    todays_races = get_todays_races()
    if not todays_races:
        st.warning(f"本日 ({now.strftime('%Y/%m/%d')}) はJRAのレースが開催されていません。")
    else:
        if action == "⏩ 次のレースを予想":
            st.subheader("🕒 まもなく出走するレース")
            races_sorted_by_time = sorted(todays_races, key=lambda x: x['time'])
            next_race = next((r for r in races_sorted_by_time if r['time'] > now), None)
            if next_race:
                mins_left = int((next_race['time'] - now).total_seconds() / 60)
                st.info(f"👉 **{next_race['place']} {next_race['num']}R** 「{next_race['title']}」 (あと **{mins_left}** 分)")
                if st.button("🚀 keiba-ebye 予想起動！", type="primary"):
                    with st.spinner('AIが推論中...'):
                        res_df, topics, reco, pace_text, conf_text, _, _, _, err_log = run_real_prediction(next_race['id'], now.strftime('%Y-%m-%d'))
                        if res_df is not None: display_result(res_df, topics, reco, pace_text, conf_text)
                        else: display_error_log(err_log)
            else: st.success("🏁 本日の全レースは終了しました。")

        elif action == "📜 本日の全レース予想":
            st.subheader(f"📅 本日の全レース一覧")
            if st.button("🚀 全レース一括予想", type="primary"):
                my_bar = st.progress(0, text="推論中...")
                results_for_txt = []
                for i, r in enumerate(todays_races):
                    st.markdown(f"#### ■ {r['place']} {r['num']}R")
                    res_df, topics, reco, pace_text, conf_text, track_type, place, dist, err_log = run_real_prediction(r['id'], now.strftime('%Y-%m-%d'))
                    if res_df is not None:
                        display_result(res_df, topics, reco, pace_text, conf_text)
                        results_for_txt.append({'date': now.strftime('%Y年%m月%d日'), 'place': place, 'num': r['num'], 'track': track_type, 'dist': dist, 'pace': pace_text, 'confidence': conf_text, 'df': res_df, 'topics': topics, 'reco': reco})
                    else: display_error_log(err_log)
                    time.sleep(1.0)
                    my_bar.progress((i + 1) / len(todays_races))
                if results_for_txt:
                    st.download_button("📥 予想レポートをダウンロード (.txt)", data=generate_txt_report(results_for_txt), file_name=f"keiba_ebye_{now.strftime('%Y%m%d')}.txt", mime="text/plain")

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
        with st.spinner('出馬表を収集中...'):
            target_races = get_todays_races(target_date)
        if not target_races: st.error("出馬表が未発表です。")
        else:
            my_bar = st.progress(0, text="推論中...")
            results_for_txt = []
            for i, r in enumerate(target_races):
                with st.expander(f"🏁 {r['place']} {r['num']}R"):
                    res_df, topics, reco, pace_text, conf_text, track_type, place, dist, err_log = run_real_prediction(r['id'], f"{target_date[:4]}-{target_date[4:6]}-{target_date[6:]}")
                    if res_df is not None:
                        display_result(res_df, topics, reco, pace_text, conf_text)
                        results_for_txt.append({'date': f"{target_date[:4]}年{target_date[4:6]}月{target_date[6:]}日", 'place': place, 'num': r['num'], 'track': track_type, 'dist': dist, 'pace': pace_text, 'confidence': conf_text, 'df': res_df, 'topics': topics, 'reco': reco})
                    else: display_error_log(err_log)
                time.sleep(1.0)
                my_bar.progress((i + 1) / len(target_races))
            if results_for_txt:
                st.download_button(f"📥 {target_date[4:6]}/{target_date[6:]} 予想レポート(.txt)", data=generate_txt_report(results_for_txt), file_name=f"keiba_weekend_{target_date}.txt", mime="text/plain")

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
                    # ★ ケリー成長曲線用
                    'kelly_invest': 0, 'kelly_return': 0, 'kelly_growth': [],
                }
                for i, r in enumerate(races):
                    res_df, topics, reco, pace_text, conf_text, track_type, place, dist, err_log = run_real_prediction(r['id'], target_date.strftime('%Y-%m-%d'))
                    payouts = get_all_payouts(r['id'])
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
                        # 【④ ケリー成長曲線計算】
                        kelly_row = res_df.iloc[0]
                        p_k = kelly_row['勝率(AI予測)']
                        b_k = kelly_row['単勝オッズ'] - 1.0
                        if b_k > 0:
                            f_star = max(0, p_k - (1.0 - p_k) / b_k)
                            bet = int(f_star * kelly_fraction * budget / 100) * 100
                            bet = min(bet, int(budget * 0.3))
                            if bet > 0:
                                stats['kelly_invest'] += bet
                                win = payouts['tansho'].get(honmei, 0)
                                stats['kelly_return'] += int(win * bet / 100) if win > 0 else 0
                                stats['kelly_growth'].append({
                                    'レース': f"{r['place']}{r['num']}R",
                                    '投資': bet,
                                    '回収': int(win * bet / 100) if win > 0 else 0
                                })
                    else:
                        if res_df is None: st.error(f"❌ {r['place']}{r['num']}R: 予想処理失敗")
                        elif not payouts['tansho']: st.warning(f"⚠️ {r['place']}{r['num']}R: 払い戻し取得失敗")
                    time.sleep(0.5)
                    my_bar.progress((i + 1) / len(races))

                tan_rate  = (stats['honmei_tan_return']  / (stats['honmei_races'] * 100) * 100) if stats['honmei_races'] > 0 else 0
                fuku_rate = (stats['honmei_fuku_return'] / (stats['honmei_races'] * 100) * 100) if stats['honmei_races'] > 0 else 0
                uma_rate  = (stats['umaren_return']  / stats['umaren_invest']  * 100) if stats['umaren_invest'] > 0 else 0
                wide_rate = (stats['wide_ana_return'] / stats['wide_ana_invest'] * 100) if stats['wide_ana_invest'] > 0 else 0
                ev_tan_rate  = (stats['ev_tan_return']  / stats['ev_invest'] * 100) if stats['ev_invest'] > 0 else 0
                ev_fuku_rate = (stats['ev_fuku_return'] / stats['ev_invest'] * 100) if stats['ev_invest'] > 0 else 0
                shiba_rate = (stats['shiba_return'] / (stats['shiba_races'] * 100) * 100) if stats['shiba_races'] > 0 else 0
                dart_rate  = (stats['dart_return']  / (stats['dart_races']  * 100) * 100) if stats['dart_races'] > 0 else 0
                exp_rate   = (stats['exp_return']   / (stats['exp_races']   * 100) * 100) if stats['exp_races'] > 0 else 0
                new_rate   = (stats['new_return']   / (stats['new_races']   * 100) * 100) if stats['new_races'] > 0 else 0
                kelly_rate = (stats['kelly_return'] / stats['kelly_invest'] * 100) if stats['kelly_invest'] > 0 else 0

                csv_file = "ai_daily_history.csv"
                daily_data = pd.DataFrame([{
                    '日付': target_date.strftime('%Y/%m/%d'),
                    '本命単勝回収率': round(tan_rate, 1),
                    '本命複勝回収率': round(fuku_rate, 1),
                    '穴馬単勝回収率': round(ev_tan_rate, 1),
                    '穴馬複勝回収率': round(ev_fuku_rate, 1),
                    'ケリー回収率': round(kelly_rate, 1),
                }])
                if os.path.exists(csv_file):
                    existing_df = pd.read_csv(csv_file)
                    for col in ['本命単勝回収率', '本命複勝回収率', '穴馬単勝回収率', '穴馬複勝回収率', 'ケリー回収率']:
                        if col not in existing_df.columns: existing_df[col] = 0.0
                    existing_df = existing_df[existing_df['日付'] != target_date.strftime('%Y/%m/%d')]
                    pd.concat([existing_df, daily_data]).to_csv(csv_file, index=False)
                else:
                    daily_data.to_csv(csv_file, index=False)

                st.markdown("---")
                st.markdown(f"### 🏆 {target_date.strftime('%Y/%m/%d')} レース振り返りレポート")
                st.markdown(f"**対象レース数: {stats['honmei_races']} レース**")
                col1, col2 = st.columns(2)
                with col1:
                    st.success("🎯 【本命(◎) 単勝・複勝成績】")
                    st.write(f"- **単勝 的中率**: {(stats['honmei_tan_hits']/stats['honmei_races']*100 if stats['honmei_races']>0 else 0):.1f}% ({stats['honmei_tan_hits']}R)")
                    st.write(f"- **単勝 回収率**: **{tan_rate:.1f}%**")
                    st.write(f"- **複勝 的中率**: {(stats['honmei_fuku_hits']/stats['honmei_races']*100 if stats['honmei_races']>0 else 0):.1f}% ({stats['honmei_fuku_hits']}R)")
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
                    st.write(f"- 単勝 回収率: **{ev_tan_rate:.1f}%** / 複勝 回収率: **{ev_fuku_rate:.1f}%**")
                    st.markdown("---")
                    st.success(f"💰 【フラクショナルケリー (×{kelly_fraction})】")
                    st.write(f"  投資: ¥{stats['kelly_invest']:,} / 回収: ¥{stats['kelly_return']:,}")
                    st.write(f"  回収率: **{kelly_rate:.1f}%**")

                # 【④ ケリー成長曲線】
                if stats['kelly_growth']:
                    st.markdown("#### 💹 ケリー投資 レース別損益")
                    kg_df = pd.DataFrame(stats['kelly_growth'])
                    kg_df['損益'] = kg_df['回収'] - kg_df['投資']
                    kg_df['累計損益'] = kg_df['損益'].cumsum()
                    import altair as alt
                    c = alt.Chart(kg_df.reset_index()).mark_line(point=True).encode(
                        x=alt.X('レース:N', sort=None, title='レース'),
                        y=alt.Y('累計損益:Q', title='累計損益 (円)'),
                        tooltip=['レース', '投資', '回収', '損益', '累計損益']
                    ).properties(height=200).interactive()
                    st.altair_chart(c, use_container_width=True)


# ==========================================
# 6. バックテスト (⑦ 強化版)
# ==========================================
elif action == "🧪 性能試験 (バックテスト)":
    st.subheader("🧪 バックテスト (強化版)")
    col_a, col_b = st.columns(2)
    with col_a:
        test_date_from = st.date_input("開始日", datetime.date.today() - datetime.timedelta(days=7))
    with col_b:
        test_date_to   = st.date_input("終了日", datetime.date.today() - datetime.timedelta(days=1))

    if st.button("🔥 バックテスト実行！", type="primary"):
        # 指定期間の全開催日を列挙
        date_range = []
        d = test_date_from
        while d <= test_date_to:
            date_range.append(d)
            d += datetime.timedelta(days=1)

        all_races = []
        for d in date_range:
            day_races = get_todays_races(d.strftime('%Y%m%d'))
            for r in day_races:
                r['date_obj'] = d
            all_races.extend(day_races)

        if not all_races:
            st.error("指定期間にレースが見つかりません。")
        else:
            with st.spinner(f'{len(all_races)} レースを推論・集計中...'):
                my_bar = st.progress(0, text="集計中...")
                bt_records = []
                results_for_txt = []
                for i, r in enumerate(all_races):
                    race_date_str = r['date_obj'].strftime('%Y-%m-%d')
                    res_df, topics, reco, pace_text, conf_text, track_type, place, dist, err_log = run_real_prediction(r['id'], race_date_str)
                    t_dict, f_dict = get_payouts(r['id'])
                    if res_df is not None and t_dict:
                        honmei_row = res_df.iloc[0]
                        honmei = honmei_row['馬番']
                        p_k = honmei_row['勝率(AI予測)']
                        b_k = honmei_row['単勝オッズ'] - 1.0
                        kelly_bet = 0
                        if b_k > 0:
                            f_star = max(0, p_k - (1.0 - p_k) / b_k)
                            kelly_bet = int(f_star * kelly_fraction * budget / 100) * 100
                            kelly_bet = min(kelly_bet, int(budget * 0.3))

                        tan_pay  = t_dict.get(honmei, 0)
                        fuku_pay = f_dict.get(honmei, 0)
                        bt_records.append({
                            '日付': r['date_obj'].strftime('%Y/%m/%d'),
                            '競馬場': r['place'],
                            'R': r['num'],
                            '芝/ダート': track_type,
                            '距離': dist,
                            '◎馬番': honmei,
                            '勝率': p_k,
                            '期待値': honmei_row['期待値'],
                            '単勝オッズ': honmei_row['単勝オッズ'],
                            '単勝払戻': tan_pay,
                            '複勝払戻': fuku_pay,
                            '的中(単)': 1 if tan_pay > 0 else 0,
                            '的中(複)': 1 if fuku_pay > 0 else 0,
                            'ケリーBET': kelly_bet,
                            'ケリー回収': int(tan_pay * kelly_bet / 100) if tan_pay > 0 and kelly_bet > 0 else 0,
                        })
                        results_for_txt.append({'date': r['date_obj'].strftime('%Y年%m月%d日'), 'place': place, 'num': r['num'], 'track': track_type, 'dist': dist, 'pace': pace_text, 'confidence': conf_text, 'df': res_df, 'topics': topics, 'reco': reco})
                    time.sleep(0.8)
                    my_bar.progress((i + 1) / len(all_races))

            if not bt_records:
                st.error("集計できるレース結果がありませんでした。")
            else:
                import altair as alt
                bt_df = pd.DataFrame(bt_records)
                total = len(bt_df)
                tan_hits  = bt_df['的中(単)'].sum()
                fuku_hits = bt_df['的中(複)'].sum()
                tan_ret   = (bt_df['単勝払戻'].sum() / (total * 100) * 100) if total > 0 else 0
                fuku_ret  = (bt_df['複勝払戻'].sum() / (total * 100) * 100) if total > 0 else 0
                kelly_inv = bt_df['ケリーBET'].sum()
                kelly_ret_amt = bt_df['ケリー回収'].sum()
                kelly_ret = (kelly_ret_amt / kelly_inv * 100) if kelly_inv > 0 else 0

                # 【⑦ 統計検定: 95%信頼区間】
                # ベルヌーイ試行として単勝的中率の95%信頼区間を計算 (Wilson法)
                n_races = total
                p_hat = tan_hits / n_races if n_races > 0 else 0
                z = 1.96  # 95%信頼区間
                denom = 1 + z**2 / n_races if n_races > 0 else 1
                centre = (p_hat + z**2 / (2 * n_races)) / denom if n_races > 0 else 0
                margin = (z * np.sqrt(p_hat * (1 - p_hat) / n_races + z**2 / (4 * n_races**2))) / denom if n_races > 0 else 0
                ci_low  = max(0, centre - margin) * 100
                ci_high = min(100, centre + margin) * 100

                st.markdown("---")
                st.markdown(f"### 🏆 バックテスト集計レポート ({test_date_from} 〜 {test_date_to})")
                st.markdown(f"**対象: {total} レース**")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("単勝 的中率", f"{tan_hits/total*100:.1f}%" if total > 0 else "-%",
                          f"95%CI: {ci_low:.1f}〜{ci_high:.1f}%")
                c2.metric("単勝 回収率", f"{tan_ret:.1f}%",
                          delta=f"{tan_ret-100:.1f}%" if tan_ret != 0 else None)
                c3.metric("複勝 回収率", f"{fuku_ret:.1f}%")
                c4.metric(f"ケリー(×{kelly_fraction}) 回収率", f"{kelly_ret:.1f}%",
                          f"¥{kelly_inv:,}→¥{kelly_ret_amt:,}")

                # 累積収支グラフ
                bt_df['単勝損益'] = bt_df['単勝払戻'] - 100
                bt_df['ケリー損益'] = bt_df['ケリー回収'] - bt_df['ケリーBET']
                bt_df['累計単勝損益'] = bt_df['単勝損益'].cumsum()
                bt_df['累計ケリー損益'] = bt_df['ケリー損益'].cumsum()
                bt_df['レース番号'] = range(1, len(bt_df) + 1)

                st.markdown("#### 📈 累積損益推移")
                melted = bt_df[['レース番号', '累計単勝損益', '累計ケリー損益']].melt(
                    '레ース番号' if 'レース番号' in bt_df.columns else 'レース番号',
                    var_name='戦略', value_name='累計損益'
                )
                # melt列名修正
                melted = bt_df.melt(id_vars='レース番号',
                                    value_vars=['累計単勝損益', '累計ケリー損益'],
                                    var_name='戦略', value_name='累計損益')
                chart = alt.Chart(melted).mark_line().encode(
                    x=alt.X('レース番号:Q', title='レース番号'),
                    y=alt.Y('累計損益:Q', title='累計損益 (円)'),
                    color='戦略:N',
                    tooltip=['レース番号', '戦略', '累計損益']
                ).properties(height=300).interactive()
                st.altair_chart(chart, use_container_width=True)

                # 条件別分析
                st.markdown("#### 🔍 条件別成績")
                cond_cols = ['芝/ダート', '競馬場']
                for cond in cond_cols:
                    if cond in bt_df.columns:
                        grp = bt_df.groupby(cond).agg(
                            レース数=('的中(単)', 'count'),
                            単勝的中=('的中(単)', 'sum'),
                            単勝払戻合計=('単勝払戻', 'sum'),
                        ).reset_index()
                        grp['投資合計'] = grp['レース数'] * 100
                        grp['単勝回収率(%)'] = (grp['単勝払戻合計'] / grp['投資合計'] * 100).round(1)
                        grp['的中率(%)'] = (grp['単勝的中'] / grp['レース数'] * 100).round(1)
                        st.markdown(f"**{cond}別**")
                        st.dataframe(grp, use_container_width=True, hide_index=True)

                # 詳細テーブル
                with st.expander("📋 全レース詳細"):
                    st.dataframe(bt_df.drop(columns=['単勝損益', 'ケリー損益']), use_container_width=True, hide_index=True)

                if results_for_txt:
                    st.download_button("📥 結果をダウンロード (.txt)",
                                       data=generate_txt_report(results_for_txt),
                                       file_name=f"keiba_backtest_{test_date_from.strftime('%Y%m%d')}_{test_date_to.strftime('%Y%m%d')}.txt",
                                       mime="text/plain")


# ==========================================
# 7. 長期成績分析
# ==========================================
elif action == "📈 長期成績分析":
    st.subheader("📈 AIの長期成績分析 (日々の成長記録)")
    csv_file = "ai_daily_history.csv"
    if not os.path.exists(csv_file):
        st.warning("まだデータがありません。「1日の振り返り」を実行してデータを蓄積してください！")
    else:
        import altair as alt
        history_df = pd.read_csv(csv_file)
        for col in ['本命単勝回収率', '本命複勝回収率', '穴馬単勝回収率', '穴馬複勝回収率', 'ケリー回収率']:
            if col not in history_df.columns: history_df[col] = 0.0
        history_df = history_df.set_index('日付')
        display_cols = ['本命単勝回収率', '本命複勝回収率', '穴馬単勝回収率', '穴馬複勝回収率', 'ケリー回収率']
        show_df = history_df[[c for c in display_cols if c in history_df.columns]].copy()
        st.dataframe(show_df.style.format('{:.1f}%'), use_container_width=True)

        melted = show_df.reset_index().melt('日付', var_name='戦略', value_name='回収率(%)')
        line = alt.Chart(melted).mark_line(point=True).encode(
            x=alt.X('日付:N', sort=None),
            y=alt.Y('回収率(%):Q'),
            color='戦略:N',
            tooltip=['日付', '戦略', '回収率(%)']
        ).interactive()
        rule = alt.Chart(pd.DataFrame({'y': [100]})).mark_rule(color='red', strokeDash=[5, 5]).encode(y='y:Q')
        st.altair_chart(line + rule, use_container_width=True)
        st.caption("赤い点線 = 回収率100% (損益分岐点)")

# ==========================================
# 8. 特徴量重要度
# ==========================================
elif action == "🔬 特徴量重要度":
    st.subheader("🔬 アンサンブルモデル 特徴量重要度")
    import altair as alt
    # 3モデルのGainを平均
    imp_arrays = [m.feature_importances_ for m in models]
    avg_imp = np.mean(imp_arrays, axis=0)
    imp_df = (
        pd.DataFrame({'特徴量': features, '重要度(Gain平均)': avg_imp})
        .sort_values('重要度(Gain平均)', ascending=False)
        .head(40)
        .reset_index(drop=True)
    )
    st.caption("3モデルのGain平均値。数値が高いほど予測への貢献が大きい特徴量です。")
    chart = (
        alt.Chart(imp_df)
        .mark_bar()
        .encode(
            x=alt.X('重要度(Gain平均):Q', title='Gain 平均'),
            y=alt.Y('特徴量:N', sort='-x', title=''),
            color=alt.condition(
                alt.datum['重要度(Gain平均)'] > float(imp_df['重要度(Gain平均)'].median()),
                alt.value('#FF4B4B'), alt.value('#4B8BFF')
            ),
            tooltip=['特徴量', '重要度(Gain平均)']
        )
        .properties(height=800)
    )
    st.altair_chart(chart, use_container_width=True)
    st.dataframe(imp_df.style.bar(subset=['重要度(Gain平均)'], color='#FF4B4B'), use_container_width=True)

# ==========================================
# 9. 愛馬の成長記録
# ==========================================
elif action == "🐴 愛馬の成長記録":
    st.subheader("🐴 愛馬のAI能力評価・成長記録")
    st.markdown("過去のレースにおけるAI指標や成績の推移を時系列でグラフ化します。")
    horse_name = st.text_input("🔍 馬名を入力してください (例: ドウデュース, リバティアイランド)")
    if st.button("成長記録を表示", type="primary") and horse_name:
        with st.spinner(f"{horse_name} のデータを検索中..."):
            try:
                data_file = 'learning_data_perfect_tier.zip'
                if not os.path.exists(data_file):
                    st.error(f"データベースファイル ({data_file}) が見つかりません。")
                else:
                    race_date_str_input = st.session_state.get('review_date', now.strftime('%Y-%m-%d'))
                    df_hist = pd.read_csv(data_file, compression='zip', dtype=str)
                    if '日付' in df_hist.columns:
                        df_hist['日付'] = pd.to_datetime(df_hist['日付'], errors='coerce')
                        target_dt = pd.to_datetime(now.strftime('%Y-%m-%d'))
                        df_hist = df_hist[df_hist['日付'] < target_dt].copy()
                    df_horse = df_hist[df_hist['馬名'] == horse_name].copy()
                    if df_horse.empty:
                        st.warning(f"データベースに「{horse_name}」の過去レース記録が見つかりませんでした。")
                    else:
                        if '日付' in df_horse.columns:
                            df_horse['日付'] = pd.to_datetime(df_horse['日付'], errors='coerce')
                            df_horse = df_horse.sort_values('日付').dropna(subset=['日付'])
                        st.success(f"✅ {len(df_horse)}戦分のデータを取得しました。")
                        weight_col = '当日馬体重' if '当日馬体重' in df_horse.columns else '馬体重' if '馬体重' in df_horse.columns else None
                        agari_col  = '上り' if '上り' in df_horse.columns else '上がり3F' if '上がり3F' in df_horse.columns else None
                        numeric_cols = ['補正タイム偏差', 'タイム差', '着順', '人気', '単勝', weight_col, agari_col]
                        for col in numeric_cols:
                            if col and col in df_horse.columns:
                                df_horse[col] = pd.to_numeric(df_horse[col], errors='coerce')
                        if '補正タイム偏差' in df_horse.columns:
                            df_horse['タイム指数'] = 50 - (df_horse['補正タイム偏差'] * 10)
                        chart_df = df_horse.set_index('日付')
                        st.markdown(f"### 📈 {horse_name} の実績推移")
                        st.info("💡 **指標の解説**\n- **タイム指数**: 走破タイム・ペース・馬場状態を補正し「50」を平均として算出した能力値。数値が高いほど優秀。\n- **タイム差**: 1着馬とのゴールタイム差（秒）。1着勝利時は「0.0」。")
                        import altair as alt
                        tab1, tab2, tab3, tab4 = st.tabs(["🚀 タイム指数", "💨 上がり3F & タイム差", "⚖️ 馬体重推移", "👑 着順・人気"])
                        with tab1:
                            st.markdown("※ 数値が高い（グラフが上に行く）ほど優秀なパフォーマンスです。")
                            if 'タイム指数' in chart_df.columns:
                                idx_data = chart_df[['タイム指数']].dropna().reset_index()
                                if not idx_data.empty:
                                    min_idx = idx_data['タイム指数'].min() - 5
                                    max_idx = idx_data['タイム指数'].max() + 5
                                    c1 = alt.Chart(idx_data).mark_line(point=True).encode(
                                        x=alt.X('日付:T', title='日付'),
                                        y=alt.Y('タイム指数:Q', scale=alt.Scale(domain=[min_idx, max_idx])),
                                        tooltip=['日付:T', 'タイム指数:Q']
                                    ).interactive()
                                    st.altair_chart(c1, use_container_width=True)
                                else: st.write("有効なタイム指数データがありません。")
                            else: st.write("データがありません。")
                        with tab2:
                            st.markdown("※ 数値が低いほど優秀です。Y軸を反転しています。")
                            if agari_col and agari_col in chart_df.columns:
                                agari_data = chart_df[[agari_col]].dropna().reset_index()
                                if not agari_data.empty:
                                    min_a = agari_data[agari_col].min() - 0.5
                                    max_a = agari_data[agari_col].max() + 0.5
                                    ca = alt.Chart(agari_data).mark_line(point=True, color='#FFA500').encode(
                                        x=alt.X('日付:T', title=''),
                                        y=alt.Y(f'{agari_col}:Q', scale=alt.Scale(domain=[min_a, max_a], reverse=True)),
                                        tooltip=['日付:T', f'{agari_col}:Q']
                                    ).interactive()
                                    st.altair_chart(ca, use_container_width=True)
                            if 'タイム差' in chart_df.columns:
                                td_data = chart_df[['タイム差']].dropna().reset_index()
                                if not td_data.empty:
                                    max_t = td_data['タイム差'].max() + 0.2
                                    ct = alt.Chart(td_data).mark_line(point=True, color='#FF4B4B').encode(
                                        x=alt.X('日付:T', title='日付'),
                                        y=alt.Y('タイム差:Q', scale=alt.Scale(domain=[0, max_t], reverse=True)),
                                        tooltip=['日付:T', 'タイム差:Q']
                                    ).interactive()
                                    st.altair_chart(ct, use_container_width=True)
                        with tab3:
                            st.markdown("※ 体重の増減を示します。")
                            if weight_col and weight_col in chart_df.columns:
                                weight_data = chart_df[[weight_col]].replace(0, np.nan).dropna().reset_index()
                                if not weight_data.empty:
                                    min_w = max(300, weight_data[weight_col].min() - 10)
                                    max_w = weight_data[weight_col].max() + 10
                                    cw = alt.Chart(weight_data).mark_line(point=True).encode(
                                        x=alt.X('日付:T', title='日付'),
                                        y=alt.Y(f'{weight_col}:Q', scale=alt.Scale(domain=[min_w, max_w])),
                                        tooltip=['日付:T', f'{weight_col}:Q']
                                    ).interactive()
                                    st.altair_chart(cw, use_container_width=True)
                                else: st.write("有効な馬体重データがありません。")
                            else: st.write("データがありません。")
                        with tab4:
                            st.markdown("※ 数値が低いほど上位です。")
                            rank_cols = [c for c in ['着順', '人気'] if c in chart_df.columns]
                            if rank_cols:
                                rank_data = chart_df[rank_cols].dropna().reset_index()
                                if not rank_data.empty:
                                    max_val = rank_data[rank_cols].max().max()
                                    max_scale = max(18, max_val + 1)
                                    melted = rank_data.melt('日付', value_vars=rank_cols, var_name='項目', value_name='順位')
                                    cr = alt.Chart(melted).mark_line(point=True).encode(
                                        x=alt.X('日付:T', title='日付'),
                                        y=alt.Y('順位:Q', scale=alt.Scale(domain=[1, max_scale], reverse=True)),
                                        color='項目:N',
                                        tooltip=['日付:T', '項目:N', '順位:Q']
                                    ).interactive()
                                    st.altair_chart(cr, use_container_width=True)
                                else: st.write("有効な着順・人気データがありません。")
                            else: st.write("データがありません。")
                        st.markdown("#### 📜 レース詳細データ")
                        display_cols_table = ['日付', 'レース名', '着順', '人気', '単勝', weight_col, '騎手', '通過', agari_col, 'タイム指数', 'タイム差']
                        if '単勝' in df_horse.columns:
                            df_horse = df_horse.rename(columns={'単勝': '単勝オッズ'})
                            display_cols_table = [c if c != '単勝' else '単勝オッズ' for c in display_cols_table]
                        show_cols = [c for c in display_cols_table if c and c in df_horse.columns]
                        show_df2 = df_horse.copy()
                        show_df2['日付'] = show_df2['日付'].dt.strftime('%Y/%m/%d')
                        for col in ['タイム指数', 'タイム差', '単勝オッズ', agari_col]:
                            if col and col in show_df2.columns:
                                show_df2[col] = pd.to_numeric(show_df2[col], errors='coerce').round(1)
                        st.dataframe(show_df2[show_cols].reset_index(drop=True), use_container_width=True)
            except Exception as e:
                st.error(f"データの読み込み中にエラーが発生しました: {e}")
