import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import requests
from bs4 import BeautifulSoup
import re
import datetime
import pytz
import traceback
import time
from sklearn.metrics import roc_auc_score, roc_curve 

st.set_page_config(page_title="keiba-ebye 予測ダッシュボード", page_icon="🐴", layout="wide")
st.title("🐴 keiba-ebye 予測ダッシュボード")
st.markdown("えーびーあい (ebi × AI × Eye) が、極限まで高められた精度でお宝馬を暴き出すかも？")

# ==========================================
# 1. 限界突破AIエンジンの学習とデータ準備
# ==========================================
@st.cache_resource
def prepare_model_and_data():
    try:
        df = pd.read_csv('learning_data_perfect_tier.zip', compression='zip', dtype=str)
    except FileNotFoundError:
        df = pd.read_csv('learning_data_perfect_tier.csv', dtype=str)

    df['日付'] = pd.to_datetime(df['日付'], format='mixed', errors='coerce')
    df = df.dropna(subset=['日付'])

    for col in ['着順', '単勝', '人気', '斤量', '距離', '上り', '枠番', '馬番']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df['性別'] = df['性齢'].astype(str).str.extract(r'([牡牝セ])')[0]
    df['年齢'] = pd.to_numeric(df['性齢'].astype(str).str.extract(r'(\d+)')[0], errors='coerce')
    df['馬体重_num'] = pd.to_numeric(df['馬体重'].astype(str).str.extract(r'(\d+)')[0], errors='coerce')

    def t2s(t):
        try:
            m = re.match(r'(\d+):(\d+\.\d+)', str(t))
            return float(m.group(1))*60 + float(m.group(2)) if m else float(t)
        except: return np.nan
    df['走破タイム秒'] = df['タイム'].apply(t2s)

    df['出走頭数'] = df.groupby('レースID')['馬ID'].transform('count')
    df['着順パーセント'] = (df['着順'] - 1) / (df['出走頭数'] - 1).replace(0, 1)
    
    race_min_times = df.groupby('レースID')['走破タイム秒'].transform('min')
    df['タイム差'] = df['走破タイム秒'] - race_min_times
    df['上がり順位'] = df.groupby('レースID')['上り'].rank(method='min')

    course_stats = df.groupby(['競馬場', '芝/ダート', '距離'])['走破タイム秒'].agg(['mean', 'std']).reset_index()
    course_stats.columns = ['競馬場', '芝/ダート', '距離', 'コース平均', 'コース標準偏差']
    df = pd.merge(df, course_stats, on=['競馬場', '芝/ダート', '距離'], how='left')
    df['スピード指数'] = np.where(df['コース標準偏差'] > 0, 50 - ((df['走破タイム秒'] - df['コース平均']) / df['コース標準偏差']) * 10, 50)

    df['調教師_騎手'] = df['調教師'].astype(str) + '_' + df['騎手'].astype(str)
    
    df = df.sort_values(['馬ID', '日付']).reset_index(drop=True)

    df['前走_着順'] = df.groupby('馬ID')['着順'].shift(1)
    df['2走前_着順'] = df.groupby('馬ID')['着順'].shift(2)
    df['3走前_着順'] = df.groupby('馬ID')['着順'].shift(3)
    df['過去3走平均着順'] = df[['前走_着順', '2走前_着順', '3走前_着順']].mean(axis=1)

    df['前走_着順パーセント'] = df.groupby('馬ID')['着順パーセント'].shift(1)
    df['2走前_着順パーセント'] = df.groupby('馬ID')['着順パーセント'].shift(2)
    df['3走前_着順パーセント'] = df.groupby('馬ID')['着順パーセント'].shift(3)
    df['過去3走平均着順パーセント'] = df[['前走_着順パーセント', '2走前_着順パーセント', '3走前_着順パーセント']].mean(axis=1)

    df['前走_タイム差'] = df.groupby('馬ID')['タイム差'].shift(1)
    df['2走前_タイム差'] = df.groupby('馬ID')['タイム差'].shift(2)
    df['3走前_タイム差'] = df.groupby('馬ID')['タイム差'].shift(3)
    df['過去3走平均タイム差'] = df[['前走_タイム差', '2走前_タイム差', '3走前_タイム差']].mean(axis=1)

    df['前走_上がり順位'] = df.groupby('馬ID')['上がり順位'].shift(1)

    df['前走_スピード指数'] = df.groupby('馬ID')['スピード指数'].shift(1)
    df['2走前_スピード指数'] = df.groupby('馬ID')['スピード指数'].shift(2)
    df['3走前_スピード指数'] = df.groupby('馬ID')['スピード指数'].shift(3)
    df['過去3走平均スピード指数'] = df[['前走_スピード指数', '2走前_スピード指数', '3走前_スピード指数']].mean(axis=1)

    # ディープ特徴量
    df['前走_通過'] = df.groupby('馬ID')['通過'].shift(1)
    df['2走前_通過'] = df.groupby('馬ID')['通過'].shift(2)
    df['3走前_通過'] = df.groupby('馬ID')['通過'].shift(3)
    
    df['前走_最終コーナー'] = pd.to_numeric(df['前走_通過'].fillna('').astype(str).apply(lambda x: str(x).split('-')[-1] if '-' in str(x) else (str(x) if str(x).isdigit() else np.nan)), errors='coerce')
    df['2走前_最終コーナー'] = pd.to_numeric(df['2走前_通過'].fillna('').astype(str).apply(lambda x: str(x).split('-')[-1] if '-' in str(x) else (str(x) if str(x).isdigit() else np.nan)), errors='coerce')
    df['3走前_最終コーナー'] = pd.to_numeric(df['3走前_通過'].fillna('').astype(str).apply(lambda x: str(x).split('-')[-1] if '-' in str(x) else (str(x) if str(x).isdigit() else np.nan)), errors='coerce')
    
    df['過去3走平均最終コーナー'] = df[['前走_最終コーナー', '2走前_最終コーナー', '3走前_最終コーナー']].mean(axis=1)
    
    def classify_style(pos):
        if pd.isna(pos): return '不明'
        if pos <= 2.5: return '逃げ'
        elif pos <= 5.5: return '先行'
        elif pos <= 9.5: return '差し'
        else: return '追込'
    df['脚質カテゴリ'] = df['過去3走平均最終コーナー'].apply(classify_style)

    df['前走逃げフラグ'] = (df['前走_最終コーナー'] <= 2).astype(int)
    df['前走先行フラグ'] = ((df['前走_最終コーナー'] > 2) & (df['前走_最終コーナー'] <= 5)).astype(int)
    df['同レース逃げ馬頭数'] = df.groupby('レースID')['前走逃げフラグ'].transform('sum')
    df['同レース先行馬頭数'] = df.groupby('レースID')['前走先行フラグ'].transform('sum')

    df['コース適性_着順パーセント'] = df.groupby(['馬ID', '競馬場', '芝/ダート'])['着順パーセント'].transform(lambda x: x.shift(1).expanding().mean()).fillna(0.5)
    df['位置取りショック'] = df['前走_最終コーナー'] - df['2走前_最終コーナー']

    df['前走_人気'] = df.groupby('馬ID')['人気'].shift(1)
    df['前走_上り'] = df.groupby('馬ID')['上り'].shift(1)
    df['前走_距離'] = df.groupby('馬ID')['距離'].shift(1)
    df['距離変化'] = df['距離'] - df['前走_距離']
    df['前走_斤量'] = df.groupby('馬ID')['斤量'].shift(1)
    df['斤量変化'] = df['斤量'] - df['前走_斤量']
    df['前走_馬体重'] = df.groupby('馬ID')['馬体重_num'].shift(1)
    df['馬体重変化'] = df['馬体重_num'] - df['前走_馬体重']
    df['前走_日付'] = df.groupby('馬ID')['日付'].shift(1)
    df['休養日数'] = (df['日付'] - df['前走_日付']).dt.days

    df_latest = df.groupby('馬ID').tail(1).copy()
    rename_dict = {
        '着順': '最新_着順', '着順パーセント': '最新_着順パーセント', 'タイム差': '最新_タイム差',
        'スピード指数': '最新_スピード指数', '上がり順位': '最新_上がり順位', '人気': '最新_人気', 
        '上り': '最新_上り', '距離': '最新_距離', '斤量': '最新_斤量', '馬体重_num': '最新_馬体重',
        '日付': '最新_日付', '通過': '最新_通過'
    }
    df_latest = df_latest.rename(columns=rename_dict)
    cols_to_keep = [
        '馬ID', '父', '父系', '母', '母系', '母父', '母父系',
        '最新_着順', '最新_着順パーセント', '最新_タイム差', '最新_スピード指数', '最新_上がり順位', 
        '最新_人気', '最新_上り', '最新_距離', '最新_斤量', '最新_馬体重', '最新_日付', '最新_通過',
        '前走_着順', '2走前_着順', '前走_着順パーセント', '2走前_着パーセント', 
        '前走_タイム差', '2走前_タイム差', '前走_スピード指数', '2走前_スピード指数',
        '前走_通過', '2走前_通過'
    ]
    # 足りない列は無視して抽出
    cols_to_keep = [c for c in cols_to_keep if c in df_latest.columns]
    latest_horse_data = df_latest[cols_to_keep].copy()

    horse_course_dict = df.groupby(['馬ID', '競馬場', '芝/ダート'])['着順パーセント'].mean().to_dict()

    # 💡 【重要修正】初出走馬（前走_着順がNaNの馬）を学習データから除外しない！
    df_valid = df.dropna(subset=['着順', '単勝']).copy()
    df_valid['馬券内'] = (df_valid['着順'] <= 3).astype(int)

    num_features = [
        '枠番', '馬番', '年齢', '馬体重_num', '距離', '斤量', '休養日数', 
        '前走_着順', '2走前_着順', '3走前_着順', '過去3走平均着順', 
        '前走_着順パーセント', '過去3走平均着順パーセント',
        '前走_タイム差', '過去3走平均タイム差',
        '前走_スピード指数', '2走前_スピード指数', '3走前_スピード指数', '過去3走平均スピード指数',
        '前走_人気', '前走_上り', '前走_上がり順位', '前走_最終コーナー',
        '距離変化', '斤量変化', '馬体重変化', '出走頭数',
        '位置取りショック', '同レース逃げ馬頭数', '同レース先行馬頭数', 'コース適性_着順パーセント' 
    ]
    cat_features = ['競馬場', '馬場', '芝/ダート', '性別', '脚質カテゴリ', '父', '父系', '母', '母系', '母父', '母父系', '騎手', '調教師', '調教師_騎手'] 
    features = cat_features + num_features

    cat_categories_dict = {}
    for col in cat_features:
        if col not in df_valid.columns: df_valid[col] = '不明'
        df_valid[col] = df_valid[col].fillna('不明').astype('category')
        cat_categories_dict[col] = list(df_valid[col].cat.categories)

    split_date = pd.to_datetime('2025-01-01')
    train_df = df_valid[df_valid['日付'] < split_date].copy()
    test_df = df_valid[df_valid['日付'] >= split_date].copy()

    model = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.01, num_leaves=63, max_bin=255, cat_smooth=10,
        random_state=42, importance_type='gain', colsample_bytree=0.7, subsample=0.8
    )
    model.fit(train_df[features], train_df['馬券内'], categorical_feature=cat_features)
    
    val_preds = model.predict_proba(test_df[features])[:, 1]
    val_auc = roc_auc_score(test_df['馬券内'], val_preds)
    fpr, tpr, _ = roc_curve(test_df['馬券内'], val_preds)

    # 💡 【重要修正】未出走馬のために血統マスターを読み込んでおく
    try:
        ped_df = pd.read_csv('pedigree_master_all.csv', dtype=str)
        ped_df['馬ID'] = ped_df['馬ID'].astype(str).str.zfill(10)
        ped_dict = ped_df.set_index('馬ID')[['父', '父系', '母', '母系', '母父', '母父系']].to_dict('index')
    except:
        ped_dict = {}

    return model, features, cat_features, num_features, cat_categories_dict, latest_horse_data, horse_course_dict, ped_dict, val_auc, fpr, tpr

with st.spinner('keiba-ebye フルパワーAIエンジンを起動・学習中... (初回のみ数分かかります)'):
    model, features, cat_features, num_features, cat_categories_dict, latest_horse_data, horse_course_dict, ped_dict, val_auc, fpr, tpr = prepare_model_and_data()

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
            res = requests.get(url, headers=headers, timeout=10); res.encoding = 'euc-jp'
            soup = BeautifulSoup(res.text, 'html.parser')
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
            res = requests.get(url, headers=headers, timeout=10); res.encoding = 'euc-jp'
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

def get_odds_from_soup(s_soup):
    o_dict = {}
    tgt_table = s_soup.select_one('.Shutuba_Table') or s_soup.select_one('.RaceTable01') or s_soup.select_one('.race_table_01') or s_soup.select_one('#All_Result_Table')
    if not tgt_table: return o_dict
    u_idx, o_idx = -1, -1
    for i, th in enumerate(tgt_table.find_all('th')):
        c_txt = re.sub(r'\s+', '', th.text)
        if '馬番' in c_txt: u_idx = i
        if '単勝' in c_txt or 'オッズ' in c_txt or '予想' in c_txt: o_idx = i
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
                o_m = re.search(r'\d+\.\d+', tds[o_idx].text)
                if o_m: odds_val = float(o_m.group(0))
            if odds_val == 0.0:
                for td in tds:
                    if any(c in ['Odds', 'Popular', 'txt_r', 'Txt_R'] for c in td.get('class', [])):
                        o_m = re.search(r'\d+\.\d+', td.text)
                        if o_m: odds_val = float(o_m.group(0)); break
            if odds_val > 0.0: o_dict[umaban] = odds_val
    except: pass
    return o_dict

def generate_txt_report(results_list):
    txt = "=== 🏇 keiba-ebye 予想レポート ===\n\n"
    for r in results_list:
        txt += "="*50 + "\n"
        txt += f"■ {r['date']} | {r['place']} {r['num']}R ({r['track']}{r['dist']}m) ■\n"
        txt += f"🐎 【展開予想】\n{r['pace']}\n"
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
    
    for fetch_url in [
        f'https://race.netkeiba.com/race/shutuba.html?race_id={race_id}',
        f'https://race.netkeiba.com/race/result.html?race_id={race_id}',
        f'https://db.netkeiba.com/race/{race_id}/'
    ]:
        try:
            r = requests.get(fetch_url, headers=headers, timeout=10); r.encoding = 'euc-jp'
            soup = BeautifulSoup(r.text, 'html.parser')
            if soup.select_one('.Shutuba_Table') or soup.select_one('.RaceTable01') or soup.select_one('.race_table_01') or soup.select_one('#All_Result_Table'):
                if not html_text: html_text = r.text 
                temp_odds = get_odds_from_soup(soup)
                if temp_odds: html_text = r.text; odds_dict = temp_odds; break 
        except Exception as e: pass

    if not html_text: return None, None, None, None, None, None, None, ["❌ 出馬表が取得できませんでした。"]
    soup = BeautifulSoup(html_text, 'html.parser')
    race_data_box = soup.find('div', class_='RaceData01') or soup.find('dl', class_='racedata')
    if not race_data_box: return None, None, None, None, None, None, None, ["❌ レース条件が見つかりません。"]

    race_text = race_data_box.text.replace('\n', '')
    baba_match = re.search(r'馬場:([良稍重不良]+)', race_text)
    todays_baba = baba_match.group(1) if baba_match else '良'
    track_dist_match = re.search(r'(芝|ダ|障|障害).*?(\d+)m', race_text)
    track_type = "芝" if track_dist_match and track_dist_match.group(1) == "芝" else "ダート" if track_dist_match and "ダ" in track_dist_match.group(1) else "障害"
    distance = float(track_dist_match.group(2)) if track_dist_match else 1600.0
    place = {'01':'札幌','02':'函館','03':'福島','04':'新潟','05':'東京','06':'中山','07':'中京','08':'京都','09':'阪神','10':'小倉'}.get(str(race_id)[4:6], '東京')

    horses = []
    table = soup.select_one('.Shutuba_Table') or soup.select_one('.RaceTable01') or soup.select_one('.race_table_01') or soup.select_one('#All_Result_Table')
    if not table: return None, None, None, None, None, None, None, ["❌ 出走馬の一覧表が見つかりません。"]

    ths = table.find_all('th')
    headers_text = [th.text.strip().replace('\n', '') for th in ths]
    def get_idx(keywords):
        for i, h in enumerate(headers_text):
            for kw in keywords:
                if kw in h: return i
        return -1

    waku_idx, uma_idx, kinryo_idx, weight_idx, odds_idx, sex_age_idx = get_idx(['枠']), get_idx(['馬番']), get_idx(['斤量']), get_idx(['馬体重']), get_idx(['単勝', 'オッズ', '予想']), get_idx(['性齢'])

    for tr in table.find_all('tr')[1:]: 
        tds = tr.find_all('td')
        if len(tds) < 5: continue
        try:
            if uma_idx == -1 or len(tds) <= uma_idx or not re.search(r'\d+', tds[uma_idx].text): continue
            umaban = int(re.search(r'\d+', tds[uma_idx].text).group(0))
            waku = int(re.search(r'\d+', tds[waku_idx].text).group(0)) if waku_idx != -1 and len(tds) > waku_idx and re.search(r'\d+', tds[waku_idx].text) else 0
            
            horse_a = tr.find('a', href=re.compile(r'/horse/'))
            if not horse_a: continue
            horse_id = re.search(r'\d+', horse_a['href']).group(0)
            
            jockey_a = tr.find('a', href=re.compile(r'/jockey/'))
            jockey_name = jockey_a.text.strip() if jockey_a else "不明"
            
            trainer_a = tr.find('a', href=re.compile(r'/trainer/'))
            trainer_name = trainer_a.text.strip() if trainer_a else "不明"
            
            kinryo_text = tds[kinryo_idx].text if kinryo_idx != -1 and len(tds) > kinryo_idx else "55.0"
            kinryo_match = re.search(r'\d+(\.\d+)?', kinryo_text)
            kinryo = float(kinryo_match.group(0)) if kinryo_match else 55.0
            
            weight_text = tds[weight_idx].text if weight_idx != -1 and len(tds) > weight_idx else ""
            weight_match = re.search(r'^(\d{3})', weight_text.strip())
            weight_val = float(weight_match.group(1)) if weight_match else np.nan
            
            odds_val = odds_dict.get(umaban, 0.0) 
            if odds_val == 0.0 and odds_idx != -1 and len(tds) > odds_idx:
                odds_match = re.search(r'\d+\.\d+', tds[odds_idx].text)
                if odds_match: odds_val = float(odds_match.group(0))
            if odds_val == 0.0: odds_val = 10.0
            
            sex_age = tds[sex_age_idx].text.strip() if sex_age_idx != -1 and len(tds) > sex_age_idx else "牡3"

            horses.append({'枠番': waku, '馬番': umaban, '馬名': horse_a.text.strip(), '馬ID': horse_id, '性齢': sex_age, '斤量': kinryo, '騎手': jockey_name, '調教師': trainer_name, '距離': distance, '競馬場': place, '芝/ダート': track_type, '馬場': todays_baba, '馬体重_num': weight_val, '単勝オッズ': odds_val})
        except: pass

    if not horses: return None, None, None, None, None, None, None, ["❌ 出走馬データの読み取りに失敗しました。"]

    try:
        df_test = pd.DataFrame(horses)
        df_test['出走頭数'] = len(df_test)
        df_test = pd.merge(df_test, latest_horse_data, on='馬ID', how='left')

        # 💡 【重要】新馬・未出走馬のための「血統完全補完」ロジック！
        for col in ['父', '父系', '母', '母系', '母父', '母父系']:
            if col not in df_test.columns: df_test[col] = np.nan

        for i, row in df_test.iterrows():
            hid = row['馬ID']
            if pd.isna(row['父']) or row['父'] == '不明':
                if hid in ped_dict:
                    for col in ['父', '父系', '母', '母系', '母父', '母父系']:
                        df_test.at[i, col] = ped_dict[hid].get(col, '不明')
                else:
                    for col in ['父', '父系', '母', '母系', '母父', '母父系']:
                        df_test.at[i, col] = '不明'

        df_test['性別'] = df_test['性齢'].astype(str).str.extract(r'([牡牝セ])')[0]
        df_test['年齢'] = pd.to_numeric(df_test['性齢'].astype(str).str.extract(r'(\d+)')[0], errors='coerce')
        df_test['調教師_騎手'] = df_test['調教師'].astype(str) + '_' + df_test['騎手'].astype(str)

        df_test['3走前_着順'] = df_test['2走前_着順'] if '2走前_着順' in df_test.columns else np.nan
        df_test['2走前_着順'] = df_test['前走_着順'] if '前走_着順' in df_test.columns else np.nan
        df_test['前走_着順'] = df_test['最新_着順'] if '最新_着順' in df_test.columns else np.nan
        df_test['過去3走平均着順'] = df_test[['前走_着順', '2走前_着順', '3走前_着順']].mean(axis=1)

        df_test['3走前_着順パーセント'] = df_test['2走前_着順パーセント'] if '2走前_着順パーセント' in df_test.columns else np.nan
        df_test['2走前_着順パーセント'] = df_test['前走_着順パーセント'] if '前走_着順パーセント' in df_test.columns else np.nan
        df_test['前走_着順パーセント'] = df_test['最新_着順パーセント'] if '最新_着順パーセント' in df_test.columns else np.nan
        df_test['過去3走平均着順パーセント'] = df_test[['前走_着順パーセント', '2走前_着順パーセント', '3走前_着順パーセント']].mean(axis=1)

        df_test['3走前_タイム差'] = df_test['2走前_タイム差'] if '2走前_タイム差' in df_test.columns else np.nan
        df_test['2走前_タイム差'] = df_test['前走_タイム差'] if '前走_タイム差' in df_test.columns else np.nan
        df_test['前走_タイム差'] = df_test['最新_タイム差'] if '最新_タイム差' in df_test.columns else np.nan
        df_test['過去3走平均タイム差'] = df_test[['前走_タイム差', '2走前_タイム差', '3走前_タイム差']].mean(axis=1)

        df_test['3走前_スピード指数'] = df_test['2走前_スピード指数'] if '2走前_スピード指数' in df_test.columns else np.nan
        df_test['2走前_スピード指数'] = df_test['前走_スピード指数'] if '前走_スピード指数' in df_test.columns else np.nan
        df_test['前走_スピード指数'] = df_test['最新_スピード指数'] if '最新_スピード指数' in df_test.columns else np.nan
        df_test['過去3走平均スピード指数'] = df_test[['前走_スピード指数', '2走前_スピード指数', '3走前_スピード指数']].mean(axis=1)

        df_test['3走前_通過'] = df_test['2走前_通過'] if '2走前_通過' in df_test.columns else np.nan
        df_test['2走前_通過'] = df_test['前走_通過'] if '前走_通過' in df_test.columns else np.nan
        df_test['前走_通過'] = df_test['最新_通過'] if '最新_通過' in df_test.columns else np.nan

        df_test['前走_最終コーナー'] = pd.to_numeric(df_test['前走_通過'].fillna('').astype(str).apply(lambda x: str(x).split('-')[-1] if '-' in str(x) else (str(x) if str(x).isdigit() else np.nan)), errors='coerce')
        df_test['2走前_最終コーナー'] = pd.to_numeric(df_test['2走前_通過'].fillna('').astype(str).apply(lambda x: str(x).split('-')[-1] if '-' in str(x) else (str(x) if str(x).isdigit() else np.nan)), errors='coerce')
        df_test['3走前_最終コーナー'] = pd.to_numeric(df_test['3走前_通過'].fillna('').astype(str).apply(lambda x: str(x).split('-')[-1] if '-' in str(x) else (str(x) if str(x).isdigit() else np.nan)), errors='coerce')
        
        df_test['過去3走平均最終コーナー'] = df_test[['前走_最終コーナー', '2走前_最終コーナー', '3走前_最終コーナー']].mean(axis=1)
        
        def classify_style(pos):
            if pd.isna(pos): return '不明'
            if pos <= 2.5: return '逃げ'
            elif pos <= 5.5: return '先行'
            elif pos <= 9.5: return '差し'
            else: return '追込'
        df_test['脚質カテゴリ'] = df_test['過去3走平均最終コーナー'].apply(classify_style)
        
        df_test['前走逃げフラグ'] = (df_test['前走_最終コーナー'] <= 2).astype(int)
        df_test['前走先行フラグ'] = ((df_test['前走_最終コーナー'] > 2) & (df_test['前走_最終コーナー'] <= 5)).astype(int)
        df_test['同レース逃げ馬頭数'] = df_test['前走逃げフラグ'].sum()
        df_test['同レース先行馬頭数'] = df_test['前走先行フラグ'].sum()
        
        df_test['コース適性_着順パーセント'] = df_test.set_index(['馬ID', '競馬場', '芝/ダート']).index.map(horse_course_dict).fillna(0.5)
        df_test['位置取りショック'] = df_test['前走_最終コーナー'] - df_test['2走前_最終コーナー']

        df_test['前走_上がり順位'] = df_test['最新_上がり順位'] if '最新_上がり順位' in df_test.columns else np.nan
        df_test['前走_人気'] = df_test['最新_人気'] if '最新_人気' in df_test.columns else np.nan
        df_test['前走_上り'] = df_test['最新_上り'] if '最新_上り' in df_test.columns else np.nan
        
        df_test['距離変化'] = df_test['距離'] - (df_test['最新_距離'] if '最新_距離' in df_test.columns else df_test['距離'])
        df_test['斤量変化'] = df_test['斤量'] - (df_test['最新_斤量'] if '最新_斤量' in df_test.columns else df_test['斤量'])
        df_test['馬体重変化'] = df_test['馬体重_num'] - (df_test['最新_馬体重'] if '最新_馬体重' in df_test.columns else df_test['馬体重_num'])
        
        race_date_obj = pd.to_datetime(race_date_str)
        if '最新_日付' in df_test.columns:
            df_test['休養日数'] = (race_date_obj - pd.to_datetime(df_test['最新_日付'])).dt.days
        else:
            df_test['休養日数'] = np.nan

        for col in num_features: df_test[col] = pd.to_numeric(df_test[col], errors='coerce')
        for col in cat_features:
            if col not in df_test.columns: df_test[col] = '不明'
            cats = cat_categories_dict.get(col, ['不明'])
            df_test[col] = pd.Categorical(df_test[col].fillna('不明'), categories=cats)

        nige_count = df_test['同レース逃げ馬頭数'].iloc[0] if not df_test.empty else 0
        senko_count = df_test['同レース先行馬頭数'].iloc[0] if not df_test.empty else 0
        
        if nige_count >= 3: pace_text = f"🔥 【ハイペース濃厚】 前走で逃げた馬が{nige_count}頭もおり先行争いが激化。差し・追込馬の台頭に警戒！"
        elif nige_count == 0: pace_text = f"🐌 【スローペース濃厚】 確たる逃げ馬が不在。先行馬({senko_count}頭)の押し切り、前残りに注意。"
        else: pace_text = f"🐎 【ミドルペース】 逃げ馬{nige_count}頭、先行馬{senko_count}頭。平均的なペースで実力が反映されやすい展開。"

        raw_probs = model.predict_proba(df_test[features])[:, 1]
        df_test['複勝率(AI予測)'] = raw_probs
        
        probs = raw_probs ** 1.2
        probs = np.maximum(probs, 0.001)
        df_test['勝率(AI予測)'] = probs / probs.sum()
        
        df_test['期待値'] = df_test['勝率(AI予測)'] * df_test['単勝オッズ']
        df_test = df_test.sort_values('勝率(AI予測)', ascending=False).reset_index(drop=True)

        marks = ['◎', '〇', '▲', '△', '☆'] + [''] * (len(df_test) - 5)
        df_test['印'] = marks[:len(df_test)]

        ana_horse_nums = []
        topics_list = []
        for rank, row in df_test.iterrows():
            if rank >= 5:
                if row['期待値'] >= 1.5:
                    topics_list.append(f"📌 {row['馬名']} (期待値特大の穴馬！)")
                    if f"{row['馬番']}番" not in ana_horse_nums: ana_horse_nums.append(f"{row['馬番']}番")

        ana_str = "・".join(str(n) for n in ana_horse_nums[:3]) if ana_horse_nums else ""
        p1, p2 = df_test.loc[0, '勝率(AI予測)'], df_test.loc[1, '勝率(AI予測)']
        if p1 >= 0.20: reco = f"🎯 馬連・馬単: ◎から印馬・穴馬({ana_str})への流し"
        elif p1 >= 0.12 and (p1 - p2) >= 0.03: reco = f"🎯 馬連・ワイド: ◎から穴馬({ana_str})への流しで高配当狙い"
        else: reco = f"⚠️ 上位評価割れ。印馬と穴馬({ana_str})のボックス推奨"

        return df_test, topics_list, reco, pace_text, track_type, place, distance, error_log

    except Exception as e:
        error_log.append(f"❌ 予測AI内部で致命的なエラーが発生: {traceback.format_exc()}")
        return None, None, None, None, None, None, None, error_log

# ==========================================
# 4. メインUI構成
# ==========================================
st.sidebar.markdown("## 🕹️ keiba-ebye メニュー")
action = st.sidebar.radio("機能を選択", [
    "⏩ 次のレースを予想", 
    "📜 本日の全レース予想", 
    "📅 今週末の全レース予想", 
    "🔍 レースを指定して予想", 
    "🧪 性能試験 (バックテスト)",
    "📈 AI精度評価 (AUCスコア)"
])

tokyo_tz = pytz.timezone('Asia/Tokyo')
now = datetime.datetime.now(tokyo_tz)

def display_error_log(err_log):
    st.error("⚠️ 予想データまたは結果の取得に失敗しました。")
    with st.expander("🔍 エラー解析ログを見る (デバッグ用)"):
        for log in err_log: st.write(f"- {log}")

def display_result(df_res, topics, reco, pace_text):
    # 💡 【UI進化】第3のタブ「性能詳細」を追加！
    tab1, tab2, tab3 = st.tabs(["📊 予想一覧", "💡 買い目・展開", "🔍 性能詳細"])
    
    with tab1:
        def highlight_ev(row): return ['background-color: rgba(255, 99, 71, 0.3)' if row['期待値'] >= 1.5 else '' for _ in row]
        show_df = df_res[['印', '馬番', '馬名', '脚質カテゴリ', '単勝オッズ', '勝率(AI予測)', '複勝率(AI予測)', '期待値']].copy()
        show_df = show_df.rename(columns={'勝率(AI予測)': '勝率', '複勝率(AI予測)': '複勝率', '単勝オッズ': 'オッズ', '脚質カテゴリ': '脚質'})
        show_df['勝率'] = (show_df['勝率'] * 100).map('{:.1f}%'.format)
        show_df['複勝率'] = (show_df['複勝率'] * 100).map('{:.1f}%'.format)
        st.dataframe(show_df.style.apply(highlight_ev, axis=1).format({'期待値': '{:.2f}'}), use_container_width=True, hide_index=True)
        
    with tab2:
        st.info(f"**🏇 展開予想:**\n{pace_text}")
        ev_horses = df_res[(df_res.index < 5) & (df_res['期待値'] >= 1.5)]
        if not ev_horses.empty: st.error(f"💰 **【期待値レーダー発動】** {', '.join(ev_horses['馬名'].tolist())} に強烈なオッズ妙味あり！")
        if topics: st.warning("**📝 要注目トピック馬:**\n\n" + "\n".join(topics))
        st.success(f"**🤖 AI推奨買い目:**\n\n{reco}")
        
    with tab3:
        # 💡 各馬の性能詳細（マニアックデータ）を一覧表示！
        detail_df = df_res[['馬番', '馬名', '父', '母父', '騎手', '調教師', '過去3走平均スピード指数', 'コース適性_着順パーセント', '位置取りショック']].copy()
        detail_df = detail_df.rename(columns={'コース適性_着順パーセント': 'コース適性(%)', '過去3走平均スピード指数': '平均スピード指数'})
        detail_df['平均スピード指数'] = detail_df['平均スピード指数'].fillna(0)
        detail_df['位置取りショック'] = detail_df['位置取りショック'].fillna(0)
        st.markdown("※『コース適性(%)』は数字が低い（0に近い）ほどそのコースが得意なことを示します。")
        st.dataframe(detail_df.style.format({
            '平均スピード指数': '{:.1f}', 
            'コース適性(%)': '{:.2f}',
            '位置取りショック': '{:.1f}'
        }), use_container_width=True, hide_index=True)

if action in ["⏩ 次のレースを予想", "📜 本日の全レース予想", "🔍 レースを指定して予想"]:
    todays_races = get_todays_races()
    if not todays_races: st.warning(f"本日 ({now.strftime('%Y/%m/%d')}) はJRAのレースが開催されていません。")
    else:
        if action == "⏩ 次のレースを予想":
            st.subheader("🕒 まもなく出走するレース")
            next_race = next((r for r in todays_races if tokyo_tz.localize(r['time']) > now), None)
            if next_race:
                mins_left = int((tokyo_tz.localize(next_race['time']) - now).total_seconds() / 60)
                st.info(f"👉 **{next_race['place']} {next_race['num']}R** 「{next_race['title']}」 (あと **{mins_left}** 分)")
                if st.button("🚀 keiba-ebye 予想起動！", type="primary"):
                    with st.spinner('AIが推論中...'):
                        res_df, topics, reco, pace_text, _, _, _, err_log = run_real_prediction(next_race['id'], now.strftime('%Y-%m-%d'))
                        if res_df is not None: display_result(res_df, topics, reco, pace_text)
                        else: display_error_log(err_log)
            else: st.success("🏁 本日の全レースは終了しました。")
        elif action == "📜 本日の全レース予想":
            st.subheader(f"📅 本日の全レース一覧")
            if st.button("🚀 全レース一括予想", type="primary"):
                my_bar = st.progress(0, text="推論中...")
                results_for_txt = []
                for i, r in enumerate(todays_races):
                    st.markdown(f"#### ■ {r['place']} {r['num']}R")
                    res_df, topics, reco, pace_text, track_type, place, dist, err_log = run_real_prediction(r['id'], now.strftime('%Y-%m-%d'))
                    if res_df is not None:
                        display_result(res_df, topics, reco, pace_text)
                        results_for_txt.append({'date': now.strftime('%Y年%m月%d日'), 'place': place, 'num': r['num'], 'track': track_type, 'dist': dist, 'pace': pace_text, 'df': res_df, 'topics': topics, 'reco': reco})
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
                    res_df, topics, reco, pace_text, _, _, _, err_log = run_real_prediction(target_race['id'], now.strftime('%Y-%m-%d'))
                    if res_df is not None: display_result(res_df, topics, reco, pace_text)
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
                    res_df, topics, reco, pace_text, track_type, place, dist, err_log = run_real_prediction(r['id'], f"{target_date[:4]}-{target_date[4:6]}-{target_date[6:]}")
                    if res_df is not None:
                        display_result(res_df, topics, reco, pace_text)
                        results_for_txt.append({'date': f"{target_date[:4]}年{target_date[4:6]}月{target_date[6:]}日", 'place': place, 'num': r['num'], 'track': track_type, 'dist': dist, 'pace': pace_text, 'df': res_df, 'topics': topics, 'reco': reco})
                    else: display_error_log(err_log)
                time.sleep(1.0)
                my_bar.progress((i + 1) / len(target_races))
            if results_for_txt:
                st.download_button(f"📥 {target_date[4:6]}/{target_date[6:]} 予想レポート(.txt)", data=generate_txt_report(results_for_txt), file_name=f"keiba_weekend_{target_date}.txt", mime="text/plain")

elif action == "🧪 性能試験 (バックテスト)":
    test_date = st.date_input("テストする日付を選択", datetime.date.today() - datetime.timedelta(days=3))
    if st.button("🔥 バックテスト実行！", type="primary"):
        with st.spinner(f'全レースを推論・集計中...'):
            test_races = get_todays_races(test_date.strftime('%Y%m%d'))
            if not test_races: st.error("レースが見つかりません。")
            else:
                my_bar = st.progress(0, text="集計中...")
                total_invest, total_return_t, total_return_f, ev_hits = 0, 0, 0, 0
                results_for_txt = []
                for i, r in enumerate(test_races):
                    with st.expander(f"🏁 {r['place']} {r['num']}R"):
                        res_df, topics, reco, pace_text, track_type, place, dist, err_log = run_real_prediction(r['id'], test_date.strftime('%Y-%m-%d'))
                        t_dict, f_dict = get_payouts(r['id'])
                        if res_df is not None and t_dict:
                            display_result(res_df, topics, reco, pace_text)
                            results_for_txt.append({'date': test_date.strftime('%Y年%m月%d日'), 'place': place, 'num': r['num'], 'track': track_type, 'dist': dist, 'pace': pace_text, 'df': res_df, 'topics': topics, 'reco': reco})
                            for _, horse in res_df[(res_df.index < 5) & (res_df['期待値'] >= 1.5)].iterrows():
                                total_invest += 100
                                if horse['馬番'] in t_dict: total_return_t += t_dict[horse['馬番']]
                                if horse['馬番'] in f_dict: total_return_f += f_dict[horse['馬番']]; ev_hits += 1
                        else: display_error_log(err_log)
                    time.sleep(1.0)
                    my_bar.progress((i + 1) / len(test_races))
                st.markdown("---")
                st.markdown("### 🏆 バックテスト 集計レポート")
                if total_invest > 0:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("総投資額", f"¥{total_invest:,}")
                    c2.metric("単勝 回収率", f"{(total_return_t / total_invest * 100):.1f}%", f"¥{total_return_t:,}")
                    c3.metric("複勝 回収率", f"{(total_return_f / total_invest * 100):.1f}%", f"的中 {ev_hits}回")
                if results_for_txt:
                    st.download_button("📥 結果をダウンロード (.txt)", data=generate_txt_report(results_for_txt), file_name=f"keiba_backtest_{test_date.strftime('%Y%m%d')}.txt", mime="text/plain")

elif action == "📈 AI精度評価 (AUCスコア)":
    st.metric(label="📊 総合AUCスコア", value=f"{val_auc:.4f}")
    if val_auc > 0.75: st.success("🔥 限界突破！0.75を超えた超絶精度です！")
    st.line_chart(pd.DataFrame({'False Positive Rate': fpr, 'True Positive Rate': tpr}), x='False Positive Rate', y='True Positive Rate')