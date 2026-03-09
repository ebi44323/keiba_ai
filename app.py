import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import requests
from bs4 import BeautifulSoup
import re
import datetime
import pytz
import time
from sklearn.metrics import roc_auc_score, roc_curve

# ==========================================
# UI・ページ設定
# ==========================================
st.set_page_config(page_title="keiba-ebye 予測ダッシュボード", page_icon="🐴", layout="wide")

st.title("🐴 keiba-ebye 予測ダッシュボード")
st.markdown("独自のデータアナリティクスとAI (ebi × AI × Eye) が、期待値に隠されたお宝馬を暴き出します。")

# ==========================================
# 1. 本格AIエンジンの学習とデータ準備 (キャッシュ化で高速化)
# ==========================================
@st.cache_resource
def prepare_model_and_data():
    df = pd.read_csv('learning_data_perfect_tier.zip', compression='zip', dtype={'馬ID': str, '騎手ID': str, 'レースID': str})
    df['日付'] = pd.to_datetime(df['日付'])

    df['上り'] = pd.to_numeric(df['上り'], errors='coerce')
    df['後半ペース値'] = pd.to_numeric(df['後半ペース値'], errors='coerce')
    df['上がり順位'] = df.groupby('レースID')['上り'].rank(method='min')
    df['上りレース差'] = df['上り'] - df['後半ペース値']

    df = df.sort_values(['馬ID', '日付'])
    if '通過' in df.columns:
        df['最初のコーナー順位'] = df['通過'].astype(str).str.split('-').str[0]
    df['最初のコーナー順位'] = pd.to_numeric(df['最初のコーナー順位'], errors='coerce')

    # 💡 【進化1】 脚質ロジックの抜本的改革（逃げ・自在の追加）
    horse_pos_stats = df.groupby('馬ID')['最初のコーナー順位'].agg(['mean', 'std']).reset_index()
    def classify_style(row):
        if pd.isna(row['mean']): return '不明'
        if pd.notna(row['std']) and row['std'] > 3.0: return '自在'
        if row['mean'] <= 1.5: return '逃げ'
        elif row['mean'] <= 4.0: return '先行'
        elif row['mean'] <= 9.0: return '差し'
        else: return '追込'
    horse_pos_stats['脚質カテゴリ'] = horse_pos_stats.apply(classify_style, axis=1)
    horse_style_dict = horse_pos_stats.set_index('馬ID')['脚質カテゴリ'].to_dict()
    df['脚質カテゴリ'] = df['馬ID'].map(horse_style_dict).fillna('不明')

    df_latest = df.groupby('馬ID').tail(1).copy()
    last_race_cols = {
        '着順': '前走着順', '着順パーセント': '前走着順パーセント', '距離補正タイム差': '前走距離補正タイム差',
        '最初のコーナー順位': '前走コーナー順位', '距離': '前走距離', '芝/ダート': '前走芝ダート',
        '騎手ID': '前走騎手ID', '日付': '前走日付', '斤量': '前走斤量',
        '上がり順位': '前走上がり順位', '上りレース差': '前走上りレース差'
    }
    df_latest_clean = df_latest[['馬ID', '父', '母父'] + list(last_race_cols.keys())].rename(columns=last_race_cols)
    df_latest_clean = df_latest_clean.loc[:, ~df_latest_clean.columns.duplicated()]

    df['前走日付'] = df.groupby('馬ID')['日付'].shift(1)
    df['出走間隔'] = (df['日付'] - df['前走日付']).dt.days.fillna(30)
    df['間隔カテゴリ'] = pd.cut(df['出走間隔'], bins=[-1, 14, 30, 9999], labels=['詰合', '標準', '休明']).astype(str)
    df['調教師_間隔'] = df['調教師'].astype(str) + '_' + df['間隔カテゴリ']
    df['調教師_騎手'] = df['調教師'].astype(str) + '_' + df['騎手ID'].astype(str)
    df['騎手_競馬場'] = df['騎手ID'].astype(str) + '_' + df['競馬場'].astype(str)
    df['騎手_距離'] = df['騎手ID'].astype(str) + '_' + df['距離'].astype(str)

    df['前走着順'] = df.groupby('馬ID')['着順'].shift(1)
    df['前走着順パーセント'] = df.groupby('馬ID')['着順パーセント'].shift(1)
    df['前走距離補正タイム差'] = df.groupby('馬ID')['距離補正タイム差'].shift(1)
    df['前走距離'] = df.groupby('馬ID')['距離'].shift(1)
    df['前走芝ダート'] = df.groupby('馬ID')['芝/ダート'].shift(1)
    df['前走騎手ID'] = df.groupby('馬ID')['騎手ID'].shift(1)
    df['前走斤量'] = df.groupby('馬ID')['斤量'].shift(1)
    df['斤量増減'] = (df['斤量'] - df['前走斤量']).fillna(0)
    df['前走上がり順位'] = df.groupby('馬ID')['上がり順位'].shift(1).fillna(9.0)
    df['前走上りレース差'] = df.groupby('馬ID')['上りレース差'].shift(1).fillna(0.0)

    def calc_race_diff(df, col_name):
        return df[col_name] - df.groupby('レースID')[col_name].transform('mean')

    df['偏差_前走タイム差'] = calc_race_diff(df, '前走距離補正タイム差')
    df['偏差_前走着順パーセント'] = calc_race_diff(df, '前走着順パーセント')
    df['偏差_前走上がり順位'] = calc_race_diff(df, '前走上がり順位')
    df['偏差_斤量'] = calc_race_diff(df, '斤量')

    df['馬場'] = df['馬場'].fillna('良') if '馬場' in df.columns else '良'
    df['複勝正解フラグ'] = (df['着順'] <= 3).astype(int)

    df = df.sort_values(['馬ID', '日付'])
    df['馬単体_馬場適性スコア'] = df.groupby(['馬ID', '馬場'])['複勝正解フラグ'].transform(lambda x: x.shift(1).expanding().mean()).fillna(0)
    df = df.sort_values(['父', '日付'])
    df['父_馬場適性スコア'] = df.groupby(['父', '馬場'])['複勝正解フラグ'].transform(lambda x: x.shift(1).expanding().mean()).fillna(0)
    df = df.sort_values(['馬ID', '日付'])

    horse_baba_dict = df.groupby(['馬ID', '馬場'])['複勝正解フラグ'].mean().to_dict()
    sire_baba_dict = df.groupby(['父', '馬場'])['複勝正解フラグ'].mean().to_dict()
    jockey_stats = df.groupby('騎手ID')['着順'].apply(lambda x: (x <= 3).mean()).to_dict()
    trainer_jockey_stats = df.groupby('調教師_騎手')['着順'].apply(lambda x: (x <= 3).mean()).to_dict()
    trainer_interval_stats = df.groupby('調教師_間隔')['着順'].apply(lambda x: (x <= 3).mean()).to_dict()

    if '馬体重' in df.columns:
        df['馬体重_数値'] = df['馬体重'].astype(str).str.extract(r'^(\d+)').astype(float)
        df['馬体カテゴリ'] = pd.cut(df['馬体重_数値'], bins=[0, 459, 499, 999], labels=['小型', '中型', '大型'])
        df['馬体_馬場シナジー'] = df['馬体カテゴリ'].astype(str) + '_' + df['馬場'].astype(str)
        df.loc[df['馬体カテゴリ'].isna(), '馬体_馬場シナジー'] = np.nan

    df['前走大敗フラグ'] = ((df['前走着順'] >= 6) & (df['前走着順パーセント'] > 0.5)).astype(int)
    df['穴馬_距離変更一変'] = df['前走大敗フラグ'] * (df['距離'] != df['前走距離']).astype(int)
    df['穴馬_馬場替わり一変'] = df['前走大敗フラグ'] * (df['芝/ダート'] != df['前走芝ダート']).astype(int)

    c_rate = df['騎手ID'].map(jockey_stats).fillna(0)
    p_rate = df['前走騎手ID'].map(jockey_stats).fillna(0)
    tj_rate = df['調教師_騎手'].map(trainer_jockey_stats).fillna(0)
    df['穴馬_勝負の乗り替わり'] = df['前走大敗フラグ'] * (((c_rate - p_rate) >= 0.10) | (tj_rate >= 0.30)).astype(int)

    cat_features = ['競馬場', '芝/ダート', '回り', 'コース地形', '調教師', '騎手ID', '父', '母父', 
                    '調教師_間隔', '調教師_騎手', '騎手_競馬場', '騎手_距離', '脚質カテゴリ', '馬場', '馬体_馬場シナジー']
    num_features = ['枠番', '距離', '斤量差', '出走間隔', '斤量増減', 
                    '前走上りレース差', '偏差_前走タイム差', '偏差_前走着順パーセント', '偏差_前走上がり順位', '偏差_斤量',
                    '父_馬場適性スコア', '馬単体_馬場適性スコア', '馬体重_数値']
    ana_flags = ['穴馬_距離変更一変', '穴馬_馬場替わり一変', '穴馬_勝負の乗り替わり']
    features = num_features + cat_features + ana_flags

    for col in num_features + ana_flags: df[col] = pd.to_numeric(df[col], errors='coerce')
    for col in cat_features: df[col] = df[col].astype('category')
    
    cat_categories_dict = {}
    for col in cat_features:
        cats = list(df[col].cat.categories)
        if col == '脚質カテゴリ':
            for style in ['逃げ', '先行', '差し', '追込', '自在', '不明']:
                if style not in cats: cats.append(style)
        cat_categories_dict[col] = cats

    # AUC用に簡易的な時系列分割（直近10%をテスト用）
    split_idx = int(len(df) * 0.9)
    train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]

    model = lgb.LGBMClassifier(
        n_estimators=100, random_state=42, importance_type='gain', max_depth=5, num_leaves=20, 
        min_child_samples=100, colsample_bytree=0.8, subsample=0.8, reg_alpha=5.0, reg_lambda=5.0, cat_smooth=50,
        verbose=-1
    )
    model.fit(train_df[features], train_df['複勝正解フラグ'])
    
    # テストデータでAUC計算
    val_preds = model.predict_proba(test_df[features])[:, 1]
    val_auc = roc_auc_score(test_df['複勝正解フラグ'], val_preds)
    fpr, tpr, _ = roc_curve(test_df['複勝正解フラグ'], val_preds)

    return df_latest_clean, horse_baba_dict, sire_baba_dict, jockey_stats, trainer_jockey_stats, trainer_interval_stats, model, features, num_features, cat_features, ana_flags, cat_categories_dict, horse_style_dict, val_auc, fpr, tpr

with st.spinner('keiba-ebye フルパワーAIエンジンを起動・学習中... (初回のみ数分かかります)'):
    df_latest_clean, horse_baba_dict, sire_baba_dict, jockey_stats, trainer_jockey_stats, trainer_interval_stats, model, features, num_features, cat_features, ana_flags, cat_categories_dict, horse_style_dict, val_auc, fpr, tpr = prepare_model_and_data()

headers = {"User-Agent": "Mozilla/5.0"}

# ==========================================
# 2. スクレイピング ＆ アナリティクス関数群
# ==========================================
def get_todays_races(date_str=None):
    races = []
    tokyo_tz = pytz.timezone('Asia/Tokyo')
    now = datetime.datetime.now(tokyo_tz)
    target_date_str = date_str if date_str else now.strftime('%Y%m%d')
    
    url = f'https://race.netkeiba.com/top/race_list.html?kaisai_date={target_date_str}'
    try:
        res = requests.get(url, headers=headers)
        res.encoding = 'euc-jp'
        soup = BeautifulSoup(res.text, 'html.parser')
        for item in soup.find_all('li', class_='RaceList_DataItem'):
            a_tag = item.find('a')
            if not a_tag: continue
            m_id = re.search(r'race_id=(\d{12})', a_tag.get('href', ''))
            if not m_id: continue
            r_id = m_id.group(1)
            if not (1 <= int(r_id[4:6]) <= 10): continue
            
            time_span = item.find('span', class_='RaceList_Itemtime')
            title_span = item.find('span', class_='ItemTitle')
            place_dict = {'01':'札幌','02':'函館','03':'福島','04':'新潟','05':'東京','06':'中山','07':'中京','08':'京都','09':'阪神','10':'小倉'}
            place = place_dict.get(r_id[4:6], '不明')
            r_num = int(r_id[10:12])
            
            if time_span and title_span:
                try: start_dt = tokyo_tz.localize(datetime.datetime.strptime(f"{target_date_str} {time_span.text.strip()}", "%Y%m%d %H:%M"))
                except: start_dt = tokyo_tz.localize(datetime.datetime.strptime(f"{target_date_str} 12:00", "%Y%m%d %H:%M"))
                races.append({
                    'id': r_id, 'place': place, 'num': r_num, 'title': title_span.text.strip(), 'time': start_dt,
                    'sort_key': f"{start_dt.strftime('%H%M')}_{r_id}"
                })
    except: pass

    if not races:
        url = f'https://db.netkeiba.com/race/list/{target_date_str}/'
        try:
            res = requests.get(url, headers=headers)
            res.encoding = 'euc-jp'
            ids = set(re.findall(r'/race/(\d{12})', res.text))
            place_dict = {'01':'札幌','02':'函館','03':'福島','04':'新潟','05':'東京','06':'中山','07':'中京','08':'京都','09':'阪神','10':'小倉'}
            for r_id in ids:
                if not (1 <= int(r_id[4:6]) <= 10): continue
                place = place_dict.get(r_id[4:6], '不明')
                r_num = int(r_id[10:12])
                dummy_time = tokyo_tz.localize(datetime.datetime.strptime(f"{target_date_str} 12:00", "%Y%m%d %H:%M"))
                races.append({
                    'id': r_id, 'place': place, 'num': r_num, 'title': f"{place} {r_num}R", 'time': dummy_time,
                    'sort_key': f"{r_id[4:6]}{r_num:02d}"
                })
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
    try:
        res = requests.get(f"https://race.netkeiba.com/race/result.html?race_id={race_id}", headers=headers)
        res.encoding = 'euc-jp'
        soup = BeautifulSoup(res.text, 'html.parser')
        pay_tables = soup.find_all('table', class_=re.compile(r'Pay_Table_01'))
        for tbl in pay_tables:
            for tr in tbl.find_all('tr'):
                th = tr.find('th')
                if not th: continue
                kind = th.text.strip()
                if kind in ['単勝', '複勝']:
                    res_td = tr.find('td', class_=re.compile(r'Result'))
                    pay_td = tr.find('td', class_=re.compile(r'Payout'))
                    if res_td and pay_td:
                        umbans = [re.sub(r'\D', '', s) for s in res_td.stripped_strings if re.sub(r'\D', '', s)]
                        pays = [re.sub(r'\D', '', s) for s in pay_td.stripped_strings if re.sub(r'\D', '', s)]
                        for u, p in zip(umbans, pays):
                            if u and p:
                                if kind == '単勝': tansho_dict[int(u)] = int(p)
                                else: fukusho_dict[int(u)] = int(p)
    except: pass

    if not tansho_dict:
        try:
            res = requests.get(f"https://db.netkeiba.com/race/{race_id}/", headers=headers)
            res.encoding = 'euc-jp'
            soup = BeautifulSoup(res.text, 'html.parser')
            for tbl in soup.find_all('table', summary='払い戻し') or soup.find_all('table', class_='pay_table_01'):
                for tr in tbl.find_all('tr'):
                    th = tr.find('th')
                    if th and th.text in ['単勝', '複勝']:
                        tds = tr.find_all('td')
                        if len(tds) >= 2:
                            umbans = list(tds[0].stripped_strings)
                            pays = list(tds[1].stripped_strings)
                            for u, p in zip(umbans, pays):
                                u, p = re.sub(r'\D', '', u), re.sub(r'\D', '', p)
                                if u and p:
                                    if th.text == '単勝': tansho_dict[int(u)] = int(p)
                                    else: fukusho_dict[int(u)] = int(p)
        except: pass
    return tansho_dict, fukusho_dict

def get_odds_from_soup(s_soup):
    o_dict = {}
    tgt_table = s_soup.select_one('.Shutuba_Table') or s_soup.select_one('#All_Result_Table') or s_soup.select_one('.race_table_01')
    if not tgt_table: return o_dict
    ths = tgt_table.find_all('th')
    u_idx, o_idx = -1, -1
    for i, th in enumerate(ths):
        c_txt = re.sub(r'\s+', '', th.text)
        if '馬番' in c_txt: u_idx = i
        if '単勝' in c_txt or 'オッズ' in c_txt: o_idx = i
    if u_idx != -1 and o_idx != -1:
        for tr in tgt_table.find_all('tr')[1:]:
            tds = tr.find_all('td')
            if len(tds) > max(u_idx, o_idx):
                u_m = re.search(r'\d+', tds[u_idx].text)
                o_m = re.search(r'\d+\.\d+', tds[o_idx].text)
                if u_m and o_m:
                    o_dict[int(u_m.group(0))] = float(o_m.group(0))
    return o_dict

# ==========================================
# 3. 本格AI予測関数
# ==========================================
def run_real_prediction(race_id, race_date_str):
    odds_dict = {}
    urls_to_try = [
        f'https://race.netkeiba.com/race/shutuba.html?race_id={race_id}',
        f'https://race.netkeiba.com/race/result.html?race_id={race_id}',
        f'https://db.netkeiba.com/race/{race_id}/'
    ]
    for fetch_url in urls_to_try:
        try:
            r = requests.get(fetch_url, headers=headers)
            r.encoding = 'euc-jp'
            odds_dict = get_odds_from_soup(BeautifulSoup(r.text, 'html.parser'))
            if odds_dict: break
        except: pass

    url = f'https://race.netkeiba.com/race/shutuba.html?race_id={race_id}'
    try:
        res = requests.get(url, headers=headers); res.encoding = 'euc-jp'
        soup = BeautifulSoup(res.text, 'html.parser')
    except: return None, None, None, None, None

    race_data_box = soup.find('div', class_='RaceData01')
    if not race_data_box: return None, None, None, None, None
    race_text = race_data_box.text.replace('\n', '')
    
    baba_match = re.search(r'馬場:([良稍重不良]+)', race_text)
    todays_baba = baba_match.group(1) if baba_match else '良'
    
    track_dist_match = re.search(r'(芝|ダ|障|障害).*?(\d+)m', race_text)
    track_type = "芝" if track_dist_match and track_dist_match.group(1) == "芝" else "ダート" if track_dist_match and "ダ" in track_dist_match.group(1) else "障害"
    distance = float(track_dist_match.group(2)) if track_dist_match else 1600.0

    place_dict = {'01':'札幌','02':'函館','03':'福島','04':'新潟','05':'東京','06':'中山','07':'中京','08':'京都','09':'阪神','10':'小倉'}
    place = place_dict.get(str(race_id)[4:6], '不明')
    course_slope = '急坂' if place in ['中山','阪神','中京'] else '平坦' if place in ['京都','新潟','小倉'] else '長い上り坂' if place == '東京' else '小回り' if place == '福島' else '洋芝'
    mawari = '左回り' if place in ['東京', '新潟', '中京'] else '右回り'

    horses = []
    table = soup.select_one('.Shutuba_Table') or soup.find('table')
    if not table: return None, None, None, None, None

    for tr in table.find_all('tr'):
        tds = tr.find_all('td')
        if len(tds) < 5: continue
        try:
            waku = int(re.search(r'\d+', (tr.select_one('td[class*="Waku"]') or tds[0]).text).group(0))
            umaban = int(re.search(r'\d+', (tr.select_one('td[class*="Umaban"]') or tds[1]).text).group(0))
            horse_a = tr.select_one('td.HorseInfo a') or tr.find('a', href=re.compile(r'/horse/'))
            horse_id = re.search(r'\d+', horse_a['href']).group(0)
            jockey_a = tr.select_one('td.Jockey a') or tr.find('a', href=re.compile(r'/jockey/'))
            jockey_id = re.search(r'\d+', jockey_a['href']).group(0) if jockey_a else "0"
            trainer_a = tr.select_one('td.Trainer a') or tr.find('a', href=re.compile(r'/trainer/'))
            trainer_id = re.search(r'\d+', trainer_a['href']).group(0) if trainer_a else "不明"
            kinryo = float(re.search(r'\d+(\.\d+)?', (tr.select_one('td.Jockey') or jockey_a.parent).find_previous_sibling('td').text).group(0))
            weight_td = tr.select_one('td.Weight')
            weight_val = float(re.search(r'\d+', weight_td.text).group(0)) if weight_td and re.search(r'\d+', weight_td.text) else np.nan
            odds_val = odds_dict.get(umaban, 0.0)
            horses.append({'枠番': waku, '馬番': umaban, '馬名': horse_a.text.strip(), '馬ID': horse_id, '斤量': kinryo, '騎手ID': jockey_id, '調教師': trainer_id, '距離': distance, '競馬場': place, '芝/ダート': track_type, '回り': mawari, 'コース地形': course_slope, '馬場': todays_baba, '馬体重_数値': weight_val, '単勝オッズ': odds_val})
        except: continue

    if not horses: return None, None, None, None, None
    df_test = pd.DataFrame(horses)
    df_test = pd.merge(df_test, df_latest_clean, on='馬ID', how='left')

    df_test['斤量差'] = df_test['斤量'] - df_test['斤量'].mean()
    df_test['偏差_斤量'] = df_test['斤量'] - df_test['斤量'].mean()
    r_date_obj = pd.to_datetime(race_date_str)
    df_test['出走間隔'] = (r_date_obj - df_test['前走日付']).dt.days.fillna(30)
    df_test['斤量増減'] = (df_test['斤量'] - df_test['前走斤量']).fillna(0)
    
    # 💡 【進化】ここでも最新の脚質辞書を使う！
    df_test['脚質カテゴリ'] = df_test['馬ID'].map(horse_style_dict).fillna('不明')
    
    df_test['偏差_前走タイム差'] = df_test['前走距離補正タイム差'] - df_test['前走距離補正タイム差'].mean()
    df_test['偏差_前走着順パーセント'] = df_test['前走着順パーセント'] - df_test['前走着順パーセント'].mean()
    df_test['偏差_前走上がり順位'] = df_test['前走上がり順位'].fillna(9.0) - df_test['前走上がり順位'].fillna(9.0).mean()
    df_test['前走上りレース差'] = df_test['前走上りレース差'].fillna(0.0)
    df_test['馬単体_馬場適性スコア'] = df_test.set_index(['馬ID', '馬場']).index.map(horse_baba_dict).fillna(0)
    df_test['父_馬場適性スコア'] = df_test.set_index(['父', '馬場']).index.map(sire_baba_dict).fillna(0)
    df_test['馬体カテゴリ'] = pd.cut(df_test['馬体重_数値'], bins=[0, 459, 499, 999], labels=['小型', '中型', '大型'])
    df_test['馬体_馬場シナジー'] = df_test['馬体カテゴリ'].astype(str) + '_' + df_test['馬場'].astype(str)
    df_test.loc[df_test['馬体カテゴリ'].isna(), '馬体_馬場シナジー'] = np.nan
    df_test['間隔カテゴリ'] = pd.cut(df_test['出走間隔'], bins=[-1, 14, 30, 9999], labels=['詰合', '標準', '休明']).astype(str)
    df_test['調教師_間隔'] = df_test['調教師'].astype(str) + '_' + df_test['間隔カテゴリ']
    df_test['調教師_騎手'] = df_test['調教師'].astype(str) + '_' + df_test['騎手ID'].astype(str)
    df_test['騎手_競馬場'] = df_test['騎手ID'].astype(str) + '_' + df_test['競馬場'].astype(str)
    df_test['騎手_距離'] = df_test['騎手ID'].astype(str) + '_' + df_test['距離'].astype(str)
    df_test['前走大敗フラグ'] = ((df_test['前走着順'] >= 6) & (df_test['前走着順パーセント'] > 0.5)).astype(int)
    df_test['穴馬_距離変更一変'] = df_test['前走大敗フラグ'] * (df_test['距離'] != df_test['前走距離']).astype(int)
    df_test['穴馬_馬場替わり一変'] = df_test['前走大敗フラグ'] * (df_test['芝/ダート'] != df_test['前走芝ダート']).astype(int)
    c_rate = df_test['騎手ID'].map(jockey_stats).fillna(0)
    p_rate = df_test['前走騎手ID'].map(jockey_stats).fillna(0)
    tj_rate = df_test['調教師_騎手'].map(trainer_jockey_stats).fillna(0)
    df_test['穴馬_勝負の乗り替わり'] = df_test['前走大敗フラグ'] * (((c_rate - p_rate) >= 0.10) | (tj_rate >= 0.30)).astype(int)

    for col in num_features + ana_flags: df_test[col] = pd.to_numeric(df_test[col], errors='coerce')
    for col in cat_features: df_test[col] = pd.Categorical(df_test[col], categories=cat_categories_dict[col])

    raw_probs = model.predict_proba(df_test[features])[:, 1]
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
        reasons = []
        if row['穴馬_馬場替わり一変'] == 1: reasons.append("馬場替わり")
        if row['穴馬_距離変更一変'] == 1: reasons.append("距離変更")
        if row['穴馬_勝負の乗り替わり'] == 1: reasons.append("勝負騎手")
        
        if rank >= 5:
            is_potential_ai = row['勝率(AI予測)'] >= (df_test.loc[4, '勝率(AI予測)'] * 0.5)
            if len(reasons) >= 2 or (len(reasons) >= 1 and is_potential_ai):
                topics_list.append(f"📌 {row['馬名']}({', '.join(reasons)})")
                if f"{row['馬番']}番" not in ana_horse_nums: ana_horse_nums.append(f"{row['馬番']}番")

    ana_str = "・".join(str(n) for n in ana_horse_nums[:3]) if ana_horse_nums else ""
    p1, p2 = df_test.loc[0, '勝率(AI予測)'], df_test.loc[1, '勝率(AI予測)']
    if p1 >= 0.20: reco = f"🎯 馬連・馬単: ◎から印馬・穴馬({ana_str})への流し"
    elif p1 >= 0.12 and (p1 - p2) >= 0.03: reco = f"🎯 馬連・ワイド: ◎から穴馬({ana_str})への流しで高配当狙い"
    else: reco = f"⚠️ 上位評価割れ。印馬と穴馬({ana_str})のボックス推奨"

    return df_test, topics_list, reco, track_type, place

# ==========================================
# 4. メインUI構成 (サイドバー)
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

# 💡 【進化2】スマホで見やすいタブ型UIに変更！
def display_result(df_res, topics, reco):
    tab1, tab2 = st.tabs(["📊 予想一覧", "💡 買い目・トピック"])
    
    with tab1:
        def highlight_ev(row):
            return ['background-color: rgba(255, 99, 71, 0.3)' if row['期待値'] >= 1.5 else '' for _ in row]
        show_df = df_res[['印', '馬番', '馬名', '脚質カテゴリ', '単勝オッズ', '勝率(AI予測)', '期待値']].copy()
        show_df = show_df.rename(columns={'勝率(AI予測)': 'AI勝率', '脚質カテゴリ': '脚質'})
        show_df['AI勝率'] = (show_df['AI勝率'] * 100).map('{:.1f}%'.format)
        st.dataframe(show_df.style.apply(highlight_ev, axis=1).format({'期待値': '{:.2f}'}), use_container_width=True)
    
    with tab2:
        ev_horses = df_res[(df_res.index < 5) & (df_res['期待値'] >= 1.5)]
        if not ev_horses.empty:
            st.error(f"💰 **【期待値レーダー発動】** {', '.join(ev_horses['馬名'].tolist())} に強烈なオッズ妙味あり！")
        if topics:
            st.info("**📝 要注目トピック馬:**\n\n" + "\n".join(topics))
        st.success(f"**🤖 AI推奨買い目:**\n\n{reco}")

# ------------------------------------------
# 各種モードの処理
# ------------------------------------------
if action in ["⏩ 次のレースを予想", "📜 本日の全レース予想", "🔍 レースを指定して予想"]:
    todays_races = get_todays_races()
    if not todays_races:
        st.warning(f"本日 ({now.strftime('%Y/%m/%d')}) はJRAのレースが開催されていません。")
    else:
        if action == "⏩ 次のレースを予想":
            st.subheader("🕒 まもなく出走するレース")
            next_race = next((r for r in todays_races if tokyo_tz.localize(r['time']) > now), None)
            if next_race:
                mins_left = int((tokyo_tz.localize(next_race['time']) - now).total_seconds() / 60)
                st.info(f"👉 **{next_race['place']} {next_race['num']}R** 「{next_race['title']}」 (あと **{mins_left}** 分)")
                if st.button("🚀 keiba-ebye 予想起動！", type="primary"):
                    with st.spinner('AIが推論中...'):
                        res_df, topics, reco, _, _ = run_real_prediction(next_race['id'], now.strftime('%Y-%m-%d'))
                        display_result(res_df, topics, reco)
            else: st.success("🏁 本日の全レースは終了しました。")

        elif action == "📜 本日の全レース予想":
            st.subheader(f"📅 本日の全レース一覧")
            if st.button("🚀 全レース一括予想", type="primary"):
                my_bar = st.progress(0, text="推論中...")
                for i, r in enumerate(todays_races):
                    st.markdown(f"#### ■ {r['place']} {r['num']}R")
                    res_df, topics, reco, _, _ = run_real_prediction(r['id'], now.strftime('%Y-%m-%d'))
                    if res_df is not None: display_result(res_df.head(5), topics, reco)
                    my_bar.progress((i + 1) / len(todays_races))

        elif action == "🔍 レースを指定して予想":
            st.subheader("🎯 レースを指定")
            options = [f"{r['place']} {r['num']}R - {r['title']}" for r in todays_races]
            selected = st.selectbox("レースを選んでください", options)
            target_race = todays_races[options.index(selected)]
            if st.button("🚀 予想開始", type="primary"):
                with st.spinner('推論中...'):
                    res_df, topics, reco, _, _ = run_real_prediction(target_race['id'], now.strftime('%Y-%m-%d'))
                    if res_df is not None: display_result(res_df, topics, reco)

elif action == "📅 今週末の全レース予想":
    st.subheader("📅 今週末 (土・日) の先取り予想")
    st.markdown("金曜や前日に、週末のレース出馬表を先取りして予想します！")
    sat_str, sun_str = get_weekend_dates()
    
    if st.button("🚀 週末の出馬表を取得・予想", type="primary"):
        with st.spinner('週末のレースを収集中...'):
            sat_races = get_todays_races(sat_str)
            sun_races = get_todays_races(sun_str)
            all_weekend_races = sat_races + sun_races
            
        if not all_weekend_races: st.error("まだ今週末の出馬表が発表されていません。")
        else:
            my_bar = st.progress(0, text="週末のレースを推論中...")
            for i, r in enumerate(all_weekend_races):
                # 曜日を判定して表示
                r_date = sat_str if r in sat_races else sun_str
                with st.expander(f"🏁 {r_date[:4]}/{r_date[4:6]}/{r_date[6:]} - {r['place']} {r['num']}R"):
                    res_df, topics, reco, _, _ = run_real_prediction(r['id'], f"{r_date[:4]}-{r_date[4:6]}-{r_date[6:]}")
                    if res_df is not None: display_result(res_df.head(5), topics, reco)
                my_bar.progress((i + 1) / len(all_weekend_races))

elif action == "🧪 性能試験 (バックテスト)":
    st.subheader("🧪 keiba-ebye 性能試験 (詳細レポート付き)")
    test_date = st.date_input("テストする日付を選択", datetime.date.today() - datetime.timedelta(days=3))
    date_str = test_date.strftime('%Y%m%d')
    
    if st.button("🔥 バックテスト実行！", type="primary"):
        with st.spinner(f'{test_date.strftime("%Y/%m/%d")} の全レースを推論・集計中...'):
            test_races = get_todays_races(date_str)
            if not test_races: st.error("レースが見つかりません。")
            else:
                my_bar = st.progress(0, text="推論＆集計中...")
                total_invest, total_return_t, total_return_f, ev_hits = 0, 0, 0, 0
                for i, r in enumerate(test_races):
                    with st.expander(f"🏁 {r['place']} {r['num']}R 予想詳細を見る"):
                        res_df, topics, reco, _, _ = run_real_prediction(r['id'], test_date.strftime('%Y-%m-%d'))
                        t_dict, f_dict = get_payouts(r['id'])
                        if res_df is not None and t_dict:
                            display_result(res_df, topics, reco)
                            ev_horses = res_df[(res_df.index < 5) & (res_df['期待値'] >= 1.5)]
                            for _, horse in ev_horses.iterrows():
                                umaban = horse['馬番']
                                total_invest += 100
                                if umaban in t_dict: total_return_t += t_dict[umaban]
                                if umaban in f_dict: 
                                    total_return_f += f_dict[umaban]
                                    ev_hits += 1
                    my_bar.progress((i + 1) / len(test_races))
                
                st.markdown("---")
                st.markdown("### 🏆 バックテスト 最終集計レポート")
                if total_invest > 0:
                    col1, col2, col3 = st.columns(3)
                    col1.metric("期待値馬 総投資額", f"¥{total_invest:,}")
                    col2.metric("単勝 回収率", f"{(total_return_t / total_invest * 100):.1f}%", f"¥{total_return_t:,}")
                    col3.metric("複勝 回収率", f"{(total_return_f / total_invest * 100):.1f}%", f"的中 {ev_hits}回")
                else: st.warning("この日は「期待値1.5超え」の推奨対象馬がいませんでした。")

# 💡 【進化3】AUC（AI精度評価）機能の搭載！
elif action == "📈 AI精度評価 (AUCスコア)":
    st.subheader("📈 keiba-ebye AI精度評価 (ROC-AUC)")
    st.markdown("現在の学習データに基づく、AIの「馬券内に入りうる馬の識別能力」を評価します。")
    
    st.metric(label="📊 総合AUCスコア (1.0に近いほど優秀)", value=f"{val_auc:.4f}")
    if val_auc > 0.7: st.success("🔥 素晴らしい精度です！AIは強い馬と弱い馬を高い精度で見分けています。")
    elif val_auc > 0.6: st.info("👍 良好な精度です。実運用に耐えうるレベルです。")
    else: st.warning("⚠️ 精度が低めです。データ量の増加や特徴量の見直しが必要です。")
    
    st.markdown("##### 📉 ROC曲線")
    roc_df = pd.DataFrame({'False Positive Rate (間違える確率)': fpr, 'True Positive Rate (当てる確率)': tpr})

    st.line_chart(roc_df, x='False Positive Rate (間違える確率)', y='True Positive Rate (当てる確率)')


