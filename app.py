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
import pickle

# ==========================================
# UI・ページ設定
# ==========================================
st.set_page_config(page_title="keiba-ebye 予測ダッシュボード", page_icon="🐴", layout="wide")

st.title("🐴 keiba-ebye 予測ダッシュボード")
st.markdown("独自のデータアナリティクスとAI (ebi × AI × Eye) が、期待値に隠されたお宝馬を暴き出します。")

# ==========================================
# 1. 学習済みアセットの読み込み (高速化の鍵)
# ==========================================
@st.cache_resource
def load_trained_assets():
    """PCで作成した学習済みデータをロードする"""
    with open('trained_assets.pkl', 'rb') as f:
        assets = pickle.load(f)
    return (
        assets["df_latest_clean"],
        assets["horse_baba_dict"],
        assets["sire_baba_dict"],
        assets["jockey_stats"],
        assets["trainer_jockey_stats"],
        assets["trainer_interval_stats"],
        assets["model"],
        assets["features"],
        assets["num_features"],
        assets["cat_features"],
        assets["ana_flags"],
        assets["cat_categories_dict"]
    )

with st.spinner('AIエンジンをロード中...'):
    try:
        (df_latest_clean, horse_baba_dict, sire_baba_dict, jockey_stats, 
         trainer_jockey_stats, trainer_interval_stats, model, features, 
         num_features, cat_features, ana_flags, cat_categories_dict) = load_trained_assets()
    except Exception as e:
        st.error(f"アセットの読み込みに失敗しました。GitHubに 'trained_assets.pkl' があるか確認してください。: {e}")
        st.stop()

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
    return sorted(races, key=lambda x: x['sort_key'])

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
        f'https://race.netkeiba.com/race/result.html?race_id={race_id}'
    ]
    for fetch_url in urls_to_try:
        try:
            r = requests.get(fetch_url, headers=headers); r.encoding = 'euc-jp'
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
    track_type = "芝" if track_dist_match and track_dist_match.group(1) == "芝" else "ダート"
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
            trainer_id = trainer_a['href'].split('/')[-2] if trainer_a else "不明"
            kinryo = float(re.search(r'\d+(\.\d+)?', (tr.select_one('td.Jockey') or jockey_a.parent).find_previous_sibling('td').text).group(0))
            weight_td = tr.select_one('td.Weight')
            weight_val = float(re.search(r'\d+', weight_td.text).group(0)) if weight_td and re.search(r'\d+', weight_td.text) else np.nan
            odds_val = odds_dict.get(umaban, 0.0)
            horses.append({'枠番': waku, '馬番': umaban, '馬名': horse_a.text.strip(), '馬ID': horse_id, '斤量': kinryo, '騎手ID': jockey_id, '調教師': trainer_id, '距離': distance, '競馬場': place, '芝/ダート': track_type, '回り': mawari, 'コース地形': course_slope, '馬場': todays_baba, '馬体重_数値': weight_val, '単勝オッズ': odds_val})
        except: continue

    if not horses: return None, None, None, None, None
    df_test = pd.DataFrame(horses)
    df_test = pd.merge(df_test, df_latest_clean, on='馬ID', how='left')

    # 特徴量生成ロジック (学習時と同じ)
    df_test['斤量差'] = df_test['斤量'] - df_test['斤量'].mean()
    df_test['偏差_斤量'] = df_test['斤量'] - df_test['斤量'].mean()
    r_date_obj = pd.to_datetime(race_date_str)
    df_test['出走間隔'] = (r_date_obj - df_test['前走日付']).dt.days.fillna(30)
    df_test['斤量増減'] = (df_test['斤量'] - df_test['前走斤量']).fillna(0)
    df_test['前走コーナー順位'] = pd.to_numeric(df_test['前走コーナー順位'], errors='coerce').fillna(7)
    df_test['脚質カテゴリ'] = pd.cut(df_test['前走コーナー順位'], bins=[-1, 3.5, 9.5, 99], labels=['先行', '差し', '追込']).astype(str)
    df_test['偏差_前走タイム差'] = df_test['前走距離補正タイム差'] - df_test['前走距離補正タイム差'].mean()
    df_test['偏差_前走着順パーセント'] = df_test['前走着順パーセント'] - df_test['前走着順パーセント'].mean()
    df_test['偏差_前走上がり順位'] = df_test['前走上がり順位'].fillna(9.0) - df_test['前走上がり順位'].fillna(9.0).mean()
    df_test['前走上りレース差'] = df_test['前走上りレース差'].fillna(0.0)
    df_test['馬単体_馬場適性スコア'] = df_test.set_index(['馬ID', '馬場']).index.map(horse_baba_dict).fillna(0)
    df_test['父_馬場適性スコア'] = df_test.set_index(['父', '馬場']).index.map(sire_baba_dict).fillna(0)
    df_test['馬体カテゴリ'] = pd.cut(df_test['馬体重_数値'], bins=[0, 459, 499, 999], labels=['小型', '中型', '大型'])
    df_test['馬体_馬場シナジー'] = df_test['馬体カテゴリ'].astype(str) + '_' + df_test['馬場'].astype(str)
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
        if rank >= 5 and (len(reasons) >= 1):
            topics_list.append(f"📌 {row['馬名']}({', '.join(reasons)})")
            if f"{row['馬番']}番" not in ana_horse_nums: ana_horse_nums.append(f"{row['馬番']}番")

    ana_str = "・".join(str(n) for n in ana_horse_nums[:3]) if ana_horse_nums else ""
    p1 = df_test.loc[0, '勝率(AI予測)']
    reco = f"🎯 ◎から印馬・穴馬({ana_str})への流し" if p1 >= 0.15 else f"⚠️ ボックス推奨 ({ana_str})"

    return df_test, topics_list, reco, track_type, place

# ==========================================
# 4. メインUI (表示ロジック)
# ==========================================
st.sidebar.markdown("## 🕹️ メニュー")
action = st.sidebar.radio("機能", ["⏩ 次のレース", "📜 本日の全レース", "🔍 指定予想", "🧪 バックテスト"])

tokyo_tz = pytz.timezone('Asia/Tokyo')
now = datetime.datetime.now(tokyo_tz)

def display_result(df_res, topics, reco):
    show_df = df_res[['印', '枠番', '馬番', '馬名', '脚質カテゴリ', '単勝オッズ', '勝率(AI予測)', '期待値']].copy()
    show_df = show_df.rename(columns={'勝率(AI予測)': 'AI勝率'})
    show_df['AI勝率'] = (show_df['AI勝率'] * 100).map('{:.1f}%'.format)
    st.dataframe(show_df.style.apply(lambda r: ['background-color: rgba(255, 99, 71, 0.3)' if r['期待値'] >= 1.5 else '' for _ in r], axis=1).format({'期待値': '{:.2f}'}), use_container_width=True)
    if topics: st.info("**📝 注目馬:**\n" + "\n".join(topics))
    st.success(f"**🤖 推奨:** {reco}")

if action == "⏩ 次のレース":
    todays_races = get_todays_races()
    next_race = next((r for r in todays_races if r['time'] > now), None)
    if next_race:
        st.info(f"👉 **{next_race['place']} {next_race['num']}R** (あと {int((next_race['time'] - now).total_seconds()/60)} 分)")
        if st.button("🚀 予想起動"):
            res_df, topics, reco, _, _ = run_real_prediction(next_race['id'], now.strftime('%Y-%m-%d'))
            display_result(res_df, topics, reco)
    else: st.success("🏁 本日の全レース終了")

elif action == "📜 本日の全レース":
    todays_races = get_todays_races()
    if st.button("🚀 全レース一括予想"):
        for r in todays_races:
            st.markdown(f"#### ■ {r['place']} {r['num']}R")
            res_df, topics, reco, _, _ = run_real_prediction(r['id'], now.strftime('%Y-%m-%d'))
            if res_df is not None: display_result(res_df.head(5), topics, reco)

elif action == "🔍 指定予想":
    todays_races = get_todays_races()
    options = [f"{r['place']} {r['num']}R" for r in todays_races]
    selected = st.selectbox("レース選択", options)
    if st.button("🚀 予想開始"):
        target_race = todays_races[options.index(selected)]
        res_df, topics, reco, _, _ = run_real_prediction(target_race['id'], now.strftime('%Y-%m-%d'))
        display_result(res_df, topics, reco)

elif action == "🧪 バックテスト":
    test_date = st.date_input("日付選択", datetime.date.today() - datetime.timedelta(days=3))
    if st.button("🔥 実行"):
        test_races = get_todays_races(test_date.strftime('%Y%m%d'))
        for r in test_races:
            with st.expander(f"🏁 {r['place']} {r['num']}R"):
                res_df, topics, reco, _, _ = run_real_prediction(r['id'], test_date.strftime('%Y-%m-%d'))
                if res_df is not None: display_result(res_df, topics, reco)