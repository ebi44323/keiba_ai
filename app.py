import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import requests
from bs4 import BeautifulSoup
import re
import datetime
import pytz

# ページ設定
st.set_page_config(page_title="神AI 競馬予想ダッシュボード", page_icon="🏇", layout="wide")

st.title("🏇 神AI 競馬予想ダッシュボード")
st.markdown("AIがリアルタイムのオッズと連動し、期待値（オッズ妙味）を瞬時に計算します！")

# ==========================================
# 1. モデルとデータの読み込み（キャッシュ化で超高速）
# ==========================================
@st.cache_resource
def load_model_and_data():
    # 実際にはここに学習済みモデルを保存したファイル（model.txt等）を読み込む処理を入れるのがベストですが、
    # 今回はアプリ起動時にサクッと簡易学習させる構成にします（本来は事前学習済みモデル推奨）。
    df = pd.read_csv('learning_data_perfect_tier.zip', dtype={'馬ID': str, '騎手ID': str, 'レースID': str})
    df['日付'] = pd.to_datetime(df['日付'])
    df['複勝正解フラグ'] = (df['着順'] <= 3).astype(int)
    horse_baba_dict = df.groupby(['馬ID', '馬場'])['複勝正解フラグ'].mean().to_dict()
    sire_baba_dict = df.groupby(['父', '馬場'])['複勝正解フラグ'].mean().to_dict()
    df_latest_clean = df.groupby('馬ID').tail(1).copy()
    
    return df, df_latest_clean, horse_baba_dict, sire_baba_dict

with st.spinner('AIの脳内データをロード中...'):
    df, df_latest_clean, horse_baba_dict, sire_baba_dict = load_model_and_data()

# ==========================================
# 2. 汎用スクレイピング関数
# ==========================================
headers = {"User-Agent": "Mozilla/5.0"}

def get_todays_races():
    # 日本時間を取得
    tokyo_tz = pytz.timezone('Asia/Tokyo')
    now = datetime.datetime.now(tokyo_tz)
    date_str = now.strftime('%Y%m%d')
    
    url = f'https://race.netkeiba.com/top/race_list.html?kaisai_date={date_str}'
    races = []
    try:
        res = requests.get(url, headers=headers)
        res.encoding = 'euc-jp'
        soup = BeautifulSoup(res.text, 'html.parser')
        
        # JRAのレース一覧リストを抽出
        race_list_items = soup.find_all('li', class_='RaceList_DataItem')
        for item in race_list_items:
            a_tag = item.find('a')
            if not a_tag: continue
            href = a_tag.get('href', '')
            m_id = re.search(r'race_id=(\d{12})', href)
            if not m_id: continue
            
            r_id = m_id.group(1)
            time_span = item.find('span', class_='RaceList_Itemtime')
            title_span = item.find('span', class_='ItemTitle')
            
            if time_span and title_span:
                start_time_str = time_span.text.strip() # "10:05" など
                start_dt = tokyo_tz.localize(datetime.datetime.strptime(f"{date_str} {start_time_str}", "%Y%m%d %H:%M"))
                
                place_dict = {'01':'札幌','02':'函館','03':'福島','04':'新潟','05':'東京','06':'中山','07':'中京','08':'京都','09':'阪神','10':'小倉'}
                place = place_dict.get(r_id[4:6], '不明')
                r_num = int(r_id[10:12])
                
                races.append({
                    'id': r_id,
                    'place': place,
                    'num': r_num,
                    'title': title_span.text.strip(),
                    'time': start_dt
                })
    except: pass
    # 時間順にソート
    return sorted(races, key=lambda x: x['time'])

# ==========================================
# 3. ダミーの予想関数（※ここにVer18の予測ロジックを合体させます）
# ==========================================
def run_prediction(race_id):
    # 本来はここで出馬表とオッズをスクレイピングし、AIで予測する処理が走ります。
    # 今回はUIデモのため、ダミーデータを返します。
    # （※実際の予測ロジックはVer18のコードをごっそり移植します）
    return pd.DataFrame({
        '印': ['◎', '〇', '▲', '△', '☆'],
        '枠番': [1, 7, 4, 8, 2],
        '馬番': [2, 14, 8, 15, 3],
        '馬名': ['デモストレーション', 'スマホテスト', 'アプリカシテ', 'クラウドデプロイ', 'スゴイハヤイ'],
        'AI勝率': [18.3, 16.9, 10.9, 7.0, 7.0],
        '単勝オッズ': [2.9, 8.5, 12.0, 44.1, 69.6],
        '期待値': [0.53, 1.43, 1.30, 3.08, 4.87]
    })

# ==========================================
# 4. メインUI構成
# ==========================================
st.sidebar.markdown("## 🕹️ コマンドパネル")
action = st.sidebar.radio("機能を選択", ["⏩ 次のレースを予想", "📜 本日の全レース予想", "🔍 レースを指定して予想"])

tokyo_tz = pytz.timezone('Asia/Tokyo')
now = datetime.datetime.now(tokyo_tz)

todays_races = get_todays_races()

if not todays_races:
    st.warning(f"本日 ({now.strftime('%Y/%m/%d')}) はJRAのレースが開催されていません。")
else:
    if action == "⏩ 次のレースを予想":
        st.subheader("🕒 まもなく出走するレース（次レース予想）")
        
        # 現在時刻より未来のレースで、一番近いものを探す
        next_race = None
        for r in todays_races:
            if r['time'] > now:
                next_race = r
                break
                
        if next_race:
            time_left = next_race['time'] - now
            mins_left = int(time_left.total_seconds() / 60)
            st.info(f"👉 **{next_race['place']} {next_race['num']}R** 「{next_race['title']}」 (発走 {next_race['time'].strftime('%H:%M')} / あと **{mins_left}** 分)")
            
            if st.button("🚀 このレースを予想する！", type="primary"):
                with st.spinner('最新オッズと出馬表を解析中...'):
                    time.sleep(1) # スクレピングの待機時間（デモ）
                    res_df = run_prediction(next_race['id'])
                    
                    st.markdown("### 📊 AI予想結果")
                    # 期待値が1.5以上の行をハイライトする関数
                    def highlight_ev(row):
                        return ['background-color: #ffcccc' if row['期待値'] >= 1.5 else '' for _ in row]
                    
                    st.dataframe(res_df.style.apply(highlight_ev, axis=1).format({'AI勝率': '{:.1f}%', '期待値': '{:.2f}'}), use_container_width=True)
                    st.success("💰 **【期待値レーダー】** 背景が赤い馬は過小評価されているオッズ妙味馬です！")
        else:
            st.success("🏁 本日の全レースは終了しました。")

    elif action == "📜 本日の全レース予想":
        st.subheader(f"📅 本日の全レース一覧 ({len(todays_races)}レース)")
        if st.button("🚀 全レース一括予想（※時間がかかります）", type="primary"):
            progress_text = "AIが全レースを処理中..."
            my_bar = st.progress(0, text=progress_text)
            
            for i, r in enumerate(todays_races):
                st.markdown(f"#### ■ {r['place']} {r['num']}R (発走 {r['time'].strftime('%H:%M')})")
                res_df = run_prediction(r['id'])
                st.dataframe(res_df.style.format({'AI勝率': '{:.1f}%', '期待値': '{:.2f}'}))
                my_bar.progress((i + 1) / len(todays_races), text=f"処理中: {i+1}/{len(todays_races)} レース")
                time.sleep(0.5)
            st.success("🎉 全レースの予想が完了しました！")

    elif action == "🔍 レースを指定して予想":
        st.subheader("🎯 レースを指定")
        options = [f"{r['place']} {r['num']}R (発走 {r['time'].strftime('%H:%M')}) - {r['title']}" for r in todays_races]
        selected = st.selectbox("レースを選んでください", options)
        
        idx = options.index(selected)
        target_race = todays_races[idx]
        
        if st.button("🚀 予想開始", type="primary"):
            with st.spinner('解析中...'):
                res_df = run_prediction(target_race['id'])
                st.dataframe(res_df.style.format({'AI勝率': '{:.1f}%', '期待値': '{:.2f}'}), use_container_width=True)