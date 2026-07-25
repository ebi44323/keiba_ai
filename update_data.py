"""
update_data.py - keiba-ebye 週次データ自動更新スクリプト v2.0
================================================================
【hokan.py から修正した主な点】
  1. 血統取得URL: /horse/{id} → /horse/ped/{id}/ (fix2.pyの正解実装を採用)
  2. 血統セレクタ: rowspan='4'→'16', rowspan='2'→'8'
  3. 全派生特徴量を計算
     (乗り替わり/馬場替わり/フラグ類/上り偏差/補正タイム偏差/穴馬フラグ 等)
  4. 即sys.exit() → リトライ付き寛容なエラーハンドリングに変更
  5. レース名・前半3F・後半3F の取得追加

【使い方】
  python update_data.py          # 先週1週間分を更新
  python update_data.py 2        # 先々週まで遡る
  python update_data.py --all    # 全未取得日を取得 (初回構築時)
================================================================
"""

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
import datetime
import time
import zipfile
import os
import sys
import sqlite3

# src パッケージを import できるようにプロジェクトルートを path に追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.config import PLACE_DICT, VENUE_MAWARI, VENUE_CHIKEI, get_headers, safe_sleep

# ================================================================
# 設定
# ================================================================
CSV_FILE  = 'learning_data_perfect_tier.csv'
ZIP_FILE  = 'learning_data_perfect_tier.zip'
PED_CSV   = 'pedigree_master_all.csv'

# PLACE_DICT, VENUE_MAWARI, VENUE_CHIKEI, get_headers, safe_sleep は src/config.py から import 済み


# ================================================================
# 血統取得 (fix2.py の正解実装を採用)
# ================================================================
def get_pedigree(horse_id, retry=3):
    empty = {'馬ID':horse_id,'父':'不明','父系':'不明','母':'不明','母系':'不明','母父':'不明','母父系':'不明'}
    for attempt in range(retry):
        try:
            # ★修正: /horse/ped/{id}/ が正解 (hokan.pyは /horse/{id} で間違っていた)
            r = requests.get(f"https://db.netkeiba.com/horse/ped/{horse_id}/",
                             headers=get_headers(), timeout=12)
            r.encoding = 'euc-jp'
            if r.status_code in (403, 503):
                print(f"    ⚠️ HTTP {r.status_code} (attempt {attempt+1}/{retry}) 待機中...")
                time.sleep(12 * (attempt + 1))
                continue
            soup = BeautifulSoup(r.text, 'html.parser')
            table = soup.find('table', class_='blood_table')
            if not table:
                return empty  # 未登録馬
            # ★修正: rowspan='16'(父・母) / rowspan='8'(祖父母層) が正解
            tds_16 = [td for td in table.find_all('td') if td.get('rowspan') == '16']
            tds_8  = [td for td in table.find_all('td') if td.get('rowspan') == '8']

            def extract(td):
                if not td: return '不明', '不明'
                a = td.find('a')
                name = re.sub(r'\s+', '', a.text) if a else re.sub(r'\s+', '', td.text)
                m = re.search(r'([A-Za-z0-9.\-]+系|[\u30A0-\u30FF]+系)', re.sub(r'\s+','',td.text))
                return name, (m.group(1) if m else '不明')

            sire, sire_sys = extract(tds_16[0]) if len(tds_16) >= 1 else ('不明','不明')
            dam,  _        = extract(tds_16[1]) if len(tds_16) >= 2 else ('不明','不明')
            bms,  bms_sys  = extract(tds_8[2])  if len(tds_8)  >= 3 else ('不明','不明')
            fno_m  = re.search(r'F-?No\.?\s*\[?([a-zA-Z0-9\-]+)\]?', table.text)
            dam_sys = f"FNo.[{fno_m.group(1)}]" if fno_m else '不明'

            return {'馬ID':horse_id,'父':sire,'父系':sire_sys,'母':dam,
                    '母系':dam_sys,'母父':bms,'母父系':bms_sys}
        except Exception as e:
            print(f"    ⚠️ 血統取得エラー {horse_id} (attempt {attempt+1}/{retry}): {e}")
            time.sleep(5)
    return empty


# ================================================================
# 血統 sqlite キャッシュ（一度取得した血統はDBに保存し再取得をスキップ）
# ================================================================
_PED_CACHE_DB = 'ped_cache.db'


def _init_ped_cache():
    conn = sqlite3.connect(_PED_CACHE_DB)
    conn.execute('''CREATE TABLE IF NOT EXISTS pedigree (
        horse_id TEXT PRIMARY KEY,
        sire TEXT, sire_sys TEXT, dam TEXT, dam_sys TEXT,
        bms TEXT, bms_sys TEXT, fetched_at TEXT
    )''')
    conn.commit()
    conn.close()


def _get_pedigree_cached(horse_id, retry=3):
    """sqlite キャッシュ付き血統取得。キャッシュヒット時は Web 取得をスキップ。"""
    conn = sqlite3.connect(_PED_CACHE_DB)
    row = conn.execute(
        'SELECT sire, sire_sys, dam, dam_sys, bms, bms_sys FROM pedigree WHERE horse_id=?',
        (horse_id,)
    ).fetchone()
    conn.close()
    if row:
        return {'馬ID': horse_id, '父': row[0], '父系': row[1], '母': row[2],
                '母系': row[3], '母父': row[4], '母父系': row[5]}
    result = get_pedigree(horse_id, retry)
    conn = sqlite3.connect(_PED_CACHE_DB)
    conn.execute(
        'INSERT OR REPLACE INTO pedigree VALUES (?,?,?,?,?,?,?,?)',
        (horse_id, result['父'], result['父系'], result['母'],
         result['母系'], result['母父'], result['母父系'],
         datetime.datetime.now().isoformat())
    )
    conn.commit()
    conn.close()
    return result


# ================================================================
# レースID 一覧取得
# ================================================================
def get_race_ids_for_date(date_str):
    race_ids = []
    for url in [
        f'https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={date_str}',
        f'https://race.netkeiba.com/top/race_list.html?kaisai_date={date_str}',
    ]:
        try:
            r = requests.get(url, headers=get_headers(), timeout=10); r.encoding='euc-jp'
            print(f"    [net] {url.split('?')[0].split('/')[-1]} → HTTP {r.status_code}")
            for a in BeautifulSoup(r.text,'html.parser').find_all('a', href=re.compile(r'race_id=(\d{12})')):
                rid = re.search(r'race_id=(\d{12})',a['href']).group(1)
                if 1 <= int(rid[4:6]) <= 10: race_ids.append(rid)
            if race_ids:
                print(f"    [net] → {len(race_ids)}件取得")
                break
            print(f"    [net] → レースID 0件（レスポンス先頭: {r.text[:80].strip()!r}）")
        except Exception as e:
            print(f"    [net] → エラー: {e}")
        safe_sleep(1.0, 0.5)
    if not race_ids:
        try:
            url_db = f'https://db.netkeiba.com/race/list/{date_str}/'
            r = requests.get(url_db, headers=get_headers(), timeout=10)
            r.encoding = 'euc-jp'
            print(f"    [net] db.netkeiba fallback → HTTP {r.status_code}")
            for m in re.findall(r'/race/(\d{12})', r.text):
                if 1 <= int(m[4:6]) <= 10: race_ids.append(m)
            if not race_ids:
                print(f"    [net] → fallbackも0件（先頭: {r.text[:80].strip()!r}）")
        except Exception as e:
            print(f"    [net] → fallbackエラー: {e}")
    return sorted(list(set(race_ids)))


# ================================================================
# レース結果スクレイプ（1レース分）
# ================================================================
def scrape_one_race(rid, date_str):
    rows = []
    try:
        r = requests.get(f"https://db.netkeiba.com/race/{rid}/", headers=get_headers(), timeout=15)
        r.encoding = 'euc-jp'
        soup = BeautifulSoup(r.text, 'html.parser')

        table = (soup.find('table', class_='race_table_01') or
                 soup.select_one('#All_Result_Table') or
                 soup.select_one('.RaceTable01'))
        if not table: return []

        data_box = (soup.find('div',class_='data_intro') or
                    soup.find('div',class_='RaceData01') or
                    soup.find('dl',class_='racedata'))
        rt = data_box.text.replace('\n','') if data_box else ''

        tdm    = re.search(r'(芝|ダ|障|障害).*?(\d+)m', rt)
        ttype  = '芝' if tdm and tdm.group(1)=='芝' else 'ダート' if tdm and 'ダ' in tdm.group(1) else '障害'
        dist   = int(tdm.group(2)) if tdm else 1600
        baba   = (re.search(r'馬場:([良稍重不良]+)',rt) or type('',(),{'group':lambda s,i:''})()).group(1) or '良'
        tenki  = (re.search(r'天候:([晴曇雨雪小雨小雪]+)',rt) or type('',(),{'group':lambda s,i:''})()).group(1) or '晴'
        place  = PLACE_DICT.get(str(rid)[4:6],'不明')

        rname_tag = soup.find('h1',class_='RaceName') or soup.find('div',class_='RaceName') or soup.find('h1')
        race_name = re.sub(r'\s+','',rname_tag.text) if rname_tag else f"{place}{int(rid[10:12])}R"

        # ── ペース・ラップタイム取得 ────────────────────────────────
        # 前半3F/前半ペース値 = 前半最初の1F(200m)ラップタイム
        # 後半3F              = 後半ラスト2F(400m)合計
        # 後半ペース値        = 後半600mの合計タイム
        # ※ netkeibaの summary='ペース' は現在のページでは存在しない場合が多い
        zenhan = kohan = zenhan_pace = kohan_pace_val = np.nan

        def extract_laps(s):
            """
            ラップタイムをページから取得する。
            複数セレクタを試みてラップ列を見つける。
            返値: list of float (1F=200mごとのタイム) または []
            """
            # 方法1: summary='ペース' テーブル（旧形式）
            ptbl = s.find('table', summary='ペース')
            if ptbl:
                full_text = ptbl.get_text()
                nums = re.findall(r'\d{1,2}\.\d', full_text)
                if len(nums) >= 4:
                    return [float(x) for x in nums]

            # 方法2: class に 'lap' 'pace' 'corner' を含むテーブル
            for tbl in s.find_all('table'):
                cls = ' '.join(tbl.get('class', [])).lower()
                if any(k in cls for k in ['lap', 'pace', 'corner', 'time']):
                    nums = re.findall(r'\d{1,2}\.\d', tbl.get_text())
                    if len(nums) >= 4:
                        # タイム列との区別: ラップは10-15秒程度
                        laps = [float(n) for n in nums if 9.0 <= float(n) <= 16.0]
                        if len(laps) >= 4:
                            return laps

            # 方法3: ページ全体からラップタイム列パターンを抽出
            # 形式: "12.3 - 11.8 - 12.0 - 11.9 - 11.5 - 11.7" 等
            page_text = s.get_text()
            # ハイフン区切りの連続するラップタイム
            for pattern in [
                r'(\d{1,2}\.\d)(?:\s*[-－]\s*\d{1,2}\.\d){3,}',
                r'(\d{1,2}\.\d)(?:\s+\d{1,2}\.\d){3,}',
            ]:
                m = re.search(pattern, page_text)
                if m:
                    laps = [float(x) for x in re.findall(r'\d{1,2}\.\d', m.group(0))]
                    if len(laps) >= 4 and all(9.0 <= x <= 16.0 for x in laps):
                        return laps

            return []

        try:
            laps = extract_laps(soup)
            if laps and len(laps) >= 4:
                half = len(laps) // 2
                zenhan        = laps[0]                    # 前半最初の1Fタイム
                zenhan_pace   = laps[0]                    # 前半ペース値 = 同じ
                kohan_total   = sum(laps[half:half+3])     # 後半最初の3F合計 = 後半ペース値
                kohan         = sum(laps[-2:])             # 後半ラスト2F = 後半3F
                kohan_pace_val = kohan_total
            # ラップ取得失敗時は上り(上がり3F)から推算する（行レベルで後で適用）
        except:
            pass

        ths = [th.text.strip().replace('\n','') for th in table.find_all('th')]
        def gi(kws):
            for i,h in enumerate(ths):
                if any(k in h for k in kws): return i
            return -1

        rank_i=gi(['着順']); waku_i=gi(['枠']); uma_i=gi(['馬番']); name_i=gi(['馬名'])
        sex_i=gi(['性齢']); kin_i=gi(['斤量']); jock_i=gi(['騎手'])
        time_i=gi(['タイム','走破']); diff_i=gi(['着差']); odds_i=gi(['単勝','オッズ'])
        pop_i=gi(['人気']); wgt_i=gi(['馬体重']); trnr_i=gi(['調教師'])
        tsuka_i=gi(['通過','コーナー']); agari_i=gi(['上り','上がり','3F'])

        for tr in table.find_all('tr')[1:]:
            tds = tr.find_all('td')
            if len(tds) < 5: continue
            try:
                def g(i): return tds[i].text.strip() if i!=-1 and i<len(tds) else ''
                chaku_s = g(rank_i)
                if not re.match(r'^\d+$', chaku_s): continue

                horse_a = tds[name_i].find('a', href=re.compile(r'/horse/\d+')) if name_i!=-1 and name_i<len(tds) else None
                jock_a  = tds[jock_i].find('a') if jock_i!=-1 and jock_i<len(tds) else None
                trnr_a  = tds[trnr_i].find('a') if trnr_i!=-1 and trnr_i<len(tds) else None
                if not horse_a: continue

                horse_id  = re.search(r'\d+', horse_a['href']).group(0).zfill(10)
                jock_id   = re.search(r'\d+', jock_a['href']).group(0) if jock_a else '0'
                kin_m     = re.search(r'\d+(\.\d+)?', g(kin_i))
                wgt_text  = g(wgt_i)
                wgt_m     = re.search(r'(\d{3})\(([+\-]?\d+)\)', wgt_text)

                rows.append({
                    '着順':g(rank_i), '枠番':g(waku_i), '馬番':g(uma_i),
                    '馬名':horse_a.text.strip(), '性齢':g(sex_i),
                    '斤量':float(kin_m.group(0)) if kin_m else 55.0,
                    '騎手':jock_a.text.strip() if jock_a else g(jock_i),
                    'タイム':g(time_i), '着差':g(diff_i), '単勝':g(odds_i), '人気':g(pop_i),
                    '馬体重':wgt_text,
                    '当日馬体重': int(wgt_m.group(1)) if wgt_m else 0,
                    '馬体重増減': int(wgt_m.group(2)) if wgt_m else 0,
                    '調教師': trnr_a.text.strip() if trnr_a else g(trnr_i),
                    '馬ID':horse_id, '騎手ID':jock_id, 'レースID':str(rid),
                    '日付':f"{date_str[:4]}/{int(date_str[4:6]):02d}/{int(date_str[6:8]):02d}",
                    '競馬場':place, '芝/ダート':ttype, '距離':dist,
                    '天候':tenki, '馬場':baba, 'レース名':race_name,
                    '通過':g(tsuka_i), '上り':g(agari_i),
                    '前半3F':zenhan, '後半3F':kohan,
                    '前半ペース値':zenhan_pace, '後半ペース値':kohan_pace_val,
                    '回り':VENUE_MAWARI.get(place,'不明'),
                    'コース地形':VENUE_CHIKEI.get(place,'不明'),
                })
            except: continue
    except Exception as e:
        print(f"  スクレイプエラー {rid}: {e}")
    return rows


# ================================================================
# 全派生特徴量の計算
# ================================================================
def compute_features(df):
    df = df.copy()

    # 数値変換（全列を明示的に変換 — df_existingがstr読み込みの場合も安全に処理）
    for c in ['着順','単勝','人気','斤量','距離','上り','枠番','馬番',
              '当日馬体重','馬体重増減','前半3F','後半3F','前半ペース値','後半ペース値']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df['日付'] = pd.to_datetime(df['日付'], format='mixed', errors='coerce')

    def t2s(t):
        try:
            m = re.match(r'(\d+):(\d+\.\d+)', str(t))
            return float(m.group(1))*60+float(m.group(2)) if m else float(t)
        except: return np.nan
    df['走破タイム秒'] = df['タイム'].apply(t2s)

    # 基本集計
    df['出走頭数']      = df.groupby('レースID')['馬ID'].transform('count')
    df['着順パーセント'] = (df['着順']-1)/(df['出走頭数']-1).replace(0,1)
    df['rank_label']    = (df['着順']<=3).astype(int)

    # コース統計・スピード指数
    # ★修正: マージキーの型を統一してからマージ（型不一致で列が欠落するバグを防ぐ）
    for _c in ['競馬場', '芝/ダート', '距離']:
        df[_c] = df[_c].astype(str).str.strip()
    df['距離'] = pd.to_numeric(df['距離'], errors='coerce')  # 数値に戻す（後続処理用）
    # マージキー用に文字列版を作成
    df['_距離str'] = df['距離'].astype(str)

    cs = (df.groupby(['競馬場', '芝/ダート', '_距離str'])['走破タイム秒']
          .agg(['mean', 'std']).reset_index()
          .rename(columns={'mean': 'コース平均', 'std': 'コース標準偏差'}))
    # 既存列があれば先に除去（二重マージ防止）
    for _c in ['コース平均', 'コース標準偏差']:
        if _c in df.columns:
            df = df.drop(columns=[_c])
    df = pd.merge(df, cs, on=['競馬場', '芝/ダート', '_距離str'], how='left')
    df = df.drop(columns=['_距離str'])

    # ★修正: マージ後に列が存在しない場合の安全フォールバック
    if 'コース標準偏差' not in df.columns:
        df['コース平均'] = df['走破タイム秒'].mean()
        df['コース標準偏差'] = df['走破タイム秒'].std()
    # std が NaN（グループ1件のみ）の場合は全体平均stdで補完
    overall_std = df['走破タイム秒'].std()
    df['コース標準偏差'] = df['コース標準偏差'].fillna(overall_std)
    df['コース平均'] = df['コース平均'].fillna(df['走破タイム秒'].mean())

    df['スピード指数'] = np.where(
        df['コース標準偏差'] > 0,
        50 - ((df['走破タイム秒'] - df['コース平均']) / df['コース標準偏差']) * 10,
        50
    )

    # タイム関連
    first_t = df[df['着順']==1].groupby('レースID')['走破タイム秒'].min().to_dict()
    df['1着タイム']         = df['レースID'].map(first_t)
    df['タイム差']          = df['走破タイム秒'] - df['1着タイム']
    df['距離補正タイム差']  = df['タイム差'] * (1000/df['距離'].replace(0,1))
    df['レース平均タイム']  = df.groupby('レースID')['走破タイム秒'].transform('mean')
    df['補正タイム偏差']    = df['走破タイム秒'] - df['レース平均タイム']
    df['レース平均斤量']    = df.groupby('レースID')['斤量'].transform('mean')
    df['斤量差']            = df['斤量'] - df['レース平均斤量']
    df['レース平均上り']    = df.groupby('レースID')['上り'].transform('mean')
    df['上り偏差']          = df['上り'] - df['レース平均上り']

    # コーナー
    def first_corner(x):
        s = str(x)
        parts = [p for p in s.split('-') if p.strip().isdigit()]
        return int(parts[0]) if parts else np.nan
    df['最初のコーナー順位'] = df['通過'].apply(first_corner)
    df['失速フラグ']          = (df['上り偏差'] > 1.5).astype(int)

    # レース内先行馬数
    df['レース内先行馬数'] = df.groupby('レースID')['最初のコーナー順位'].transform(lambda x: (pd.to_numeric(x, errors='coerce') <= 5).sum())

    # 馬ごと時系列
    df = df.sort_values(['馬ID','日付']).reset_index(drop=True)
    df['前走着順']          = df.groupby('馬ID')['着順'].shift(1)
    df['前走日付']          = df.groupby('馬ID')['日付'].shift(1)
    df['出走間隔']          = (df['日付']-df['前走日付']).dt.days
    df['前走コーナー順位']  = df.groupby('馬ID')['最初のコーナー順位'].shift(1)
    df['前走上り偏差']      = df.groupby('馬ID')['上り偏差'].shift(1)
    df['前走失速フラグ']    = df.groupby('馬ID')['失速フラグ'].shift(1)
    df['前走着順パーセント']= df.groupby('馬ID')['着順パーセント'].shift(1)
    df['前走距離補正タイム差'] = df.groupby('馬ID')['距離補正タイム差'].shift(1)
    df['前走芝ダート']      = df.groupby('馬ID')['芝/ダート'].shift(1)
    df['前走距離']          = df.groupby('馬ID')['距離'].shift(1)
    df['前走騎手ID']        = df.groupby('馬ID')['騎手ID'].shift(1)

    # フラグ
    df['乗り替わりフラグ']  = (df['騎手ID'].astype(str)!=df['前走騎手ID'].astype(str)).fillna(False).astype(int)
    df['馬場替わりフラグ']  = (df['芝/ダート']!=df['前走芝ダート']).fillna(False).astype(int)
    df['距離変更フラグ']    = (df['距離']!=df['前走距離']).fillna(False).astype(int)
    n = df['出走頭数'].replace(0,1)
    df['前走大敗フラグ']    = (df['前走着順']/n > 0.7).fillna(False).astype(int)

    # 直近3走
    df['直近3走平均着順']       = df.groupby('馬ID')['着順'].transform(lambda x: x.shift(1).rolling(3,min_periods=1).mean())
    df['直近3走着順パーセント'] = df.groupby('馬ID')['着順パーセント'].transform(lambda x: x.shift(1).rolling(3,min_periods=1).mean())

    # 穴馬フラグ
    df['穴馬_距離変更一変']     = ((df['距離変更フラグ']==1)&(df['直近3走着順パーセント']<0.4)).astype(int)
    df['穴馬_馬場替わり一変']   = ((df['馬場替わりフラグ']==1)&(df['直近3走着順パーセント']<0.4)).astype(int)
    df['穴馬_勝負の乗り替わり'] = ((df['乗り替わりフラグ']==1)&(df['直近3走着順パーセント'].fillna(0.5)<0.5)).astype(int)
    df['穴馬_実力馬の巻き返し'] = ((df['前走大敗フラグ']==1)&(df['直近3走着順パーセント']<0.4)).astype(int)

    # 複合キー
    df['調教師_騎手'] = df['調教師'].astype(str)+'_'+df['騎手ID'].astype(str)
    df['騎手_競馬場'] = df['騎手ID'].astype(str)+'_'+df['競馬場'].astype(str)
    df['騎手_距離']   = df['騎手ID'].astype(str)+'_'+df['距離'].astype(str)

    # 日付を文字列に戻す
    df['日付'] = df['日付'].dt.strftime('%Y/%m/%d')
    return df


# ================================================================
# メイン処理
# ================================================================
def main():
    print("="*60)
    print("keiba-ebye 週次データ更新スクリプト v2.0")
    print("="*60)
    _init_ped_cache()

    weeks_back = 1
    fetch_all  = False
    from_date  = None   # --from YYYYMMDD で指定
    if len(sys.argv) > 1:
        if sys.argv[1] == '--all':
            fetch_all = True
            print("🔍 全未取得日付モード（ギャップ検出）")
        elif sys.argv[1] == '--from' and len(sys.argv) > 2:
            try:
                from_date = datetime.datetime.strptime(sys.argv[2], '%Y%m%d').date()
                print(f"🔍 指定日付モード: {from_date} 以降")
            except ValueError:
                print(f"⚠️ --from の日付形式が不正です: {sys.argv[2]} (YYYYMMDD で指定)")
                sys.exit(1)
        else:
            try: weeks_back = int(sys.argv[1])
            except: pass

    # 既存データ読み込み
    df_existing, existing_ids = None, set()
    for path in [ZIP_FILE, CSV_FILE]:
        if os.path.exists(path):
            kw = {'compression':'zip'} if path.endswith('.zip') else {}
            print(f"📊 既存データ読み込み: {path}")
            df_existing = pd.read_csv(path, dtype=str, **kw)
            # 指数表記バグ修正
            df_existing['レースID'] = df_existing['レースID'].apply(
                lambda x: str(int(float(x))).zfill(12) if re.match(r'[\d.]+[Ee][+\-]?\d+', str(x)) else str(x))
            df_existing['馬ID'] = df_existing['馬ID'].astype(str).str.replace(r'\.0$','',regex=True).str.zfill(10)
            # JRAのみ
            def is_jra(rid):
                s=str(rid).strip()
                return len(s)==12 and s[4:6].isdigit() and 1<=int(s[4:6])<=10
            df_existing = df_existing[df_existing['レースID'].apply(is_jra)].copy()
            existing_ids = set(df_existing['レースID'].tolist())
            print(f"  → {len(df_existing):,}行 / {len(existing_ids):,}レース")
            break

    # 血統マスター読み込み
    ped_dict = {}
    if os.path.exists(PED_CSV):
        df_ped = pd.read_csv(PED_CSV, dtype=str).fillna('不明')
        df_ped['馬ID'] = df_ped['馬ID'].astype(str).str.replace(r'\.0$','',regex=True).str.zfill(10)
        unkn = (df_ped['父']=='不明').mean()
        if unkn > 0.2:
            print(f"⚠️ 血統マスターに「不明」が{unkn*100:.0f}%あります。fix2.py での修復を推奨します。")
        ped_dict = df_ped.set_index('馬ID').to_dict('index')
        print(f"📜 血統マスター: {len(ped_dict):,}頭")
    else:
        print(f"⚠️ {PED_CSV} が見つかりません。血統はWeb取得のみになります。")

    # 取得対象日付の決定
    today = datetime.date.today()
    if from_date is not None:
        # --from YYYYMMDD モード: 指定日以降の全土日（既存IDはスクレイプ時にスキップ）
        target_dates = [
            d.date() for d in pd.date_range(start=from_date, end=today)
            if d.weekday() in (5, 6)
        ]
    elif fetch_all and df_existing is not None:
        # ギャップ検出モード: 既存データの最古日から今日まで、土日で未収録の日付を全部対象にする
        existing_dates = set(
            pd.to_datetime(df_existing['日付'], errors='coerce').dt.date.dropna().unique()
        )
        data_start = min(existing_dates)
        all_weekends = [
            d.date() for d in pd.date_range(start=data_start, end=today)
            if d.weekday() in (5, 6)
        ]
        target_dates = [d for d in all_weekends if d not in existing_dates]
        if not target_dates:
            # ギャップなし → 最新日以降を探す（新規データ取得）
            last_dt = pd.to_datetime(df_existing['日付'], errors='coerce').max()
            start_d = (last_dt + pd.Timedelta(days=1)).date()
            target_dates = [d.date() for d in pd.date_range(start=start_d, end=today) if d.weekday() in (5,6)]
        else:
            print(f"⚠️  {len(target_dates)} 日分のギャップを検出: {[str(d) for d in target_dates]}")
    else:
        target_dates = []
        for w in range(weeks_back):
            base = today - datetime.timedelta(weeks=w)
            sat  = base - datetime.timedelta(days=(base.weekday()-5)%7)
            for d in [sat, sat+datetime.timedelta(days=1)]:
                if d <= today: target_dates.append(d)
        target_dates = sorted(set(target_dates))

    if not target_dates:
        print("✅ 取得対象日なし（データは最新です）")
    else:
        print(f"\n📅 取得対象: {[d.strftime('%Y/%m/%d') for d in target_dates]}")

    # スクレイプ
    all_new_rows = []
    for d in target_dates:
        date_str = d.strftime('%Y%m%d')
        print(f"\n--- {d.strftime('%Y/%m/%d')} ---")
        race_ids = get_race_ids_for_date(date_str)
        print(f"  レース数: {len(race_ids)}")
        for rid in race_ids:
            if rid in existing_ids:
                print(f"  スキップ（既存）: {rid}")
                continue
            print(f"  取得中: {rid}")
            rows = scrape_one_race(rid, date_str)
            if rows:
                all_new_rows.extend(rows)
                print(f"    → {len(rows)}頭")
            safe_sleep(2.0, 1.5)

    print(f"\n新規データ: {len(all_new_rows)}行")
    if not all_new_rows:
        print("新しいデータはありませんでした。")
        return

    df_new_raw = pd.DataFrame(all_new_rows)

    # 血統付与
    print("\n🧬 血統データを照合中...")
    unique_hids = [h for h in df_new_raw['馬ID'].unique() if h not in ped_dict and h != '0000000000']
    new_peds = []
    for i, hid in enumerate(unique_hids):
        print(f"  血統取得 {i+1}/{len(unique_hids)}: {hid}")
        ped = _get_pedigree_cached(hid)
        ped_dict[hid] = ped
        new_peds.append(ped)
        safe_sleep(1.5, 0.5)

    for col in ['父','父系','母','母系','母父','母父系']:
        df_new_raw[col] = df_new_raw['馬ID'].map(lambda hid: ped_dict.get(hid, {}).get(col, '不明'))

    # 全体結合してから特徴量計算（前走系は全期間の時系列が必要）
    if df_existing is not None:
        df_combined = pd.concat([df_existing, df_new_raw.astype(str)], ignore_index=True)
        df_combined = df_combined.drop_duplicates(subset=['レースID','馬ID'], keep='last')
    else:
        df_combined = df_new_raw.astype(str)

    # ★修正: compute_features に渡す前に数値列を正しく変換する
    # （df_existing が dtype=str で読まれているため全列が文字列になっている）
    for _num_col in ['着順','単勝','人気','斤量','距離','上り','枠番','馬番',
                     '当日馬体重','馬体重増減','前半3F','後半3F','前半ペース値','後半ペース値']:
        if _num_col in df_combined.columns:
            df_combined[_num_col] = pd.to_numeric(df_combined[_num_col], errors='coerce')
    # 指数表記バグ修正（レースID/馬IDが "1.23e+11" 形式になるケース）
    df_combined['レースID'] = df_combined['レースID'].apply(
        lambda x: str(int(float(x))).zfill(12)
        if re.match(r'[\d.]+[Ee][+\-]?\d+', str(x)) else str(x)
    )
    df_combined['馬ID'] = df_combined['馬ID'].astype(str).str.replace(r'\.0$','',regex=True).str.zfill(10)

    print("\n⚙️  特徴量計算中...")
    df_final = compute_features(df_combined)
    df_final = df_final.sort_values(['日付','レースID','馬番'], na_position='last').reset_index(drop=True)

    # 保存
    print(f"\n💾 保存中: {len(df_final):,}行")
    df_final.to_csv(CSV_FILE, index=False, encoding='utf-8-sig')
    with zipfile.ZipFile(ZIP_FILE, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(CSV_FILE)
    print(f"✅ {CSV_FILE} + {ZIP_FILE} 保存完了")

    # 血統マスター更新
    if new_peds:
        print(f"\n💾 血統マスター更新: {len(new_peds)}頭追記")
        if os.path.exists(PED_CSV):
            df_ped_old = pd.read_csv(PED_CSV, dtype=str)
            df_ped_new = pd.concat([df_ped_old, pd.DataFrame(new_peds)], ignore_index=True)
            df_ped_new = df_ped_new.drop_duplicates(subset=['馬ID'], keep='last')
        else:
            df_ped_new = pd.DataFrame(new_peds)
        df_ped_new.to_csv(PED_CSV, index=False, encoding='utf-8-sig')
        print(f"✅ {PED_CSV} 更新完了")

    print("\n"+"="*60)
    print("🎉 更新完了！")
    print("="*60)


# ================================================================
# テスト用: ドライラン（実際には保存しない）
# ================================================================
def dry_run_test(date_str=None):
    """
    update_data.py --test で呼び出し。
    指定日（デフォルト: 先週の土曜日）のレースIDとレース結果を1レース分だけ取得して
    スクレイピングが正常に動くか確認する。保存は一切行わない。
    """
    if date_str is None:
        today = datetime.date.today()
        # 直近の土曜日
        sat = today - datetime.timedelta(days=(today.weekday() - 5) % 7 + 7)
        date_str = sat.strftime('%Y%m%d')

    print(f"\n{'='*60}")
    print(f"🧪 ドライランテスト: {date_str[:4]}/{date_str[4:6]}/{date_str[6:]}")
    print(f"{'='*60}")

    # Step1: レースID取得
    print("\n📋 Step1: レースID取得")
    race_ids = get_race_ids_for_date(date_str)
    if not race_ids:
        print("❌ レースIDが見つかりません（開催なし or ネットワークエラー）")
        return False
    print(f"✅ {len(race_ids)}件のレースIDを取得: {race_ids[:3]}...")

    # Step2: 1レース分の結果スクレイプ
    test_rid = race_ids[0]
    print(f"\n🐎 Step2: レース結果スクレイプ (先頭1件: {test_rid})")
    rows = scrape_one_race(test_rid, date_str)
    if not rows:
        print("❌ レース結果が取得できませんでした")
        return False
    print(f"✅ {len(rows)}頭分のデータを取得")
    print(f"   サンプル: {rows[0]['馬名']} / {rows[0]['着順']}着 / {rows[0]['タイム']}")
    print(f"   取得列数: {len(rows[0])}列")

    # Step3: 血統取得テスト（1頭だけ）
    test_hid = rows[0]['馬ID']
    print(f"\n🧬 Step3: 血統取得テスト (馬ID: {test_hid})")
    ped = get_pedigree(test_hid, retry=1)
    if ped['父'] == '不明':
        print(f"⚠️  血統取得失敗（未登録馬 or ネットワーク問題）")
    else:
        print(f"✅ 血統取得成功: 父={ped['父']} ({ped['父系']}) / 母父={ped['母父']}")

    # Step4: 特徴量計算テスト
    print(f"\n⚙️  Step4: 特徴量計算テスト")
    df_test = pd.DataFrame(rows)
    df_test['父'] = ped['父']; df_test['父系'] = ped['父系']
    df_test['母'] = ped['母']; df_test['母系'] = ped['母系']
    df_test['母父'] = ped['母父']; df_test['母父系'] = ped['母父系']
    try:
        df_feat = compute_features(df_test)
        print(f"✅ 特徴量計算成功: {len(df_feat.columns)}列")
        key_cols = ['走破タイム秒','スピード指数','着順パーセント','rank_label']
        for c in key_cols:
            val = df_feat[c].iloc[0] if c in df_feat.columns else 'なし'
            print(f"   {c}: {val}")
    except Exception as e:
        print(f"❌ 特徴量計算エラー: {e}")
        return False

    print(f"\n{'='*60}")
    print("✅ ドライランテスト完了！スクレイピング・特徴量計算とも正常です。")
    print("   本番実行: python update_data.py")
    print(f"{'='*60}\n")
    return True


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        date_arg = sys.argv[2] if len(sys.argv) > 2 else None
        dry_run_test(date_arg)
    else:
        main()
