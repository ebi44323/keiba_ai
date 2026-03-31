import requests
from bs4 import BeautifulSoup
import re
import datetime
import pytz
import logging
import pandas as pd
import numpy as np
import streamlit as st
from src.config import PLACE_DICT, get_headers

logger = logging.getLogger('keiba_ebye')

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
            res = requests.get(url, headers=get_headers(), timeout=10)
            soup = BeautifulSoup(res.content, 'html.parser', from_encoding='euc-jp')
            
            for a_tag in soup.find_all('a', href=re.compile(r'race_id=(\d{12})')):
                r_id = re.search(r'race_id=(\d{12})', a_tag.get('href')).group(1)
                if not (1 <= int(r_id[4:6]) <= 10): continue
                if r_id in added_ids: continue
                added_ids.add(r_id)
                
                place = PLACE_DICT.get(r_id[4:6], '不明')
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
        except Exception as _e:
            logger.warning(f'get_todays_races スクレイプ失敗 url={url}: {_e}')
        if races: break

    if not races:
        url = f'https://db.netkeiba.com/race/list/{target_date_str}/'
        try:
            res = requests.get(url, headers=get_headers(), timeout=10)
            res.encoding = 'euc-jp'
            soup = BeautifulSoup(res.content, 'html.parser', from_encoding='euc-jp')
            ids = set(re.findall(r'/race/(\d{12})', res.text))
            for r_id in ids:
                if not (1 <= int(r_id[4:6]) <= 10): continue
                place = PLACE_DICT.get(r_id[4:6], '不明')
                r_num = int(r_id[10:12])
                dummy_time = tokyo_tz.localize(datetime.datetime.strptime(f"{target_date_str} 12:00", "%Y%m%d %H:%M"))
                races.append({'id': r_id, 'place': place, 'num': r_num, 'title': f"{place} {r_num}R", 'time': dummy_time, 'sort_key': f"{r_id[4:6]}{r_num:02d}"})
        except Exception as _e:
            logger.warning(f'get_todays_races db.netkeiba スクレイプ失敗: {_e}')
    return sorted(races, key=lambda x: x['sort_key'])

def get_weekend_dates():
    """今週末（月〜日曜視点で直近の土・日）の日付を返す。
    月〜土: 当週の土日
    日曜:   当日(日)と翌週の土 ではなく → 当日(日)を「今週日曜」として当週の土日を返す
    """
    tokyo_tz = pytz.timezone('Asia/Tokyo')
    now = datetime.datetime.now(tokyo_tz)
    wd = now.weekday()  # 月=0 ... 土=5, 日=6

    if wd == 6:
        # 日曜日: 今日が日曜なので「今週の土曜(昨日)」と「今週の日曜(今日)」
        saturday = now - datetime.timedelta(days=1)
        sunday   = now
    else:
        # 月〜土: 今週の土曜・日曜
        days_to_sat = 5 - wd          # 土曜まであと何日（月なら5, 土なら0）
        saturday = now + datetime.timedelta(days=days_to_sat)
        sunday   = saturday + datetime.timedelta(days=1)

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
        except Exception as _e:
            logger.warning(f'get_payouts 失敗 url={url}: {_e}')
    return tansho_dict, fukusho_dict

def get_all_payouts(race_id):
    payouts = {'tansho': {}, 'fukusho': {}, 'umaren': {}, 'wide': {}, 'sanrenpuku': {}}
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
                        elif 'sanrenpuku' in th_class or '三連複' in th_text: current_kind = '三連複'
                        else: current_kind = None

                    if not current_kind: continue
                    
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
                        elif current_kind == '三連複' and len(nums) >= 3: payouts['sanrenpuku'][tuple(sorted(nums[:3]))] = pay

            if payouts['tansho'] and payouts['wide']: return payouts
        except Exception as _e:
            logger.warning(f'get_all_payouts netkeiba失敗: {_e}')

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
                if th_text in ['単勝', '複勝', '馬連', 'ワイド', '三連複']: current_kind = th_text
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
                elif current_kind == '三連複' and len(nums) >= 3: payouts['sanrenpuku'][tuple(sorted(nums[:3]))] = pay
    except Exception as _e:
        logger.warning(f'get_all_payouts Yahoo解析失敗: {_e}')

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
    except Exception as _e:
        logger.warning(f'get_odds_from_soup 解析エラー: {_e}')
    return o_dict

def fetch_odds_realtime(race_id: str) -> tuple[dict, dict]:
    """
    単勝オッズのみを素早く取得する軽量関数。全推論は実行しない。
    Returns:
        odds_dict:      {馬番(int): オッズ(float)}
        name_odds_dict: {馬名(str): オッズ(float)}  ※APIに馬名が含まれる場合のみ
    """
    import json as _json
    odds_dict: dict = {}
    name_odds_dict: dict = {}

    # ── netkeiba オッズAPI（プライマリ）──────────────────────────
    try:
        import time as _time
        _ts = int(_time.time())  # キャッシュバスター（CDNキャッシュ回避）
        api_url = (
            f'https://race.netkeiba.com/api/api_get_jra_odds.html'
            f'?type=1&action=init&race_id={race_id}&_={_ts}'
        )
        api_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Referer": f"https://race.netkeiba.com/odds/index.html?type=b1&race_id={race_id}",
            "X-Requested-With": "XMLHttpRequest",
            "Cache-Control": "no-cache, no-store",
            "Pragma": "no-cache",
        }
        r = requests.get(api_url, headers=api_headers, timeout=5)
        api_data = _json.loads(r.text)
        if 'data' in api_data and 'odds' in api_data['data'] and '1' in api_data['data']['odds']:
            odds_raw = api_data['data']['odds']['1']
            if 'horses' in api_data.get('data', {}):
                for h in api_data['data']['horses']:
                    hname = h.get('name', '').strip()
                    hnum  = h.get('num', '')
                    if hname and hnum and str(hnum) in odds_raw:
                        name_odds_dict[hname] = float(odds_raw[str(hnum)][0])
                        odds_dict[int(hnum)]  = float(odds_raw[str(hnum)][0])
            if not odds_dict:
                for uma_num, odds_list in odds_raw.items():
                    if str(uma_num).isdigit():
                        odds_dict[int(uma_num)] = float(odds_list[0])
    except Exception as e:
        logger.warning(f'fetch_odds_realtime netkeiba失敗: {e}')

    # ── Yahoo競馬（フォールバック）──────────────────────────────
    if not odds_dict:
        try:
            r_y = requests.get(
                f"https://sports.yahoo.co.jp/keiba/race/odds/tfw/{str(race_id)[2:]}/",
                headers=get_headers(), timeout=5
            )
            soup_y = BeautifulSoup(r_y.text, 'html.parser')
            for tr in soup_y.find_all('tr'):
                tds = tr.find_all('td')
                if len(tds) >= 4:
                    u_m = re.search(r'^\s*(\d+)\s*$', tds[1].text)
                    odds_span = tr.find('span', class_='fB')
                    o_m = re.search(r'\d{1,4}\.\d+', odds_span.text) if odds_span else None
                    if u_m and o_m:
                        odds_dict[int(u_m.group(1))] = float(o_m.group(0))
        except Exception as e:
            logger.warning(f'fetch_odds_realtime Yahoo失敗: {e}')

    return odds_dict, name_odds_dict


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

    except Exception as _e:
        logger.warning(f'fetch_horse_last_race 失敗 horse_id={horse_id}: {_e}')
    return result
