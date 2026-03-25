import pandas as pd
import numpy as np
import requests
import json
from bs4 import BeautifulSoup
import re
import datetime
import logging
import traceback
from src.utils import get_headers, resolve_name, VENUE_MAWARI, VENUE_CHIKEI, TRACK_CONDITION_MAP, classify_race_class
from src.scraper import fetch_horse_last_race
from src.features_engine import classify_style, TE_COLS

logger = logging.getLogger('keiba_ebye')

def _safe_col(df, col, default=np.nan):
    """列が存在しない or スカラーになっている場合でも必ずSeriesを返す安全ラッパー"""
    if col not in df.columns:
        return pd.Series([default] * len(df), index=df.index)
    val = df[col]
    if isinstance(val, pd.Series):
        return val
    return pd.Series([val] * len(df), index=df.index)

# ==========================================
def run_real_prediction(race_id, race_date_str, bundle, skip_live_scrape=False, ev_first=False, ev_threshold=1.0, min_win_prob=0.10):
    """
    skip_live_scrape=True: バックテスト時に使用。
      fetch_horse_last_race()を呼ばない（速度維持＆日付ズレ防止）
    """
    (model, model_win, model_reg, features, cat_features, num_features, cat_categories_dict,
     latest_horse_data, horse_course_dict, ped_dict,
     known_jockeys, known_trainers, te_dicts, global_mean, recent_return_rate, ensemble_weight,
     auc_win, auc_place, *_extra) = bundle
    calibrator = _extra[0] if _extra else None
    model_d    = _extra[1] if len(_extra) > 1 else None
    
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
    except Exception as e:
        logger.warning(f'netkeiba APIオッズ取得失敗: {e}')
        error_log.append(f"netkeiba APIオッズ取得失敗: {e}")

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
        except Exception as e:
            logger.warning(f'Yahoo競馬オッズ取得失敗: {e}')
            error_log.append(f"Yahoo競馬オッズ取得失敗: {e}")

    for fetch_url in [f'https://race.netkeiba.com/race/shutuba.html?race_id={race_id}',f'https://race.netkeiba.com/race/result.html?race_id={race_id}',f'https://db.netkeiba.com/race/{race_id}/']:
        try:
            r = requests.get(fetch_url, headers=get_headers(), timeout=10); r.encoding = 'euc-jp'
            soup = BeautifulSoup(r.text, 'html.parser')
            if soup.select_one('.Shutuba_Table') or soup.select_one('.RaceTable01') or soup.select_one('.race_table_01') or soup.select_one('#All_Result_Table'):
                html_text = r.text; break
        except Exception as _e:
            logger.warning(f'出馬表取得失敗 {fetch_url}: {_e}')

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
            horse_id = re.search(r'\d+', horse_a['href']).group(0).zfill(10)
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
                except Exception as _e: logger.debug(f'name_odds_dict参照失敗: {_e}')

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
        except Exception as _e:
            logger.warning(f'出走馬パース失敗 race_id={race_id}: {_e}')

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
        _s5_infer = ['前走_スピード指数','2走前_スピード指数','3走前_スピード指数','4走前_スピード指数','5走前_スピード指数']
        df_test['近5走_中央値スピード指数'] = df_test[_s5_infer].median(axis=1)
        df_test['近5走_最高スピード指数']   = df_test[_s5_infer].max(axis=1)
        df_test['上昇度_スピード指数'] = df_test['前走_スピード指数']-df_test['近5走_中央値スピード指数']
        df_test['ベスト3走_中央値スピード指数'] = df_test[_s5_infer].apply(
            lambda r: r.dropna().nlargest(3).median() if r.dropna().shape[0] >= 1 else np.nan, axis=1)
        df_test['近5走_スピード指数安定性'] = df_test[_s5_infer].std(axis=1)

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
            aliases = {"鮫島駿":"鮫島克駿","鮫島良":"鮫島良太","吉田隼":"吉田隼人","武幸":"武幸四郎","菅原明":"菅原明良"}
            return aliases.get(n, n)

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
                    except Exception as _e:
                        logger.debug(f'前走日付パース失敗: {_e}')

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

        # 長期休養フラグ（休養日数から計算）
        df_test['長期休養フラグ'] = (df_test['休養日数'] >= 180).astype(float)

        # レース格上挑戦フラグ（前走_レースクラスコード が latest_horse_data に含まれる場合）
        if 'レースクラスコード' in df_test.columns and '前走_レースクラスコード' in df_test.columns:
            df_test['レース格上挑戦フラグ'] = (
                pd.to_numeric(df_test['レースクラスコード'], errors='coerce') >
                pd.to_numeric(df_test['前走_レースクラスコード'], errors='coerce')
            ).astype(float).fillna(0.0)
        else:
            df_test['レース格上挑戦フラグ'] = 0.0

        # コース初挑戦フラグ（horse_course_dictにキーがない = 初出走）
        if '競馬場' in df_test.columns and '芝/ダート' in df_test.columns:
            df_test['コース初挑戦フラグ'] = df_test.apply(
                lambda r: 0.0 if (r['馬ID'], r.get('競馬場',''), r.get('芝/ダート','')) in horse_course_dict
                else 1.0, axis=1)
        else:
            df_test['コース初挑戦フラグ'] = 0.0

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

        # 追加特徴量の推論時設定
        df_test['キャリア数']        = pd.to_numeric(_safe_col(df_test, 'キャリア数',        np.nan), errors='coerce')
        df_test['上り順位率']        = pd.to_numeric(_safe_col(df_test, '上り順位率',        np.nan), errors='coerce')
        df_test['前走_上り順位率']   = pd.to_numeric(_safe_col(df_test, '前走_上り順位率',   np.nan), errors='coerce')
        df_test['前走_前半ペース値'] = pd.to_numeric(_safe_col(df_test, '前走_前半ペース値', np.nan), errors='coerce')
        df_test['前走_後半ペース値'] = pd.to_numeric(_safe_col(df_test, '前走_後半ペース値', np.nan), errors='coerce')
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

        # ── 新特徴量: 馬場指数 ──────────────────────────────────────
        if '馬場' in df_test.columns:
            df_test['馬場指数'] = df_test['馬場'].map(TRACK_CONDITION_MAP).fillna(0).astype(float)
        else:
            df_test['馬場指数'] = TRACK_CONDITION_MAP.get(todays_baba, 0)

        # ── 新特徴量: レースクラスコード ───────────────────────────
        # レース情報からクラスを取得（全馬共通値）
        _race_class = classify_race_class(race_text)
        df_test['レースクラスコード'] = float(_race_class)

        # ── 新特徴量: 市場勝率（オッズの逆数） ─────────────────────
        df_test['市場勝率'] = (1.0 / df_test['単勝オッズ'].replace(0, np.nan)).clip(0, 1)

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

        # アンサンブル: 3モデルの予測を結合
        _sa = model.predict(df_test[features]).astype(float)
        _sa = (_sa - _sa.min()) / (_sa.max() - _sa.min() + 1e-9)
        try:
            _sb = model_win.predict(df_test[features]).astype(float)
            _sb = (_sb - _sb.min()) / (_sb.max() - _sb.min() + 1e-9)
            
            _sc = 1.0 - model_reg.predict(df_test[features]).astype(float)
            _sc = (_sc - _sc.min()) / (_sc.max() - _sc.min() + 1e-9)
            
            raw_scores = _sa * 0.35 + _sb * 0.50 + _sc * 0.15
        except Exception as _e:
            logger.warning(f'model_win/reg予測失敗、model_aのみ使用: {_e}')
            raw_scores = _sa  # フォールバック
        # Isotonic Calibration: AI勝率(softmax後)→実勝率補正→再正規化
        exp_scores    = np.exp(raw_scores - np.max(raw_scores))
        softmax_probs = exp_scores / np.sum(exp_scores)
        if calibrator is not None:
            try:
                calibrated = np.clip(calibrator.predict(softmax_probs), 1e-6, 1.0)
                win_probs  = calibrated / calibrated.sum()
            except Exception:
                win_probs = softmax_probs
        else:
            win_probs = softmax_probs
        df_test['勝率(AI予測)']   = win_probs
        df_test['複勝率(AI予測)'] = np.clip(win_probs*2.8, 0, 0.99)
        df_test['期待値'] = df_test['勝率(AI予測)']*df_test['単勝オッズ']
        df_test['期待値'] = df_test['期待値'].clip(upper=50.0)  # 取消馬などの異常EV防止
        df_test = df_test.sort_values('勝率(AI予測)', ascending=False).reset_index(drop=True)
        marks = ['◎','〇','▲','△','☆']+['']*(len(df_test)-5)
        df_test['印'] = marks[:len(df_test)]
        # EV優先モード: EV>=閾値 かつ AI勝率>=min_win_prob の馬を◎に昇格
        if ev_first:
            ev_cands = df_test[(df_test['期待値'] >= ev_threshold) & (df_test['勝率(AI予測)'] >= min_win_prob)]
            if not ev_cands.empty:
                # EVが最大の候補を◎に
                best_ev_idx = ev_cands['期待値'].idxmax()
                if best_ev_idx != 0:  # 元の◎と異なる場合のみ入れ替え
                    old_honmei_mark = df_test.loc[0, '印']  # '◎'
                    old_best_mark   = df_test.loc[best_ev_idx, '印']
                    df_test.loc[0,            '印'] = old_best_mark if old_best_mark else '〇'
                    df_test.loc[best_ev_idx,  '印'] = old_honmei_mark

        # ── モデルD: 穴馬スコア計算 ──────────────────────────────────
        df_test['穴馬スコア'] = 0.0
        df_test['穴馬マーク'] = ''
        if model_d is not None:
            try:
                d_features = [f for f in features if f != '市場勝率' and f in df_test.columns]
                d_proba = model_d.predict_proba(df_test[d_features])[:, 1]
                df_test['穴馬スコア'] = d_proba
                # マーク条件: レース内スコア上位2頭 AND オッズ8倍以上
                score_threshold = df_test['穴馬スコア'].quantile(0.70)
                df_test['穴馬マーク'] = df_test.apply(
                    lambda r: '🎯' if (
                        r['穴馬スコア'] >= score_threshold and
                        r['穴馬スコア'] >= 0.05 and
                        float(r.get('単勝オッズ', 0)) >= 8.0
                    ) else '',
                    axis=1
                )
            except Exception as _e:
                logger.warning(f'モデルD推論失敗（スキップ）: {_e}')

        p1,p2 = df_test.loc[0,'勝率(AI予測)'],df_test.loc[1,'勝率(AI予測)']
        score_diff = p1-p2
        top1_umaban = df_test.loc[0,'馬番']
        himo_umabans = df_test.loc[1:4,'馬番'].astype(str).tolist() if len(df_test)>=5 else df_test.loc[1:,'馬番'].astype(str).tolist()
        himo_str = "・".join(himo_umabans)
        # 未出走馬混在の判定: レーステキストのみで判断する。
        # 注意: df_test['前走_着順'].isna() はリーク防止コードで NaN 上書きされるため使用不可
        #       → skip_live_scrape=True（振り返り）時にほぼ全レースが誤判定される
        has_unraced = ('新馬' in race_text) or ('未出走' in race_text)
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
                    '近5走_中央値スピード指数': '近5走の地力(中央値)',
                    '近5走_最高スピード指数':   '過去最高スピード指数',
                    '上昇度_スピード指数':       'スピード指数の上昇度',
                    '前走_スピード指数':         '前走スピード指数',
                    '2走前_スピード指数':        '2走前スピード指数',
                    '過去3走平均スピード指数':   '近3走平均スピード指数',
                    'コース適性_着順パーセント': 'このコースの適性',
                    '前走着順パーセント':        '前走の相対着順',
                    '直近3走着順パーセント':     '直近3走の安定感',
                    '前走距離補正タイム差':      '前走のタイム差',
                    '前走上り偏差':              '前走の末脚の切れ味',
                    '前走_上り順位率':           '前走の末脚ランク',
                    '休養日数':                  '休養明けの上積み',
                    '乗り替わりフラグ':          '騎手強化の乗り替わり',
                    '位置取りショック':          '位置取りの変化',
                    '穴馬_実力馬の巻き返し':     '前走凡走からの巻き返し',
                    '穴馬_勝負の乗り替わり':     '勝負の乗り替わり',
                    '馬場指数':                  '馬場状態への適性',
                    'レースクラスコード':        'レースグレードへの対応力',
                    '前走_着順':                 '前走着順',
                    '過去3走平均着順':           '近3走平均着順',
                    '同レース逃げ馬頭数':        '展開(逃げ馬の少なさ)',
                    '距離変更フラグ':            '距離変更への対応',
                    '馬場替わりフラグ':          '馬場替わりへの対応',
                }
                # 除外特徴量: 市場勝率(循環論法)・上り順位率(現レースリーク)
                _exclude = {'市場勝率', '上り順位率', 'キャリア数'}
                feat_contrib_filtered = [(f,v) for f,v in zip(features, contribs) if f not in _exclude]
                top3 = sorted(feat_contrib_filtered, key=lambda x: x[1], reverse=True)[:3]
                reasons = [feat_label.get(f, f) for f, _ in top3]
                shap_reason = (
                    f"AIの推し理由: {best_horse_name}\n"
                    f"　「{reasons[0]}」「{reasons[1]}」「{reasons[2]}」が高評価"
                )
        except Exception as shap_e:
            shap_reason = ""  # エラー時は何も追加しない
        # SHAPはreco変数には含めず、topics_listに追加してUI表示のみに使う
        if shap_reason:
            topics_list.append(shap_reason.strip())

        return df_test, topics_list, reco, pace_text, confidence_text, track_type, place, distance, error_log

    except Exception as e:
        tb = traceback.format_exc()
        error_log.append(f"❌ 予測AI内部で致命的なエラーが発生:\n{tb}")
        # エラーログが空でも必ず何か入るよう保証
        if not error_log:
            error_log = [f"❌ 不明なエラー: {str(e)}"]
        return None,None,None,None,None,None,None,None,error_log
