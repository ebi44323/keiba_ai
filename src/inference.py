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
from src.scraper import fetch_horse_last_race, fetch_oikiri_data
from src.gemini_utils import score_oikiri_comments, check_gemini_available
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
def run_real_prediction(race_id, race_date_str, bundle, skip_live_scrape=False, ev_first=False, ev_threshold=1.5, min_win_prob=0.18, baba_override=None, use_oikiri=None):
    """
    skip_live_scrape=True: バックテスト時に使用。
      fetch_horse_last_race()を呼ばない（速度維持＆日付ズレ防止）
    baba_override: {'芝': '重', 'ダート': '良'} のように指定すると馬場を手動上書き。
    use_oikiri: 調教データ取得の制御。
      None  → skip_live_scrapeに従う（デフォルト: 通常予想=取得、長期バックテスト=スキップ）
      True  → 強制取得（当日振り返りで使用: 直近レースの調教データはまだ残っている）
      False → 強制スキップ（Optuna等の長期バックテスト）
    """
    (model, model_win, model_reg, features, cat_features, num_features, cat_categories_dict,
     latest_horse_data, horse_course_dict, ped_dict,
     known_jockeys, known_trainers, te_dicts, global_mean, recent_return_rate, ensemble_weight,
     auc_win, auc_place, *_extra) = bundle
    calibrator        = _extra[0] if _extra else None
    model_d           = _extra[1] if len(_extra) > 1 else None
    ped_aptitude_dict = _extra[2] if len(_extra) > 2 else {}
    horse_heavy_dict  = _extra[3] if len(_extra) > 3 else {}  # 馬ID → 重/不良馬場 着順パーセント平均
    sire_heavy_dict   = _extra[4] if len(_extra) > 4 else {}  # 父名  → 重/不良馬場 着順パーセント平均
    jockey_overall_dict = _extra[5] if len(_extra) > 5 else {}  # 騎手名 → 全期間平均着順パーセント
    jockey_venue_dict   = _extra[6] if len(_extra) > 6 else {}  # (騎手名,競馬場) → 平均着順パーセント
    score_norms         = _extra[7] if len(_extra) > 7 else None  # (norm_a,norm_b,norm_c) 各=(lo,hi) 絶対スコア正規化定数
    softmax_temperature = _extra[8] if len(_extra) > 8 else None  # 学習時と共有するsoftmax温度
    place_calibrator    = _extra[9] if len(_extra) > 9 else None  # 複勝率キャリブレータ（AI勝率→複勝率）
    
    error_log = []
    odds_dict = {}      # 馬番(int) → オッズ(float)
    name_odds_dict = {} # 馬名(str) → オッズ(float)  ★try外で初期化（API失敗時も参照可能）
    html_text = ""

    try:
        import time as _t
        _ts = int(_t.time())
        odds_api_url = f'https://race.netkeiba.com/api/api_get_jra_odds.html?type=1&action=init&race_id={race_id}&_={_ts}'
        api_headers = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36","Referer":f"https://race.netkeiba.com/odds/index.html?type=b1&race_id={race_id}","X-Requested-With":"XMLHttpRequest","Cache-Control":"no-cache, no-store","Pragma":"no-cache"}
        r_api = requests.get(odds_api_url, headers=api_headers, timeout=5)
        api_data = json.loads(r_api.text)
        if 'data' in api_data and 'odds' in api_data['data'] and '1' in api_data['data']['odds']:
            odds_raw = api_data['data']['odds']['1']
            if 'horses' in api_data.get('data', {}):
                # horses リストがある → 馬名と馬番の両方を確実にマッピング
                for h in api_data['data']['horses']:
                    hname = h.get('name', '').strip()
                    hnum  = h.get('num', '')
                    if hname and hnum and str(hnum) in odds_raw:
                        name_odds_dict[hname] = float(odds_raw[str(hnum)][0])
                        odds_dict[int(hnum)]  = float(odds_raw[str(hnum)][0])
            # horses なし or 一部未取得の場合は odds_raw のキー（馬番）で補完
            # ★ odds_raw のキーは常に馬番。人気順ではない。
            for uma_num, odds_list in odds_raw.items():
                if str(uma_num).isdigit() and int(uma_num) not in odds_dict:
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
            r = requests.get(fetch_url, headers=get_headers(), timeout=10)
            try:
                _html = r.content.decode('utf-8')
            except UnicodeDecodeError:
                _html = r.content.decode('euc-jp', errors='replace')
            soup = BeautifulSoup(_html, 'html.parser')
            if soup.select_one('.Shutuba_Table') or soup.select_one('.RaceTable01') or soup.select_one('.race_table_01') or soup.select_one('#All_Result_Table'):
                html_text = _html; break
        except Exception as _e:
            logger.warning(f'出馬表取得失敗 {fetch_url}: {_e}')

    if not html_text: return None,None,None,None,None,None,None,None,["❌ 出馬表が取得できませんでした。"]
    soup = BeautifulSoup(html_text, 'html.parser')
    race_data_box = soup.find('div', class_='RaceData01') or soup.find('dl', class_='racedata')
    if not race_data_box: return None,None,None,None,None,None,None,None,["❌ レース条件が見つかりません。"]

    race_text = race_data_box.text.replace('\n','')
    # レース名（クラス判定用）: RaceName > RaceData02 > pageTitle の順で補完
    _race_name_tag = (soup.find(class_='RaceName') or soup.find(class_='race_name')
                      or soup.find('h1', class_=re.compile(r'Race', re.I)))
    _race_name_text = _race_name_tag.get_text(' ', strip=True) if _race_name_tag else ''
    _race_data02 = soup.find('div', class_='RaceData02')
    _race_data02_text = _race_data02.get_text(' ', strip=True) if _race_data02 else ''
    race_class_text = _race_name_text + ' ' + _race_data02_text + ' ' + race_text
    baba_match = re.search(r'馬場:([良稍重不良]+)', race_text)
    todays_baba = baba_match.group(1) if baba_match else '良'
    tdm = re.search(r'(芝|ダ|障|障害).*?(\d+)m', race_text)
    track_type = "芝" if tdm and tdm.group(1)=="芝" else "ダート" if tdm and "ダ" in tdm.group(1) else "障害"
    # ── 馬場手動上書き（サイドバーからの設定が優先）──────────────────
    # place を先に導出（競馬場別override判定に使用）
    place = {'01':'札幌','02':'函館','03':'福島','04':'新潟','05':'東京','06':'中山','07':'中京','08':'京都','09':'阪神','10':'小倉'}.get(str(race_id)[4:6], '東京')
    if baba_override:
        # 形式A: 競馬場別 {'東京': {'芝': '不良', 'ダート': '重'}, '阪神': {'芝': '重'}}
        _first_val = next(iter(baba_override.values()), None)
        if isinstance(_first_val, dict):
            _venue_baba = baba_override.get(place, {})
            if track_type in _venue_baba and _venue_baba[track_type]:
                todays_baba = _venue_baba[track_type]
                logger.info(f'馬場手動上書き（{place}）: {track_type} → {todays_baba}')
        # 形式B: 全会場共通 {'芝': '不良', 'ダート': '重'}
        elif track_type in baba_override and baba_override[track_type]:
            todays_baba = baba_override[track_type]
            logger.info(f'馬場手動上書き（全会場）: {track_type} → {todays_baba}')
    distance = float(tdm.group(2)) if tdm else 1600.0
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

            # ── オッズ取得（優先順: 1.馬名 → 2.馬番 → 3.HTMLテーブル → 4.デフォルト10倍）──
            # ★ APIキーは枠番確定前後ともに「馬番」。row_posは使わない。
            # 1. 馬名マッチ（最確実。枠番確定前でも馬名は変わらない）
            odds_val = name_odds_dict.get(horse_name, 0.0)

            # 2. 馬番マッチ
            if odds_val == 0.0:
                odds_val = odds_dict.get(umaban, 0.0)

            # 3. ページ内オッズ列
            if odds_val == 0.0 and odds_idx != -1 and len(tds) > odds_idx:
                om = re.search(r'\d{1,4}\.\d+', tds[odds_idx].text)
                if om: odds_val = float(om.group(0))

            # 4. クラス属性
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

        # 案3: スピード指数NaNフォールバック（海外帰り等でNaNの場合、直近の有効値で補完）
        # 前走がNaN → 2走前、2走前がNaN → 3走前... の順でバックフィル
        for _i in range(len(_s5_infer) - 1):
            _nan_mask = df_test[_s5_infer[_i]].isna() & df_test[_s5_infer[_i + 1]].notna()
            if _nan_mask.any():
                df_test.loc[_nan_mask, _s5_infer[_i]] = df_test.loc[_nan_mask, _s5_infer[_i + 1]]
        # フォールバック後に派生特徴量を再計算
        df_test['過去3走平均スピード指数']     = df_test[_s5_infer[:3]].mean(axis=1)
        df_test['近5走_中央値スピード指数']    = df_test[_s5_infer].median(axis=1)
        df_test['近5走_最高スピード指数']      = df_test[_s5_infer].max(axis=1)
        df_test['上昇度_スピード指数']         = df_test['前走_スピード指数'] - df_test['近5走_中央値スピード指数']
        df_test['ベスト3走_中央値スピード指数'] = df_test[_s5_infer].apply(
            lambda r: r.dropna().nlargest(3).median() if r.dropna().shape[0] >= 1 else np.nan, axis=1)
        df_test['近5走_スピード指数安定性']    = df_test[_s5_infer].std(axis=1)

        df_test['前走_通過'] = _safe_col(df_test, '最新_通過', '')
        def parse_corner(x):
            s=str(x); return s.split('-')[-1] if '-' in s else (s if s.isdigit() else np.nan)
        df_test['前走_最終コーナー'] = pd.to_numeric(df_test['前走_通過'].fillna('').astype(str).apply(parse_corner), errors='coerce')
        df_test['脚質カテゴリ'] = df_test['前走_最終コーナー'].apply(classify_style)
        df_test['前走逃げフラグ']  = (df_test['前走_最終コーナー']<=2).astype(int)
        df_test['前走先行フラグ']  = ((df_test['前走_最終コーナー']>2)&(df_test['前走_最終コーナー']<=5)).astype(int)
        df_test['同レース逃げ馬頭数'] = df_test['前走逃げフラグ'].sum()
        df_test['同レース先行馬頭数'] = df_test['前走先行フラグ'].sum()

        # ── 展開×脚質 交互作用特徴量（features_engine.py と同一ロジック）────
        df_test['逃げ_単独優位スコア'] = (
            df_test['前走逃げフラグ'].fillna(0) * np.maximum(0.0, 3.0 - df_test['同レース逃げ馬頭数'])
        )
        _is_oikomi = ((1 - df_test['前走逃げフラグ'].fillna(0)) * (1 - df_test['前走先行フラグ'].fillna(0)))
        df_test['追込_展開向き度'] = _is_oikomi * df_test['同レース逃げ馬頭数']
        _pace_inf = pd.to_numeric(_safe_col(df_test, '前走_前半ペース値', np.nan), errors='coerce')
        if _pace_inf.notna().sum() > 0:
            _pm_inf = float(_pace_inf.mean()) if _pace_inf.notna().sum() > 0 else 0.0
            _ps_inf = float(_pace_inf.std()) if _pace_inf.notna().sum() > 1 and _pace_inf.std() > 1e-3 else 1.0
            _pace_z_inf = (_pace_inf - _pm_inf) / _ps_inf
            df_test['前走_ペース補正スピード指数'] = (
                pd.to_numeric(_safe_col(df_test, '前走_スピード指数', 50.0), errors='coerce').fillna(50.0)
                - df_test['前走逃げフラグ'].fillna(0) * _pace_z_inf.fillna(0) * 2.0
            )
        else:
            df_test['前走_ペース補正スピード指数'] = pd.to_numeric(
                _safe_col(df_test, '前走_スピード指数', 50.0), errors='coerce').fillna(50.0)

        df_test['コース適性_着順パーセント'] = df_test.set_index(['馬ID','競馬場','芝/ダート']).index.map(horse_course_dict).fillna(0.5)

        # 騎手能力特徴量（学習データの全期間平均 → リアルタイム推論）
        df_test['騎手_通算着順パーセント'] = df_test['騎手'].map(jockey_overall_dict).fillna(0.5)
        _jv_keys = list(zip(df_test['騎手'], df_test['競馬場']))
        df_test['騎手_競馬場_着順パーセント'] = [
            jockey_venue_dict.get(k, jockey_overall_dict.get(k[0], 0.5))
            for k in _jv_keys
        ]

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
                         '前走_最終コーナー','2走前_最終コーナー','最新_斤量']
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
        df_test['海外帰りフラグ'] = 0.0  # 案1: 海外帰りフラグ初期化
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

                # 案1: 海外帰りフラグをセット
                if prev.get('海外帰りフラグ'):
                    df_test.at[idx, '海外帰りフラグ'] = 1.0
                    logger.info(f'海外帰り馬検出: 馬ID={hid}')

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
        # 案1: 海外帰り馬は長期休養扱いにしない（休養ではなく海外遠征のため）
        overseas_mask = df_test.get('海外帰りフラグ', pd.Series(0, index=df_test.index)) == 1.0
        if overseas_mask.any():
            df_test.loc[overseas_mask, '長期休養フラグ'] = 0.0
            logger.info(f'海外帰り馬の長期休養フラグをリセット: {overseas_mask.sum()}頭')

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
            # 文字列'nan'や空文字をfloat NaNに統一（馬場替わりフラグと同様の対策）
            _jockey_clean = df_test['最新_騎手'].replace({'nan': np.nan, '': np.nan})
            now_j  = df_test['騎手'].apply(_norm_name)
            prev_j = _jockey_clean.apply(_norm_name)
            df_test['乗り替わりフラグ'] = ((now_j != prev_j) & (prev_j != '')).astype(int)
            df_test['_前走騎手']        = _jockey_clean.fillna('不明')
        else:
            df_test['乗り替わりフラグ'] = 0
            df_test['_前走騎手']        = '不明'

        # 馬場替わりフラグ
        # 障害レースは「障害」カテゴリとして扱い、芝/ダート→障害は常に「変更」とする
        # ただし障害→障害は変化なし
        if '最新_芝ダート' in df_test.columns:
            # 文字列'nan'や空文字をfloat NaNに統一（CSVやpkl由来の'nan'文字列を防ぐ）
            _surf_clean = df_test['最新_芝ダート'].replace({'nan': np.nan, '': np.nan})
            now_s  = df_test['芝/ダート'].fillna('').astype(str).str.strip()
            prev_s = _surf_clean.fillna('').astype(str).str.strip()
            # 障害同士は変化なし扱い（今回も前走も障害なら変化なし）
            both_shogai = (now_s.str.contains('障') & prev_s.str.contains('障'))
            surf_changed = ((now_s != prev_s) & (prev_s != '') & ~both_shogai)
            df_test['馬場替わりフラグ'] = surf_changed.astype(int)
            df_test['_前走馬場']        = _surf_clean.fillna('不明')
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
        # ── 新馬フラグ（初出走: キャリア数が0またはNaN かつ 前走着順も不明）──────
        # 注意: CSVに載っていない馬はキャリア数がNaNになるが、
        #       fetch_horse_last_race()で最新_着順が取れていれば走経験あり → フラグ立てない
        _no_career  = df_test['キャリア数'].isna() | (df_test['キャリア数'] == 0)
        _no_last_race = pd.to_numeric(_safe_col(df_test, '最新_着順', np.nan), errors='coerce').isna()
        df_test['新馬フラグ'] = (_no_career & _no_last_race).astype(float)
        df_test['上り順位率']        = pd.to_numeric(_safe_col(df_test, '上り順位率',        np.nan), errors='coerce')
        df_test['前走_上り順位率']   = pd.to_numeric(_safe_col(df_test, '前走_上り順位率',   np.nan), errors='coerce')
        df_test['前走_前半ペース値'] = pd.to_numeric(_safe_col(df_test, '前走_前半ペース値', np.nan), errors='coerce')
        df_test['前走_後半ペース値'] = pd.to_numeric(_safe_col(df_test, '前走_後半ペース値', np.nan), errors='coerce')
        # 馬体重フォールバック: 発表前（NaN）は前回体重で補完し朝予想を安定化
        _prev_weight = pd.to_numeric(_safe_col(df_test, '最新_馬体重', np.nan), errors='coerce')
        _weight_nan  = df_test['馬体重_num'].isna()
        if _weight_nan.any():
            df_test.loc[_weight_nan, '馬体重_num'] = _prev_weight[_weight_nan]
            logger.debug(f'馬体重フォールバック: {_weight_nan.sum()}頭を前回体重で補完')
        df_test['馬体重増減']            = df_test['馬体重_num'] - _prev_weight
        df_test['斤量差'] = pd.to_numeric(df_test['斤量'],errors='coerce') - pd.to_numeric(df_test['斤量'],errors='coerce').mean()
        _prev_kinryo = pd.to_numeric(_safe_col(df_test, '最新_斤量', np.nan), errors='coerce')
        df_test['斤量_前走差'] = pd.to_numeric(df_test['斤量'], errors='coerce') - _prev_kinryo
        df_test['穴馬_距離変更一変']     = ((df_test['距離変更フラグ']==1)&(df_test['直近3走着順パーセント']<0.4)).astype(int)
        df_test['穴馬_馬場替わり一変']   = ((df_test['馬場替わりフラグ']==1)&(df_test['直近3走着順パーセント']<0.4)).astype(int)
        df_test['穴馬_勝負の乗り替わり'] = ((df_test['乗り替わりフラグ']==1)&(df_test['直近3走着順パーセント']<0.5)).astype(int)
        df_test['穴馬_実力馬の巻き返し'] = ((df_test['前走大敗フラグ']==1)&(df_test['近5走_最高スピード指数']>=55)).astype(int)
        df_test['回り']       = df_test['競馬場'].map(VENUE_MAWARI).fillna('不明')
        df_test['コース地形'] = df_test['競馬場'].map(VENUE_CHIKEI).fillna('不明')

        # ── 新特徴量: 馬場指数 ──────────────────────────────────────
        if '馬場' in df_test.columns:
            df_test['馬場指数'] = df_test['馬場'].map(TRACK_CONDITION_MAP).fillna(0).astype(float)
        else:
            df_test['馬場指数'] = TRACK_CONDITION_MAP.get(todays_baba, 0)

        # ── 新特徴量: 馬・父の重馬場適性 ────────────────────────────
        # 父の重馬場適性（種牡馬産駒の重・不良馬場での平均着順パーセント）
        df_test['父_重馬場_着順パーセント'] = (
            df_test['父'].map(sire_heavy_dict)
            .fillna(0.5)
            .astype(float)
        )
        # 馬の重馬場適性（当該馬の実績 → なければ父の統計でフォールバック）
        df_test['馬_重馬場_着順パーセント'] = (
            df_test['馬ID'].map(horse_heavy_dict)
            .fillna(df_test['父_重馬場_着順パーセント'])
            .astype(float)
        )

        # ── 調教データ取得（ポストモデル補正用・モデル特徴量ではない）──────
        # use_oikiri=None → skip_live_scrapeに従う
        # use_oikiri=True → 当日振り返りなど直近レース用に強制取得
        # use_oikiri=False → Optuna等の長期バックテストで強制スキップ
        _should_fetch_oikiri = (not skip_live_scrape) if use_oikiri is None else use_oikiri
        _oikiri_data = {}
        if _should_fetch_oikiri:
            try:
                _oikiri_data = fetch_oikiri_data(str(race_id))
            except Exception as _oe:
                logger.warning(f'fetch_oikiri_data 失敗（スキップ）: {_oe}')

        # ── 新特徴量: レースクラスコード ───────────────────────────
        # RaceName + RaceData02 + RaceData01 を結合してクラス判定（RaceData01だけだとクラス情報がない）
        _race_class = classify_race_class(race_class_text)
        df_test['レースクラスコード'] = float(_race_class)

        # ── 新特徴量: 市場勝率（オッズの逆数） ─────────────────────
        df_test['市場勝率'] = (1.0 / df_test['単勝オッズ'].replace(0, np.nan)).clip(0, 1)

        # ── 血統距離適性スコア（ped_aptitude_dict lookup）────────────────
        def _dist_bucket_fn(d):
            try:
                d = float(d)
                if d < 1400: return 'sprint'
                elif d < 1800: return 'mile'
                elif d < 2200: return 'intermediate'
                else: return 'long'
            except: return 'unknown'
        _dist_bkt = _dist_bucket_fn(distance)

        def _ped_aptitude(row):
            sire = str(row.get('父', '不明') or '不明')
            if sire in ('不明', 'nan', ''):
                return 0.5
            track = str(row.get('芝/ダート', '芝') or '芝')
            v = ped_aptitude_dict.get((sire, _dist_bkt, track))
            if v is None:
                v = ped_aptitude_dict.get((sire,))
            return v if v is not None else 0.5

        df_test['血統距離適性スコア'] = df_test.apply(_ped_aptitude, axis=1).astype(float)

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
        # ── 絶対スコア正規化（2026-07-25）─────────────────────────────
        # bundleにscore_norms(学習データ基準の1〜99%tile)があれば、それを固定基準に normalize。
        # レース間で「馬の絶対的な強さ」が比較可能になり、小頭数での過大評価が解消される。
        # 旧bundle(score_norms=None)の場合は従来のレース内min-maxにフォールバック（後方互換）。
        def _norm_abs(s, lohi):
            if lohi is None:
                return (s - s.min()) / (s.max() - s.min() + 1e-9)  # フォールバック: レース内min-max
            lo, hi = lohi
            return np.clip((s - lo) / (hi - lo + 1e-9), 0.0, 1.0)

        _na = score_norms[0] if score_norms else None
        _nb = score_norms[1] if score_norms else None
        _nc = score_norms[2] if score_norms else None

        _sa = _norm_abs(model.predict(df_test[features]).astype(float), _na)
        try:
            _sb = _norm_abs(model_win.predict(df_test[features]).astype(float), _nb)
            _sc = _norm_abs(1.0 - model_reg.predict(df_test[features]).astype(float), _nc)
            raw_scores = _sa * 0.0581 + _sb * 0.8159 + _sc * 0.1261  # アンサンブル重み最適化 @ 2026-03-30
        except Exception as _e:
            logger.warning(f'model_win/reg予測失敗、model_aのみ使用: {_e}')
            raw_scores = _sa  # フォールバック

        # ── ポストモデル調整: 調教グレード＋コメントスコア ─────────────
        # モデルは歴史データのみ学習 → 調教情報は「当日加点」として補正
        # バックテスト(skip_live_scrape=True)では _oikiri_data={} のため補正ゼロ
        if _oikiri_data:
            # Phase1: グレード補正（S=+0.07, A=+0.07, B=0, C=-0.07, D=-0.10）
            # 補正値: 2025/01〜2026/04の4,300レース実績から算出 (SCALE=0.5)
            # A: 1着率25% vs B:9.8% (着順%差-0.146), C: 1着率3.4% (着順%差+0.141)
            # Sはデータなし → Aと同値で設定、DはCより強めのペナルティ
            _GRADE_BOOST = {'S': 0.07, 'A': 0.07, 'B': 0.0, 'C': -0.07, 'D': -0.10}
            _grade_map = {uban: _GRADE_BOOST.get(v.get('評価', 'B'), 0.0)
                          for uban, v in _oikiri_data.items()}
            _grade_arr = df_test['馬番'].astype(int).map(_grade_map).fillna(0.0).values
            raw_scores = raw_scores + _grade_arr

            # Phase2: Geminiコメントスコア補正（コメントあり時のみ）
            if check_gemini_available() and any(v.get('コメント', '') for v in _oikiri_data.values()):
                try:
                    _comment_scores = score_oikiri_comments(_oikiri_data)
                    if _comment_scores:
                        _comment_arr = (df_test['馬番'].astype(int)
                                        .map(_comment_scores).fillna(0.0).values * 0.02)
                        raw_scores = raw_scores + _comment_arr
                        logger.info(f'調教コメント補正適用: {len(_comment_scores)}頭')
                except Exception as _ce:
                    logger.warning(f'調教コメント補正スキップ: {_ce}')

        # 調教評価列をdf_testに追加（表示・確認用）
        df_test['調教評価'] = (df_test['馬番'].astype(int)
                               .map({uban: v.get('評価', '') for uban, v in _oikiri_data.items()})
                               .fillna(''))

        # Temperature Scaling + Isotonic Calibration
        # 学習時と共有する温度をbundleから取得（絶対スコア化に伴いcalibrator主導へ）。
        # 旧bundle(softmax_temperature=None)では従来値1.5でレース内min-maxと整合させる。
        TEMPERATURE = float(softmax_temperature) if softmax_temperature else 1.5
        exp_scores    = np.exp((raw_scores - np.max(raw_scores)) / TEMPERATURE)
        softmax_probs = exp_scores / np.sum(exp_scores)
        if calibrator is not None:
            try:
                calibrated = np.clip(calibrator.predict(softmax_probs), 1e-6, 1.0)
                # IsotonicRegressionはステップ関数のため、同じステップに落ちた馬が
                # 完全同値になる（例: 5頭が全員5.0%）。
                # 同値グループ内はraw softmax確率の比率で按分してタイブレークする。
                win_probs_adj = calibrated.copy().astype(float)
                for v in np.unique(calibrated):
                    mask = calibrated == v
                    if mask.sum() > 1:
                        group_sp = softmax_probs[mask]
                        group_sp_sum = group_sp.sum()
                        if group_sp_sum > 1e-9:
                            win_probs_adj[mask] = v * (group_sp / group_sp_sum)
                win_probs = win_probs_adj / win_probs_adj.sum()
            except Exception:
                win_probs = softmax_probs
        else:
            win_probs = softmax_probs
        df_test['勝率(AI予測)']   = win_probs
        # 複勝率(3着内): place_calibrator があれば学習済みIsotonicで算出（#5・データドリブン）。
        # 無い旧bundleは従来のBradley-Terry式 3p/(2p+1) にフォールバック。
        if place_calibrator is not None:
            try:
                df_test['複勝率(AI予測)'] = np.clip(place_calibrator.predict(win_probs), 0.0, 0.98)
            except Exception:
                df_test['複勝率(AI予測)'] = np.clip((3.0 * win_probs) / (2.0 * win_probs + 1.0 + 1e-9), 0, 0.95)
        else:
            df_test['複勝率(AI予測)'] = np.clip((3.0 * win_probs) / (2.0 * win_probs + 1.0 + 1e-9), 0, 0.95)
        df_test['期待値'] = df_test['勝率(AI予測)']*df_test['単勝オッズ']
        df_test['期待値'] = df_test['期待値'].clip(upper=50.0)  # 取消馬などの異常EV防止
        # 未出走馬の強制除外を撤廃（2026-07-25）:
        # 旧: 新馬フラグ==1の馬を必ず上位5頭の外へソートしていた。
        # 新: 純粋に勝率(AI予測)順でソートし、モデル評価(新馬フラグ/血統距離適性スコア)に委ねる。
        #     未出走混在時は confidence_text に注意書きを付記するのみ（後述）。
        df_test = df_test.sort_values('勝率(AI予測)', ascending=False).reset_index(drop=True)
        marks = ['◎','〇','▲','△','☆']+['']*(len(df_test)-5)
        df_test['印'] = marks[:len(df_test)]

        # ── モデルD: 穴馬スコア計算（EV優先判定より先に行う）──────────────────────
        df_test['穴馬スコア'] = 0.0
        df_test['穴馬マーク'] = ''
        if model_d is not None:
            try:
                d_features = [f for f in features if f != '市場勝率' and f in df_test.columns]
                d_proba = model_d.predict_proba(df_test[d_features])[:, 1]
                df_test['穴馬スコア'] = d_proba
                # マーク条件: レース内スコア上位30% AND スコア>=0.05 AND オッズ8倍以上
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

        # EV優先モード: EV>=閾値 かつ AI勝率>=勝率フロア の馬を◎に昇格
        # 穴馬スコアが高い馬は複合EVスコア(EV × (1 + 穴馬スコア×0.5))で優先される
        if ev_first:
            # EV昇格の勝率フロアを頭数連動で厳格化（2026-07-25・原因B対策）
            # 小頭数レースはsoftmaxが弱い馬にも高勝率を配分するため、高オッズだけで◎昇格しやすかった。
            # 基本フロア0.25、かつ 1/N の1.4倍を下限に → 5頭:0.28 / 8頭:0.25 / 18頭:0.25。
            # 新馬フラグ==0 条件は撤廃（#3と整合）し、勝率フロアで弱い未出走馬を自然に排除する。
            _n_runners = max(len(df_test), 1)
            ev_win_floor = max(0.25, 1.4 / _n_runners, float(min_win_prob))
            ev_cands = df_test[
                (df_test['期待値'] >= ev_threshold) &
                (df_test['勝率(AI予測)'] >= ev_win_floor)
            ]
            if not ev_cands.empty:
                ev_cands = ev_cands.copy()
                ev_cands['_ev_composite'] = ev_cands['期待値'] * (1.0 + ev_cands['穴馬スコア'] * 0.5)
                best_ev_idx = ev_cands['_ev_composite'].idxmax()
                if best_ev_idx != 0:  # 元の◎と異なる場合のみ入れ替え
                    old_ev_mark = df_test.loc[best_ev_idx, '印']
                    # 印だけでなく行ごと入れ替え（以降の処理がloc[0]基準のため）
                    idx_list = [best_ev_idx] + [i for i in df_test.index if i != best_ev_idx]
                    df_test = df_test.loc[idx_list].reset_index(drop=True)
                    df_test.loc[0, '印'] = '◎'
                    df_test.loc[1, '印'] = old_ev_mark if old_ev_mark else '〇'

        p1,p2 = df_test.loc[0,'勝率(AI予測)'],df_test.loc[1,'勝率(AI予測)']
        score_diff = p1-p2
        top1_umaban = df_test.loc[0,'馬番']
        himo_umabans = df_test.loc[1:4,'馬番'].astype(str).tolist() if len(df_test)>=5 else df_test.loc[1:,'馬番'].astype(str).tolist()
        himo_str = "・".join(himo_umabans)
        # 未出走馬混在の判定: レーステキスト + 新馬フラグで判断
        # 注意: df_test['前走_着順'].isna() はリーク防止コードで NaN 上書きされるため使用不可
        #       → skip_live_scrape=True（振り返り）時にほぼ全レースが誤判定される
        has_unraced = (
            ('新馬' in race_text) or ('未出走' in race_text) or
            (df_test['新馬フラグ'].sum() > 0)
        )
        ana_horse_nums = []; topics_list = []
        for rank, row in df_test.iterrows():
            if rank>=4 and row['期待値']>=1.5:
                topics_list.append(f"📌 {row['馬名']} (期待値特大の穴馬！)")
                if f"{row['馬番']}番" not in ana_horse_nums: ana_horse_nums.append(f"{row['馬番']}番")
        ana_str = "・".join(str(n) for n in ana_horse_nums[:3]) if ana_horse_nums else ""

        # ── 勝負/回避レース判定（#5・EV考慮に拡張 2026-07-25）──────────────
        # 🔥勝負: (A) 本命が抜けている＝高勝率×勝率差  __または__
        #         (B) 本命の期待値が大きい＝妙味大（勝率が高くなくてもオッズ妙味で勝負）
        # ⚠️回避: 未出走混在 or 決め手のない低確率混戦（EVも小さい）
        # 期待値 = 勝率×オッズ。EV優先で◎が入替わった後の◎の期待値を見る。
        top_ev = float(df_test.loc[0, '期待値'])
        EV_KACHI = 2.0   # ◎の期待値がこの値以上なら勝率が高くなくても🔥勝負扱い（調整可）
        if (p1 >= 0.25 and score_diff >= 0.10) or (top_ev >= EV_KACHI):
            race_grade = "🔥 勝負レース"
        elif has_unraced or (score_diff <= 0.03 and p1 < 0.20):
            race_grade = "⚠️ 回避（様子見）レース"
        else:
            race_grade = "🟡 通常レース"

        if p1>=0.25 and score_diff>=0.10:
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

        # ── モデルD穴馬を複勝推奨に活用（#6・2026-07-25）──────────────────
        # 穴馬マーク🎯（モデルD高スコア×オッズ8倍+）が付いた馬を複勝の妙味候補として明示。
        if '穴馬マーク' in df_test.columns:
            _dmark = df_test[df_test['穴馬マーク'] == '🎯']
            if not _dmark.empty:
                _d_nums = "・".join(str(n) for n in _dmark['馬番'].tolist()[:3])
                reco += f"\n  🎯 モデルD妙味穴: {_d_nums}番 の複勝・ワイドに一考の価値。"

        # 未出走混在時: 強制見送りはせず注意書きを付記（#3・強制除外撤廃と整合）
        if has_unraced:
            confidence_text += "\n⚠️ 未出走馬が含まれます。過去データが無いため該当馬の評価は不確実です（印はモデル評価に基づきます）。"

        # 勝負/回避ラベルを先頭に付与（#5）
        confidence_text = f"{race_grade}\n{confidence_text}"

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
