"""
collect_oikiri_history.py
=========================
netkeiba の調教ページから過去レースの調教評価(S/A/B/C)を収集し、
実際のレース結果との相関を分析して最適な補正値を算出するスクリプト。

使い方:
  python collect_oikiri_history.py           # スクレイピング + 分析
  python collect_oikiri_history.py --analyze # 既存 oikiri_history.csv の分析のみ

進捗は oikiri_history.csv に逐次保存。中断しても再開可能。
"""

import argparse
import time
import re
import zipfile
import json
import os
import sys
import logging

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup

from src.config import get_headers

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

SAVE_PATH    = 'oikiri_history.csv'
PROGRESS_PATH = 'oikiri_history_progress.json'
SLEEP_SEC    = 1.5      # リクエスト間隔（サーバー負荷軽減）
SAVE_EVERY   = 50       # N レースごとに保存
GRADE_MAP    = {'S': 4.0, 'A': 3.0, 'B': 2.0, 'C': 1.0}

# ─── スクレイピング本体（Streamlit 非依存版） ───────────────────────────────

def _scrape_oikiri(race_id: str) -> list[dict]:
    """
    指定レースの調教評価を取得。
    Returns: [{'race_id': ..., '馬番': ..., '評価': 'A', '評価スコア': 3.0}, ...]
    取得失敗 / 未公開は []
    """
    rows_out = []
    try:
        url = f"https://race.netkeiba.com/race/oikiri.html?race_id={race_id}"
        headers = get_headers()
        headers['Referer'] = f'https://race.netkeiba.com/race/shutuba.html?race_id={race_id}'
        r = requests.get(url, headers=headers, timeout=10)
        try:
            content = r.content.decode('utf-8')
        except UnicodeDecodeError:
            content = r.content.decode('euc-jp', errors='replace')

        soup = BeautifulSoup(content, 'html.parser')

        # 評価セルが最も多いテーブルを選択
        best_table, best_score = None, 0
        for tbl in soup.find_all('table'):
            cnt = sum(1 for td in tbl.find_all(['td', 'th'])
                      if td.get_text(strip=True) in GRADE_MAP)
            if cnt > best_score:
                best_score, best_table = cnt, tbl

        if best_table is None or best_score < 2:
            return rows_out

        # ヘッダーで馬番列を確定
        uban_col = None
        header_row = best_table.find('tr')
        if header_row:
            for i, th in enumerate(header_row.find_all(['th', 'td'])):
                if '馬番' in th.get_text(strip=True):
                    uban_col = i
                    break

        for row in best_table.find_all('tr'):
            tds = row.find_all('td')
            if len(tds) < 2:
                continue

            grade_letter = None
            for td in tds:
                txt = td.get_text(strip=True)
                if txt in GRADE_MAP:
                    grade_letter = txt
                    break
            if grade_letter is None:
                continue

            uban = None
            if uban_col is not None and uban_col < len(tds):
                t = tds[uban_col].get_text(strip=True)
                if re.match(r'^\d{1,2}$', t) and 1 <= int(t) <= 18:
                    uban = int(t)
            if uban is None:
                nums = [int(td.get_text(strip=True)) for td in tds[:3]
                        if re.match(r'^\d{1,2}$', td.get_text(strip=True))
                        and 1 <= int(td.get_text(strip=True)) <= 18]
                uban = nums[1] if len(nums) >= 2 else (nums[0] if nums else None)
            if uban is None:
                continue

            rows_out.append({
                'race_id': race_id,
                '馬番': uban,
                '評価': grade_letter,
                '評価スコア': GRADE_MAP[grade_letter],
            })

    except Exception as e:
        logger.warning(f'スクレイピング失敗 {race_id}: {e}')

    return rows_out


# ─── レースID収集 ──────────────────────────────────────────────────────────

def load_race_ids(since: str = '2025-01-01') -> list[str]:
    """学習データから 2025年以降の JRA レースID を取得"""
    logger.info('学習データ読み込み中...')
    try:
        with zipfile.ZipFile('learning_data_perfect_tier.zip') as z:
            with z.open(z.namelist()[0]) as f:
                df = pd.read_csv(f, usecols=['日付', 'レースID'], encoding='utf-8')
    except Exception as e:
        logger.error(f'学習データ読み込み失敗: {e}')
        sys.exit(1)

    df['日付'] = pd.to_datetime(df['日付'], errors='coerce')
    df = df[df['日付'] >= since].dropna(subset=['レースID'])
    df['レースID'] = df['レースID'].astype(str)

    # 地方競馬除外（place code = 文字3-4桁目, 01-10 = JRA）
    def is_jra(rid):
        if len(rid) == 12:
            try:
                return 1 <= int(rid[4:6]) <= 10
            except:
                pass
        return False

    race_ids = sorted(df[df['レースID'].apply(is_jra)]['レースID'].unique())
    logger.info(f'対象レース数: {len(race_ids)} ({since} 以降・JRAのみ)')
    return race_ids


# ─── スクレイピング実行 ────────────────────────────────────────────────────

def run_scraping(race_ids: list[str]):
    # 既存の進捗を読み込む
    done_ids: set[str] = set()
    existing_rows: list[dict] = []

    if os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH, 'r', encoding='utf-8') as f:
            done_ids = set(json.load(f).get('done', []))
        logger.info(f'既存進捗: {len(done_ids)} レース完了済み')

    if os.path.exists(SAVE_PATH):
        df_ex = pd.read_csv(SAVE_PATH, dtype={'race_id': str})
        existing_rows = df_ex.to_dict('records')

    todo = [rid for rid in race_ids if rid not in done_ids]
    logger.info(f'残り: {len(todo)} レース')

    if not todo:
        logger.info('全レース取得済み。スキップします。')
        return

    new_rows: list[dict] = []
    for i, race_id in enumerate(todo):
        rows = _scrape_oikiri(race_id)
        new_rows.extend(rows)
        done_ids.add(race_id)

        # 進捗表示
        total_done = len(done_ids)
        total = len(race_ids)
        if (i + 1) % 10 == 0 or i == 0:
            pct = total_done / total * 100
            found = len(new_rows)
            logger.info(f'[{total_done}/{total} ({pct:.1f}%)] 取得済み {found}件 (今回) | {race_id} → {len(rows)}頭')

        # 定期保存
        if (i + 1) % SAVE_EVERY == 0:
            _save(existing_rows + new_rows, done_ids)
            logger.info(f'中間保存: {SAVE_PATH}')

        time.sleep(SLEEP_SEC)

    _save(existing_rows + new_rows, done_ids)
    logger.info(f'完了。合計 {len(existing_rows) + len(new_rows)} 件保存: {SAVE_PATH}')


def _save(rows: list[dict], done_ids: set[str]):
    if rows:
        pd.DataFrame(rows).to_csv(SAVE_PATH, index=False, encoding='utf-8-sig')
    with open(PROGRESS_PATH, 'w', encoding='utf-8') as f:
        json.dump({'done': list(done_ids)}, f)


# ─── 分析 ─────────────────────────────────────────────────────────────────

def run_analysis():
    if not os.path.exists(SAVE_PATH):
        logger.error(f'{SAVE_PATH} が見つかりません。先にスクレイピングを実行してください。')
        return

    logger.info('分析開始...')
    df_oikiri = pd.read_csv(SAVE_PATH, dtype={'race_id': str})
    df_oikiri['馬番'] = pd.to_numeric(df_oikiri['馬番'], errors='coerce')
    logger.info(f'調教データ: {len(df_oikiri)}件 / {df_oikiri["race_id"].nunique()}レース')

    # 学習データ読み込み
    logger.info('学習データ読み込み中...')
    with zipfile.ZipFile('learning_data_perfect_tier.zip') as z:
        with z.open(z.namelist()[0]) as f:
            df_learn = pd.read_csv(f, usecols=['日付', 'レースID', '馬番', '着順パーセント', '着順', '出走頭数'],
                                   encoding='utf-8')
    df_learn['日付'] = pd.to_datetime(df_learn['日付'], errors='coerce')
    df_learn = df_learn[df_learn['日付'] >= '2025-01-01']
    df_learn['レースID'] = df_learn['レースID'].astype(str)
    df_learn['馬番'] = pd.to_numeric(df_learn['馬番'], errors='coerce')
    df_learn['着順'] = pd.to_numeric(df_learn['着順'], errors='coerce')
    df_learn['着順パーセント'] = pd.to_numeric(df_learn['着順パーセント'], errors='coerce')

    # 結合
    df = pd.merge(
        df_learn,
        df_oikiri[['race_id', '馬番', '評価']],
        left_on=['レースID', '馬番'],
        right_on=['race_id', '馬番'],
        how='inner',
    )
    logger.info(f'結合後: {len(df)}件 ({df["レースID"].nunique()}レース)')

    if len(df) < 100:
        logger.warning('データ不足のため分析をスキップします')
        return

    df['1着フラグ'] = (df['着順'] == 1).astype(int)
    df['複勝フラグ'] = (df['着順'] <= 3).astype(int)

    # ─── グレード別集計 ────────────────────────────────────────────
    print('\n' + '='*60)
    print('【調教評価グレード別 成績集計】')
    print('='*60)

    grade_order = ['S', 'A', 'B', 'C']
    stats = []
    for grade in grade_order:
        g = df[df['評価'] == grade]
        if len(g) == 0:
            continue
        stats.append({
            'グレード': grade,
            '頭数': len(g),
            'レース数': g['レースID'].nunique(),
            '1着率(%)': g['1着フラグ'].mean() * 100,
            '複勝率(%)': g['複勝フラグ'].mean() * 100,
            '平均着順%': g['着順パーセント'].mean(),
        })

    df_stats = pd.DataFrame(stats)
    print(df_stats.to_string(index=False, float_format='{:.3f}'.format))

    # ─── B基準での差分 ────────────────────────────────────────────
    b_row = df_stats[df_stats['グレード'] == 'B']
    if len(b_row) == 0:
        logger.warning('Bグレードのデータなし、分析スキップ')
        return

    b_win    = b_row['1着率(%)'].values[0]
    b_place  = b_row['複勝率(%)'].values[0]
    b_rank   = b_row['平均着順%'].values[0]

    print('\n' + '='*60)
    print('【B(標準)基準との差分】')
    print('='*60)
    for _, row in df_stats.iterrows():
        g = row['グレード']
        delta_win   = row['1着率(%)']   - b_win
        delta_place = row['複勝率(%)']  - b_place
        delta_rank  = row['平均着順%']  - b_rank
        print(f"  {g}: 1着率差 {delta_win:+.2f}pp / 複勝率差 {delta_place:+.2f}pp / 着順%差 {delta_rank:+.4f}")

    # ─── 補正値の推定 ────────────────────────────────────────────
    print('\n' + '='*60)
    print('【補正値の推奨（着順パーセント差をraw_scoreスケールに変換）】')
    print('='*60)
    print('変換式: 着順%の差 × (-1) × スケール係数')
    print('  ※ 着順%が低いほど良い成績のため符号反転')
    print('  ※ スケール係数 = 0.5（raw_scoresの典型レンジ~1.0の半分を想定）')
    print()

    SCALE = 0.5  # raw_score スケール係数（後で調整可能）
    recommendations = {}
    for _, row in df_stats.iterrows():
        g = row['グレード']
        delta_rank = row['平均着順%'] - b_rank
        recommended = round(-delta_rank * SCALE, 4)
        recommendations[g] = recommended
        current = {'S': 0.06, 'A': 0.03, 'B': 0.0, 'C': -0.04}.get(g, 0.0)
        print(f"  {g}: 推奨={recommended:+.4f}  (現在={current:+.4f})")

    print()
    print('  ↑ スケール係数 SCALE=0.5 は調整可能。')
    print('    より保守的にするなら 0.2〜0.3、積極的なら 0.6〜0.8')

    # ─── サンプル数チェック ────────────────────────────────────────
    print('\n' + '='*60)
    print('【信頼性チェック】')
    print('='*60)
    total = len(df)
    for _, row in df_stats.iterrows():
        n = row['頭数']
        pct = n / total * 100
        reliability = '★★★ 十分' if n >= 500 else ('★★ やや少' if n >= 200 else '★ 要注意')
        print(f"  {row['グレード']}: {n}頭 ({pct:.1f}%) {reliability}")

    print('\n' + '='*60)
    print('【補足: レースごとのグレード分布】')
    print('='*60)
    per_race = df.groupby('レースID')['評価'].value_counts().unstack(fill_value=0)
    print(f"  1レースあたり平均グレード数:")
    if 'S' in per_race.columns:
        print(f"    S: {per_race['S'].mean():.2f}頭/レース")
    if 'A' in per_race.columns:
        print(f"    A: {per_race['A'].mean():.2f}頭/レース")
    if 'B' in per_race.columns:
        print(f"    B: {per_race['B'].mean():.2f}頭/レース")
    if 'C' in per_race.columns:
        print(f"    C: {per_race['C'].mean():.2f}頭/レース")

    # ─── CSV保存 ────────────────────────────────────────────────
    out_path = 'oikiri_analysis.csv'
    df_stats.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f'\n  集計結果を保存: {out_path}')
    print('='*60)


# ─── エントリポイント ─────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--analyze', action='store_true', help='分析のみ実行（スクレイピングスキップ）')
    parser.add_argument('--since', default='2025-01-01', help='取得開始日 (default: 2025-01-01)')
    args = parser.parse_args()

    if not args.analyze:
        race_ids = load_race_ids(since=args.since)
        run_scraping(race_ids)

    run_analysis()
