"""
backfill_race_name.py - レース名補完スクリプト
=================================================
学習データ中の「レース名」列が空欄のレースIDに対して
ネットからレース名だけを取得して補完します。

全データの取り直しは不要。レース名のみ軽量スクレイプします。

【使い方】
  python backfill_race_name.py           # 全空欄レースを補完
  python backfill_race_name.py --dry-run  # 件数確認のみ（保存しない）
  python backfill_race_name.py --limit 100 # 最初の100件だけ試す

【所要時間の目安】
  2022年〜2025年 ≒ 約3,000〜4,000レース
  1レースあたり約2秒 → 約2〜2.5時間
  → GitHub Actions でバックグラウンド実行推奨
=================================================
"""

import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import time
import random
import zipfile
import os
import sys

CSV_FILE = 'learning_data_perfect_tier.csv'
ZIP_FILE = 'learning_data_perfect_tier.zip'

_UA_LIST = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
]
def get_headers():
    return {"User-Agent": random.choice(_UA_LIST)}

def safe_sleep(base=1.8, jitter=0.8):
    time.sleep(base + random.uniform(0, jitter))

PLACE_DICT = {
    '01':'札幌','02':'函館','03':'福島','04':'新潟','05':'東京',
    '06':'中山','07':'中京','08':'京都','09':'阪神','10':'小倉'
}

def fetch_race_name(race_id: str, retry=3) -> str:
    """
    1レースIDからレース名だけを取得して返す。
    取得できない場合はフォールバック文字列を返す。
    """
    place = PLACE_DICT.get(str(race_id)[4:6], '不明')
    r_num = int(str(race_id)[10:12])
    fallback = f"{place}{r_num}R"

    for attempt in range(retry):
        for url in [
            f"https://db.netkeiba.com/race/{race_id}/",
            f"https://race.netkeiba.com/race/result.html?race_id={race_id}",
        ]:
            try:
                r = requests.get(url, headers=get_headers(), timeout=12)
                if r.status_code in (403, 503):
                    print(f"    ⚠️ HTTP {r.status_code} (attempt {attempt+1}/{retry}) 待機中...")
                    time.sleep(15 * (attempt + 1))
                    continue
                r.encoding = 'euc-jp'
                soup = BeautifulSoup(r.text, 'html.parser')

                # 複数のセレクタを試みる
                for selector in [
                    'h1.RaceName',
                    'div.RaceName',
                    'h1.race_name',
                    'div.race_name',
                    'h1',
                ]:
                    tag = soup.select_one(selector)
                    if tag:
                        name = re.sub(r'\s+', '', tag.text).strip()
                        # 明らかに違う（ページタイトルなど）は除外
                        if name and len(name) >= 2 and len(name) <= 30:
                            return name

            except Exception as e:
                print(f"    ⚠️ 取得エラー {race_id} (attempt {attempt+1}): {e}")

        safe_sleep(2.0, 1.0)

    return fallback  # 全試行失敗 → フォールバック


def main():
    dry_run = '--dry-run' in sys.argv
    limit = None
    if '--limit' in sys.argv:
        idx = sys.argv.index('--limit')
        try:
            limit = int(sys.argv[idx + 1])
        except (IndexError, ValueError):
            pass

    print("=" * 60)
    print("レース名補完スクリプト")
    print("=" * 60)

    # ── データ読み込み ──────────────────────────────────────────
    df = None
    for path in [ZIP_FILE, CSV_FILE]:
        if os.path.exists(path):
            kw = {'compression': 'zip'} if path.endswith('.zip') else {}
            print(f"📊 データ読み込み: {path}")
            df = pd.read_csv(path, dtype=str, **kw)
            break
    if df is None:
        print("❌ 学習データファイルが見つかりません。")
        sys.exit(1)

    print(f"  → {len(df):,}行 読み込み完了")

    # ── レース名が空 or NaN の行を特定 ─────────────────────────
    if 'レース名' not in df.columns:
        print("⚠️ 「レース名」列が存在しません。新規作成します。")
        df['レース名'] = ''

    # 空欄 or NaN or フォールバック形式（〇〇NR）のみ対象
    empty_mask = (
        df['レース名'].isna() |
        (df['レース名'].str.strip() == '') |
        df['レース名'].str.match(r'^[^\s]{1,3}\d{1,2}R$')  # "東京12R"形式のフォールバック
    )

    # 対象レースIDを一意に抽出
    target_ids = df.loc[empty_mask, 'レースID'].unique().tolist()

    # 指数表記バグを修正
    target_ids = [
        str(int(float(rid))).zfill(12) if re.match(r'[\d.]+[Ee][+\-]?\d+', str(rid)) else str(rid)
        for rid in target_ids
    ]

    print(f"\n📋 レース名が空欄のレース数: {len(target_ids):,}件")

    if dry_run:
        print("（--dry-run モード: 保存は行いません）")
        print(f"実際に補完する場合: python backfill_race_name.py")
        return

    if limit:
        target_ids = target_ids[:limit]
        print(f"（--limit {limit} モード: 先頭{limit}件のみ処理）")

    if not target_ids:
        print("✅ 補完が必要なレースはありません！")
        return

    print(f"\n🔄 補完開始... (合計 {len(target_ids)} レース)")
    print("Ctrl+C で中断可能（途中まで補完済みデータは保存されます）\n")

    # ── 1レースIDごとにレース名を取得 ────────────────────────
    race_name_map = {}  # race_id → race_name
    success = 0
    fallback_count = 0

    try:
        for i, rid in enumerate(target_ids):
            name = fetch_race_name(rid)

            place = PLACE_DICT.get(str(rid)[4:6], '不明')
            r_num = int(str(rid)[10:12])
            fb = f"{place}{r_num}R"
            if name == fb:
                fallback_count += 1
            else:
                success += 1

            race_name_map[rid] = name

            # 進捗表示（10件ごと）
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  [{i+1:4d}/{len(target_ids)}] {rid} → {name}")

            # 途中保存（100件ごと）
            if (i + 1) % 100 == 0:
                _apply_and_save(df, race_name_map, CSV_FILE, ZIP_FILE, interim=True)
                print(f"  💾 中間保存完了 ({i+1}件処理済み)")

            safe_sleep(1.8, 0.8)

    except KeyboardInterrupt:
        print("\n\n⚠️ 中断されました。途中まで補完したデータを保存します...")

    # ── 最終保存 ─────────────────────────────────────────────
    print(f"\n📊 結果: 成功={success}件 / フォールバック={fallback_count}件")
    _apply_and_save(df, race_name_map, CSV_FILE, ZIP_FILE, interim=False)

    print("\n" + "=" * 60)
    print("🎉 レース名補完完了！")
    print(f"  補完件数: {len(race_name_map):,} レース")
    print("  次回の再学習時から レースクラスコード 特徴量が有効になります。")
    print("=" * 60)


def _apply_and_save(df, race_name_map, csv_file, zip_file, interim=False):
    """補完マップをDataFrameに適用して保存する"""
    df_out = df.copy()

    # レースIDを正規化して突き合わせ
    def norm_rid(x):
        s = str(x)
        if re.match(r'[\d.]+[Ee][+\-]?\d+', s):
            return str(int(float(s))).zfill(12)
        return s

    df_out['_rid_norm'] = df_out['レースID'].apply(norm_rid)

    if 'レース名' not in df_out.columns:
        df_out['レース名'] = ''

    # 空欄の行にのみ適用（すでに値があれば上書きしない）
    for rid, name in race_name_map.items():
        mask = (
            df_out['_rid_norm'] == rid
        ) & (
            df_out['レース名'].isna() |
            (df_out['レース名'].str.strip() == '') |
            df_out['レース名'].str.match(r'^[^\s]{1,3}\d{1,2}R$')
        )
        df_out.loc[mask, 'レース名'] = name

    df_out = df_out.drop(columns=['_rid_norm'])

    label = "中間" if interim else "最終"
    print(f"\n💾 {label}保存中...")
    df_out.to_csv(csv_file, index=False, encoding='utf-8-sig')
    with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_file)
    print(f"✅ {csv_file} + {zip_file} {label}保存完了")


if __name__ == '__main__':
    main()
