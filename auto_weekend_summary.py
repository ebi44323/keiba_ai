"""
土日まとめ振り返りスクリプト（GitHub Actions から呼び出し）
- 直近の土曜・日曜の全レースを集計
- 2日分合算の成績を Discord に送信（キュー経由）

使い方:
  python auto_weekend_summary.py [--sat YYYYMMDD] [--sun YYYYMMDD]
  （省略時は直近の土・日を自動判定）

必要な環境変数:
  HF_TOKEN        - HuggingFace API トークン
  HF_REPO_ID      - モデル保存先 Dataset リポジトリ ID
"""

import os
import sys
import argparse
import datetime
import logging
import unittest.mock as mock
import pytz

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("auto_weekend_summary")

HF_TOKEN   = os.environ.get("HF_TOKEN", "")
HF_REPO_ID = os.environ.get("HF_REPO_ID", "")

if not HF_TOKEN or not HF_REPO_ID:
    logger.error("HF_TOKEN / HF_REPO_ID が未設定です。")
    sys.exit(1)

def _passthrough(func=None, **kw):
    if callable(func): return func
    return lambda f: f

with mock.patch("streamlit.cache_resource", _passthrough), \
     mock.patch("streamlit.cache_data",     _passthrough), \
     mock.patch("streamlit.spinner",        lambda *a, **kw: mock.MagicMock()):
    from src.core_model import prepare_model_and_data
    from src.scraper import get_todays_races, get_all_payouts
    from src.inference import run_real_prediction
    from src.discord_utils import _push_discord_queue

JST = pytz.timezone("Asia/Tokyo")

_EMPTY_STATS = lambda: {
    "honmei_races": 0, "honmei_tan_hits": 0, "honmei_tan_return": 0,
    "honmei_fuku_hits": 0, "honmei_fuku_return": 0,
    "umaren_races": 0, "umaren_invest": 0, "umaren_hits": 0, "umaren_return": 0,
    # 超狙い馬: AI上位5頭(index<5) かつ EV>=1.5
    "choko_invest": 0, "choko_tan_hits": 0, "choko_tan_return": 0,
    "choko_fuku_hits": 0, "choko_fuku_return": 0,
    # 穴馬: AI6位以下(index>=5) かつ EV>=1.5
    "ana_invest": 0, "ana_tan_hits": 0, "ana_tan_return": 0,
    "ana_fuku_hits": 0, "ana_fuku_return": 0,
    "shiba_races": 0, "shiba_return": 0, "dart_races": 0, "dart_return": 0,
}


def collect_day_stats(date_str8: str, date_hf: str, bundle) -> dict:
    """1日分の統計を収集して返す"""
    stats = _EMPTY_STATS()
    races = get_todays_races(date_str8)
    if not races:
        return stats

    logger.info(f"  {date_str8}: {len(races)} レース")
    for r in races:
        try:
            res_df, _, _, _, _, track_type, _, _, _ = run_real_prediction(
                r["id"], date_hf, bundle,
                skip_live_scrape=True, ev_first=True,
                ev_threshold=1.5, min_win_prob=0.18,
            )
        except Exception as e:
            logger.warning(f"    推論失敗 {r['id']}: {e}")
            continue
        if res_df is None:
            continue

        try:
            payouts = get_all_payouts(r["id"])
        except Exception:
            continue
        if not payouts.get("tansho"):
            continue

        honmei = res_df.iloc[0]["馬番"]
        stats["honmei_races"] += 1
        if track_type == "芝":   stats["shiba_races"] += 1
        elif track_type == "ダート": stats["dart_races"] += 1

        if honmei in payouts["tansho"]:
            stats["honmei_tan_hits"]   += 1
            pay = payouts["tansho"][honmei]
            stats["honmei_tan_return"] += pay
            if track_type == "芝":   stats["shiba_return"] += pay
            elif track_type == "ダート": stats["dart_return"] += pay
        if honmei in payouts["fukusho"]:
            stats["honmei_fuku_hits"]   += 1
            stats["honmei_fuku_return"] += payouts["fukusho"][honmei]

        if len(res_df) >= 5:
            himo = res_df.iloc[1:5]["馬番"].tolist()
            stats["umaren_races"]  += 1
            stats["umaren_invest"] += len(himo) * 100
            for h in himo:
                key = tuple(sorted([honmei, h]))
                if key in payouts.get("umaren", {}):
                    stats["umaren_hits"]   += 1
                    stats["umaren_return"] += payouts["umaren"][key]

        # 超狙い馬: AI上位5頭(index<5) かつ EV>=1.5
        for _, row in res_df[(res_df.index < 5) & (res_df["期待値"] >= 1.5)].iterrows():
            uban = row["馬番"]
            stats["choko_invest"] += 100
            if uban in payouts["tansho"]:
                stats["choko_tan_hits"]   += 1
                stats["choko_tan_return"] += payouts["tansho"][uban]
            if uban in payouts["fukusho"]:
                stats["choko_fuku_hits"]   += 1
                stats["choko_fuku_return"] += payouts["fukusho"][uban]

        # 穴馬: AI6位以下(index>=5) かつ EV>=1.5
        for _, row in res_df[(res_df.index >= 5) & (res_df["期待値"] >= 1.5)].iterrows():
            uban = row["馬番"]
            stats["ana_invest"] += 100
            if uban in payouts["tansho"]:
                stats["ana_tan_hits"]   += 1
                stats["ana_tan_return"] += payouts["tansho"][uban]
            if uban in payouts["fukusho"]:
                stats["ana_fuku_hits"]   += 1
                stats["ana_fuku_return"] += payouts["fukusho"][uban]

    return stats


def merge_stats(a: dict, b: dict) -> dict:
    return {k: a[k] + b[k] for k in a}


def build_discord_message(stats: dict, sat_label: str, sun_label: str) -> str:
    races = stats["honmei_races"]
    if races == 0:
        return f"📊 **keiba-ebye 週末まとめ** | {sat_label}・{sun_label}\nデータなし（レース結果未取得）"

    def _r(ret, inv): return round(ret / inv * 100, 1) if inv > 0 else 0.0
    def _e(v): return "🔥" if v >= 150 else "✅" if v >= 100 else "🟡" if v >= 70 else "❌"

    tan    = _r(stats["honmei_tan_return"],  races * 100)
    fuku   = _r(stats["honmei_fuku_return"], races * 100)
    uma    = _r(stats["umaren_return"],      max(stats["umaren_invest"], 1))
    choko_t = _r(stats["choko_tan_return"], max(stats["choko_invest"], 1))
    choko_f = _r(stats["choko_fuku_return"], max(stats["choko_invest"], 1))
    ana_t  = _r(stats["ana_tan_return"],    max(stats["ana_invest"], 1))
    ana_f  = _r(stats["ana_fuku_return"],   max(stats["ana_invest"], 1))
    shiba  = _r(stats["shiba_return"],      max(stats["shiba_races"] * 100, 1))
    dart   = _r(stats["dart_return"],       max(stats["dart_races"]  * 100, 1))

    tan_hi  = stats["honmei_tan_hits"]
    fuku_hi = stats["honmei_fuku_hits"]
    uma_hi  = stats["umaren_hits"]

    lines = [
        f"📊 **keiba-ebye 週末まとめ** | {sat_label} & {sun_label}",
        f"対象 **{races}レース**（土 {stats.get('_sat_races',0)}R + 日 {stats.get('_sun_races',0)}R）",
        "",
        "**【本命(◎) 成績 2日合計】**",
        "```",
        f"単勝  {_e(tan)} {tan:6.1f}%   的中 {tan_hi}/{races}R",
        f"複勝  {_e(fuku)} {fuku:6.1f}%   的中 {fuku_hi}/{races}R",
        f"馬連  {_e(uma)} {uma:6.1f}%   的中 {uma_hi}回",
        "```",
        "",
        "**【超狙い馬(AI上位5頭 EV1.5+) ベタ買い】**",
        "```",
        f"単勝  {_e(choko_t)} {choko_t:6.1f}%   的中 {stats['choko_tan_hits']}/{int(stats['choko_invest']//100)}頭",
        f"複勝  {_e(choko_f)} {choko_f:6.1f}%   的中 {stats['choko_fuku_hits']}/{int(stats['choko_invest']//100)}頭",
        "```",
        "",
        "**【穴馬(AI6位以下 EV1.5+) ベタ買い】**",
        "```",
        f"単勝  {_e(ana_t)} {ana_t:6.1f}%   的中 {stats['ana_tan_hits']}/{int(stats['ana_invest']//100)}頭",
        f"複勝  {_e(ana_f)} {ana_f:6.1f}%   的中 {stats['ana_fuku_hits']}/{int(stats['ana_invest']//100)}頭",
        "```",
        "",
        f"🌱 芝: {shiba:.1f}%  🏜️ ダート: {dart:.1f}%",
        "",
        "-# keiba-ebye 週末まとめ / 結果は参考情報です",
    ]
    return "\n".join(lines)


def get_last_weekend() -> tuple[str, str]:
    """直近の土曜・日曜の日付文字列を返す（YYYYMMDD形式）"""
    now = datetime.datetime.now(JST)
    wd  = now.weekday()  # 月=0 ... 土=5, 日=6
    if wd == 6:          # 日曜
        sat = now - datetime.timedelta(days=1)
        sun = now
    elif wd == 5:        # 土曜
        sat = now
        sun = now + datetime.timedelta(days=1)
    else:                # 平日
        days_to_last_sun = wd + 1
        sun = now - datetime.timedelta(days=days_to_last_sun)
        sat = sun - datetime.timedelta(days=1)
    return sat.strftime("%Y%m%d"), sun.strftime("%Y%m%d")


def run(sat_str: str = None, sun_str: str = None):
    if not sat_str or not sun_str:
        sat_str, sun_str = get_last_weekend()

    sat_dt = datetime.datetime.strptime(sat_str, "%Y%m%d").replace(tzinfo=JST)
    sun_dt = datetime.datetime.strptime(sun_str, "%Y%m%d").replace(tzinfo=JST)
    sat_label = sat_dt.strftime("%Y/%m/%d")
    sun_label = sun_dt.strftime("%Y/%m/%d")

    logger.info(f"週末まとめ: {sat_label}（土）& {sun_label}（日）")
    logger.info("モデルロード中...")
    bundle = prepare_model_and_data(force_retrain=False)
    logger.info("モデルロード完了。集計中...")

    logger.info(f"土曜 {sat_label} 集計中...")
    sat_stats = collect_day_stats(sat_str, sat_dt.strftime("%Y-%m-%d"), bundle)
    logger.info(f"日曜 {sun_label} 集計中...")
    sun_stats = collect_day_stats(sun_str, sun_dt.strftime("%Y-%m-%d"), bundle)

    combined = merge_stats(sat_stats, sun_stats)
    combined["_sat_races"] = sat_stats["honmei_races"]
    combined["_sun_races"] = sun_stats["honmei_races"]

    msg = build_discord_message(combined, sat_label, sun_label)
    logger.info(f"集計完了: 合計{combined['honmei_races']}レース")

    dedup_key = f"weekend_{sat_str}_{sun_str}"
    ok = _push_discord_queue(msg, channel="review",
                             username="keiba-ebye 📊週末", dedup_key=dedup_key)
    if ok:
        logger.info("Discord キューへの書き込み成功")
    else:
        logger.error("Discord キューへの書き込み失敗")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="土日まとめ振り返り → Discord")
    parser.add_argument("--sat", type=str, default=None, help="土曜日 YYYYMMDD")
    parser.add_argument("--sun", type=str, default=None, help="日曜日 YYYYMMDD")
    args = parser.parse_args()
    run(args.sat, args.sun)
