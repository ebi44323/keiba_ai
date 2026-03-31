"""
自動振り返りスクリプト（GitHub Actions から呼び出し）
- 指定日（デフォルト: 本日）の全レースを振り返り
- 本命◎・期待値馬の成績を集計
- 結果を Discord キュー（HF Hub）に書き込む
- discord_notify.yml（5分毎）がキューを読んで Discord に送信する

使い方:
  python auto_review.py [--date YYYYMMDD]

必要な環境変数:
  HF_TOKEN        - HuggingFace API トークン（read/write 権限）
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
logger = logging.getLogger("auto_review")

HF_TOKEN   = os.environ.get("HF_TOKEN", "")
HF_REPO_ID = os.environ.get("HF_REPO_ID", "")

if not HF_TOKEN or not HF_REPO_ID:
    logger.error("HF_TOKEN / HF_REPO_ID が未設定です。GitHub Secrets を確認してください。")
    sys.exit(1)

# ── Streamlit デコレータをモックして src モジュールをインポート ──────────────
def _passthrough(func=None, **kw):
    if callable(func):
        return func
    return lambda f: f

with mock.patch("streamlit.cache_resource", _passthrough), \
     mock.patch("streamlit.cache_data",     _passthrough), \
     mock.patch("streamlit.spinner",        lambda *a, **kw: mock.MagicMock()):
    from src.core_model import prepare_model_and_data
    from src.scraper import get_todays_races, get_all_payouts
    from src.inference import run_real_prediction
    from src.discord_utils import send_discord_review

JST = pytz.timezone("Asia/Tokyo")


def run(date_str: str = None):
    now = datetime.datetime.now(JST)
    if date_str:
        target_dt = datetime.datetime.strptime(date_str, "%Y%m%d").replace(tzinfo=JST)
    else:
        target_dt = now
    date_label = target_dt.strftime("%Y/%m/%d")
    date_str8  = target_dt.strftime("%Y%m%d")
    date_hf    = target_dt.strftime("%Y-%m-%d")

    logger.info(f"振り返り対象日: {date_label}")

    races = get_todays_races(date_str8)
    if not races:
        logger.info("指定日のレースなし。終了。")
        return

    logger.info(f"{len(races)} レース取得。モデルロード中...")

    bundle = prepare_model_and_data(force_retrain=False)
    logger.info("モデルロード完了。推論・集計開始...")

    stats = {
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

    for r in races:
        try:
            res_df, _, _, _, _, track_type, _, _, err_log = run_real_prediction(
                r["id"], date_hf, bundle,
                skip_live_scrape=True,  # 振り返りは高速モード
                ev_first=True,
                ev_threshold=1.0,
                min_win_prob=0.15,
            )
        except Exception as e:
            logger.warning(f"推論失敗 {r['id']}: {e}")
            continue

        if res_df is None:
            logger.warning(f"推論結果なし {r['id']}")
            continue

        try:
            payouts = get_all_payouts(r["id"])
        except Exception as e:
            logger.warning(f"払戻取得失敗 {r['id']}: {e}")
            continue

        if not payouts.get("tansho"):
            continue  # レース結果未確定またはスクレイプ失敗

        honmei = res_df.iloc[0]["馬番"]
        stats["honmei_races"] += 1

        if track_type == "芝":
            stats["shiba_races"] += 1
        elif track_type == "ダート":
            stats["dart_races"] += 1

        if honmei in payouts["tansho"]:
            stats["honmei_tan_hits"] += 1
            pay = payouts["tansho"][honmei]
            stats["honmei_tan_return"] += pay
            if track_type == "芝":
                stats["shiba_return"] += pay
            elif track_type == "ダート":
                stats["dart_return"] += pay

        if honmei in payouts["fukusho"]:
            stats["honmei_fuku_hits"] += 1
            stats["honmei_fuku_return"] += payouts["fukusho"][honmei]

        # 馬連（◎から〇〜☆への流し）
        if len(res_df) >= 5:
            himo_list = res_df.iloc[1:5]["馬番"].tolist()
            stats["umaren_races"] += 1
            stats["umaren_invest"] += len(himo_list) * 100
            for himo in himo_list:
                key = tuple(sorted([honmei, himo]))
                if key in payouts.get("umaren", {}):
                    stats["umaren_hits"] += 1
                    stats["umaren_return"] += payouts["umaren"][key]

        # 超狙い馬: AI上位5頭(index<5) かつ EV>=1.5
        choko_df = res_df[(res_df.index < 5) & (res_df["期待値"] >= 1.5)]
        for _, row in choko_df.iterrows():
            uban = row["馬番"]
            stats["choko_invest"] += 100
            if uban in payouts["tansho"]:
                stats["choko_tan_hits"] += 1
                stats["choko_tan_return"] += payouts["tansho"][uban]
            if uban in payouts["fukusho"]:
                stats["choko_fuku_hits"] += 1
                stats["choko_fuku_return"] += payouts["fukusho"][uban]

        # 穴馬: AI6位以下(index>=5) かつ EV>=1.5
        ana_df = res_df[(res_df.index >= 5) & (res_df["期待値"] >= 1.5)]
        for _, row in ana_df.iterrows():
            uban = row["馬番"]
            stats["ana_invest"] += 100
            if uban in payouts["tansho"]:
                stats["ana_tan_hits"] += 1
                stats["ana_tan_return"] += payouts["tansho"][uban]
            if uban in payouts["fukusho"]:
                stats["ana_fuku_hits"] += 1
                stats["ana_fuku_return"] += payouts["fukusho"][uban]

        logger.info(
            f"  {r['place']} {r['num']}R: ◎{honmei}番 "
            f"単{'◎' if honmei in payouts['tansho'] else '×'} "
            f"複{'◎' if honmei in payouts['fukusho'] else '×'}"
        )

    races_n = stats["honmei_races"]
    if races_n == 0:
        logger.info("集計対象レースなし（結果未確定の可能性）。終了。")
        return

    def _rate(ret, inv):
        return round(ret / inv * 100, 1) if inv > 0 else 0.0

    rates = {
        "tan_rate":       _rate(stats["honmei_tan_return"],  races_n * 100),
        "fuku_rate":      _rate(stats["honmei_fuku_return"], races_n * 100),
        "choko_tan_rate": _rate(stats["choko_tan_return"], stats["choko_invest"]),
        "choko_fuku_rate":_rate(stats["choko_fuku_return"], stats["choko_invest"]),
        "ana_tan_rate":   _rate(stats["ana_tan_return"],  stats["ana_invest"]),
        "ana_fuku_rate":  _rate(stats["ana_fuku_return"], stats["ana_invest"]),
        "uma_rate":   _rate(stats["umaren_return"], max(stats["umaren_invest"], 1)),
        "shiba_rate": _rate(stats["shiba_return"], max(stats["shiba_races"] * 100, 1)),
        "dart_rate":  _rate(stats["dart_return"],  max(stats["dart_races"] * 100, 1)),
    }

    logger.info(
        f"集計完了 {races_n}R: 本命単勝{rates['tan_rate']}% 複勝{rates['fuku_rate']}%"
    )

    ok = send_discord_review(stats, rates, date_label)
    if ok:
        logger.info("Discord キューへの書き込み成功")
    else:
        logger.error("Discord キューへの書き込み失敗")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="振り返り自動集計 → Discord 通知")
    parser.add_argument("--date", type=str, default=None, help="対象日 YYYYMMDD（省略時は本日）")
    args = parser.parse_args()
    run(args.date)
