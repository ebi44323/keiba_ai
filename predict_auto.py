"""
自動予想スクリプト（GitHub Actions から呼び出し）
- 本日の開催レースを取得
- 発走まで 15〜50 分のレースを対象に推論
- 予想を Discord キュー（HF Hub）に書き込む
- discord_notify.yml（5分毎）がキューを読んで Discord に送信する

使い方:
  python predict_auto.py [--window-min 15] [--window-max 50]

必要な環境変数:
  HF_TOKEN        - HuggingFace API トークン（read/write 権限）
  HF_REPO_ID      - モデル保存先 Dataset リポジトリ ID
  DISCORD_WEBHOOK_URL (任意 / discord_utils.py が参照)
"""

import os
import sys
import argparse
import datetime
import logging
import unittest.mock as mock
import pytz

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("predict_auto")

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
    from src.scraper import get_todays_races
    from src.inference import run_real_prediction
    from src.discord_utils import send_discord_prediction

JST = pytz.timezone("Asia/Tokyo")


def load_bundle():
    logger.info("モデルを HF Hub からロード中...")
    bundle = prepare_model_and_data(force_retrain=False)
    logger.info("モデルロード完了")
    return bundle


def run(window_min: int = 15, window_max: int = 50):
    now = datetime.datetime.now(JST)
    date_str = now.strftime("%Y-%m-%d")
    logger.info(f"実行日時: {now.strftime('%Y-%m-%d %H:%M')} JST")

    races = get_todays_races(now.strftime("%Y%m%d"))
    if not races:
        logger.info("本日の開催なし。終了。")
        return

    logger.info(f"{len(races)} レース取得")

    # 発走まで window_min〜window_max 分のレースを対象にする
    targets = []
    for r in races:
        mins = (r["time"] - now).total_seconds() / 60
        if window_min <= mins <= window_max:
            targets.append((r, int(mins)))

    if not targets:
        logger.info(f"対象レースなし (発走まで {window_min}〜{window_max} 分のレースがありません)")
        return

    logger.info(f"予想対象: {len(targets)} レース")
    bundle = load_bundle()

    for race, mins_left in targets:
        race_id = race["id"]
        logger.info(f"推論中: {race['place']} {race['num']}R ({race_id}) 発走まで {mins_left}分")
        try:
            res_df, topics, reco, pace_text, conf_text, _, _, _, err_log = run_real_prediction(
                race_id, date_str, bundle,
                skip_live_scrape=False,
                ev_first=True,
                ev_threshold=1.0,
                min_win_prob=0.10,
            )
        except Exception as e:
            logger.error(f"推論失敗 {race_id}: {e}")
            continue

        if res_df is None:
            logger.warning(f"推論結果なし {race_id}: {err_log}")
            continue

        race_info = {
            "race_id":   race_id,
            "place":     race["place"],
            "num":       race["num"],
            "title":     race["title"],
            "mins_left": mins_left,
        }
        ok = send_discord_prediction(res_df, topics, reco, pace_text, conf_text, race_info)
        if ok:
            logger.info(f"Discord キューに追加: {race['place']} {race['num']}R")
        else:
            logger.warning(f"Discord キュー書き込み失敗: {race_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="自動予想 → Discord 通知")
    parser.add_argument("--window-min", type=int, default=10, help="発走まで何分以上のレースを対象にするか")
    parser.add_argument("--window-max", type=int, default=60, help="発走まで何分以内のレースを対象にするか")
    args = parser.parse_args()
    run(args.window_min, args.window_max)
