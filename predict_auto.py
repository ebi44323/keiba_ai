"""
自動予想スクリプト（GitHub Actions から呼び出し）
- 本日の開催レースを取得
- 発走まで window_min〜window_max 分のレースを対象に推論
- Discord Webhook に直接送信（GitHub Actions は Discord への通信が可能）

使い方:
  python predict_auto.py [--window-min 10] [--window-max 60]

必要な環境変数:
  HF_TOKEN             - HuggingFace API トークン（read権限）
  HF_REPO_ID           - モデル保存先 Dataset リポジトリ ID
  DISCORD_WEBHOOK_URL  - 直前予想チャンネルの Discord Webhook URL
"""

import os
import sys
import argparse
import datetime
import logging
import unittest.mock as mock
import requests
import pytz

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("predict_auto")

HF_TOKEN            = os.environ.get("HF_TOKEN", "")
HF_REPO_ID          = os.environ.get("HF_REPO_ID", "")
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "").strip()

if not HF_TOKEN or not HF_REPO_ID:
    logger.error("HF_TOKEN / HF_REPO_ID が未設定です。GitHub Secrets を確認してください。")
    sys.exit(1)
if not DISCORD_WEBHOOK_URL:
    logger.error("DISCORD_WEBHOOK_URL が未設定です。GitHub Secrets を確認してください。")
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

JST = pytz.timezone("Asia/Tokyo")


def load_bundle():
    logger.info("モデルを HF Hub からロード中...")
    bundle = prepare_model_and_data(force_retrain=False)
    logger.info("モデルロード完了")
    return bundle


def _send_discord_direct(res_df, topics, reco, pace_text, conf_text,
                          race_info: dict) -> bool:
    """予想結果を Discord Webhook に直接送信する（GitHub Actions 用）"""
    place = race_info.get('place', '')
    num   = race_info.get('num', '')
    title = race_info.get('title', '')
    mins  = race_info.get('mins_left', 0)

    lines = [
        f"🐴 **keiba-ebye 予想** | {place} {num}R「{title}」",
        f"⏰ 発走まであと **{mins}分**",
        "",
    ]
    if conf_text:
        lines.append(f"> {conf_text}")
    if pace_text:
        lines.append(f"> {pace_text}")
    lines.append("")

    lines.append("```")
    lines.append(f"{'印':<3} {'馬番':>3} {'馬名':<12} {'オッズ':>6} {'勝率':>6} {'EV':>5}")
    lines.append("-" * 42)
    for rank, row in res_df.head(7).iterrows():
        try:
            imp  = str(row.get('印', '') or '').ljust(2)
            num_ = int(float(row.get('馬番', 0)))
            name = str(row.get('馬名', ''))[:10]
            odds = float(row.get('単勝オッズ', 0))
            wp   = float(row.get('勝率(AI予測)', 0)) * 100
            ev   = float(row.get('期待値', 0) or 0)
            ev_mark = " ★" if ev >= 1.5 else ""
            lines.append(
                f"{imp:<3} {num_:>3} {name:<12} {odds:>5.1f}倍 {wp:>5.1f}% {ev:>4.2f}{ev_mark}"
            )
        except Exception:
            continue
    lines.append("```")
    lines.append("★ = 期待値1.5以上の注目馬")
    lines.append("")

    if topics:
        lines.append("**📝 注目トピック**")
        for t in topics[:3]:
            lines.append(f"• {t.replace('**', '')}")
        lines.append("")

    if reco:
        lines.append(f"**🎯 推奨** {reco[:200]}{'…' if len(reco)>200 else ''}")

    lines.append("")
    lines.append("-# keiba-ebye AI予想 / 馬券は自己責任でお願いします")

    content = "\n".join(lines)
    try:
        resp = requests.post(
            DISCORD_WEBHOOK_URL,
            json={"content": content[:1990], "username": "keiba-ebye 🐴"},
            timeout=15,
        )
        if resp.status_code in (200, 204):
            return True
        logger.warning(f"Discord送信失敗 HTTP {resp.status_code}: {resp.text[:200]}")
        return False
    except Exception as e:
        logger.error(f"Discord送信エラー: {e}")
        return False


def run(window_min: int = 10, window_max: int = 60):
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
                ev_threshold=1.5,
                min_win_prob=0.18,
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
        ok = _send_discord_direct(res_df, topics, reco, pace_text, conf_text, race_info)
        if ok:
            logger.info(f"Discord 送信成功: {race['place']} {race['num']}R")
        else:
            logger.warning(f"Discord 送信失敗: {race_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="自動予想 → Discord 通知")
    parser.add_argument("--window-min", type=int, default=10, help="発走まで何分以上のレースを対象にするか")
    parser.add_argument("--window-max", type=int, default=60, help="発走まで何分以内のレースを対象にするか")
    args = parser.parse_args()
    run(args.window_min, args.window_max)
