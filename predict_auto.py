"""
自動予想スクリプト（GitHub Actions から呼び出し）
- 本日の開催レースを取得
- 発走まで window_min〜window_max 分のレースを対象に推論
- Discord Webhook に直接送信（GitHub Actions は Discord への通信が可能）

- 同じレースを何度も投稿しないよう、HF Hub の投稿済みレジストリ（posted_races.json）で
  重複を防ぐ。アプリからの投稿とも共有されるため「1レース1回」になる。

使い方:
  python predict_auto.py [--window-min 5] [--window-max 25]

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
    from src.discord_utils import posted_races_get, posted_races_mark

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


def _send_alert(text: str) -> bool:
    """直前予想が全滅/クラッシュしたとき Discord に警告を送る（サイレント障害の検知用）。

    15分毎に走るため「開催なし/対象レースなし」は正常が多い。ノイズを避けるため
    アラートは (a) 例外クラッシュ (b) 対象レースがあったのに全て失敗 の2ケースに絞る。
    """
    if not DISCORD_WEBHOOK_URL:
        return False
    try:
        resp = requests.post(
            DISCORD_WEBHOOK_URL,
            json={"content": text[:1900], "username": "keiba-ebye ⚠️"},
            timeout=15,
        )
        if resp.status_code not in (200, 204):
            logger.warning(f"警告送信失敗 HTTP {resp.status_code}: {resp.text[:200]}")
            return False
        return True
    except Exception as e:
        logger.error(f"警告送信エラー: {e}")
        return False


def run(window_min: int = 5, window_max: int = 25):
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

    # ── 投稿済みレースを除外（2026-09-25）────────────────────────────────
    # 旧実装はウィンドウ(10〜60分前)に入っているレースを毎回投稿していたため、
    # 15分ポーリングで同じレースが3〜4回 Discord に流れていた。
    # HF Hub の共有レジストリで「1レース1回」にし、アプリからの投稿とも重複させない。
    already = posted_races_get(date_str)
    if already:
        logger.info(f"本日の投稿済み: {len(already)}R")
    fresh = [(r, m) for r, m in targets if r["id"] not in already]
    if not fresh:
        logger.info(f"対象 {len(targets)}R はすべて投稿済み。終了。")
        return
    targets = fresh

    logger.info(f"予想対象: {len(targets)} レース（未投稿のみ）")
    bundle = load_bundle()

    sent_ok = 0
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
            sent_ok += 1
            # 投稿できたレースだけ記録する（失敗したら次のポーリングで再挑戦される）
            posted_races_mark(race_id, date_str)
            logger.info(f"Discord 送信成功: {race['place']} {race['num']}R")
        else:
            logger.warning(f"Discord 送信失敗: {race_id}")

    # 対象レースがあったのに1件も送れなかった＝推論/送信の全滅（サイレント障害）
    if targets and sent_ok == 0:
        logger.error("対象レースがあったが1件も送信できず。")
        _send_alert(
            f"⚠️ **直前予想 全滅** | {now.strftime('%m/%d %H:%M')} JST\n"
            f"発走 {window_min}〜{window_max}分前の {len(targets)}R を対象にしましたが、"
            f"推論または送信が全て失敗しました。\n"
            f"モデルロード/スクレイプ/特徴量不整合の可能性。Actionsログを確認してください。"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="自動予想 → Discord 通知")
    # 既定は 5〜25分前。幅20分 > ポーリング間隔15分 なので、cronが多少ずれても
    # 各レースは必ず1回はウィンドウ内で観測される（投稿済みレジストリで重複は防ぐ）。
    parser.add_argument("--window-min", type=int, default=5, help="発走まで何分以上のレースを対象にするか")
    parser.add_argument("--window-max", type=int, default=25, help="発走まで何分以内のレースを対象にするか")
    args = parser.parse_args()
    try:
        run(args.window_min, args.window_max)
    except Exception as e:
        logger.exception("直前予想が異常終了しました")
        try:
            _dt = datetime.datetime.now(JST).strftime("%m/%d %H:%M")
        except Exception:
            _dt = "?"
        _send_alert(
            f"🔴 **直前予想クラッシュ** | {_dt} JST\n"
            f"`predict_auto.py` が例外で停止しました:\n"
            f"```{type(e).__name__}: {str(e)[:400]}```\n"
            f"Actionsログを確認してください。"
        )
        sys.exit(1)
