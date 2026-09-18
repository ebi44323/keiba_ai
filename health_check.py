"""
週末ヘルスチェック（GitHub Actions・日曜夜に実行）

目的:
  「振り返りが数ヶ月サイレントに蓄積されない」事故（2026-04〜09 に発生）の再発防止。
  各週末の全通知が終わった後に、成績履歴 CSV に今週末の行が実際に書き込まれたかを検査し、
  見当たらなければ Discord（振り返りチャンネル）に警告を出す「見張り番」。

検査対象（HF Hub / Dataset リポジトリ）:
  - ai_daily_history.csv  … 日次成績（auto_review.py が書く）
  - ai_race_history.csv   … レース×馬 明細（auto_review.py が書く・2026-08-29〜）

判定:
  実行日（日曜）を含む今週末の土曜以降の日付を持つ行が daily 履歴に無ければ警告。
  ※ 本当にJRA無開催の週末なら誤検知しうるが、その場合は無視可（数ヶ月の沈黙よりはるかに安全）。

使い方:
  python health_check.py

必要な環境変数:
  HF_TOKEN                    - HuggingFace API トークン（read 権限）
  HF_REPO_ID                 - モデル保存先 Dataset リポジトリ ID
  DISCORD_REVIEW_WEBHOOK_URL - 振り返りチャンネル Webhook（無ければ DISCORD_WEBHOOK_URL）
"""

import os
import sys
import datetime
import logging

import pytz
import requests
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("health_check")

HF_TOKEN   = os.environ.get("HF_TOKEN", "")
HF_REPO_ID = os.environ.get("HF_REPO_ID", "")
WEBHOOK    = (os.environ.get("DISCORD_REVIEW_WEBHOOK_URL", "").strip()
             or os.environ.get("DISCORD_WEBHOOK_URL", "").strip())

JST = pytz.timezone("Asia/Tokyo")


def _alert(text: str) -> bool:
    if not WEBHOOK:
        logger.error("Webhook 未設定のため警告送信をスキップ")
        return False
    try:
        resp = requests.post(
            WEBHOOK,
            json={"content": text[:1900], "username": "keiba-ebye 🩺"},
            timeout=15,
        )
        if resp.status_code not in (200, 204):
            logger.warning(f"警告送信失敗 HTTP {resp.status_code}: {resp.text[:200]}")
            return False
        return True
    except Exception as e:
        logger.error(f"警告送信エラー: {e}")
        return False


def _latest_date(repo_id: str, filename: str) -> "pd.Timestamp | None":
    """HF Hub の CSV を取得し、'日付' 列の最新日付を返す。取得/解析失敗は例外を送出。"""
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id, filename, repo_type="dataset", token=HF_TOKEN)
    df = pd.read_csv(path)
    if "日付" not in df.columns or df.empty:
        return None
    dates = pd.to_datetime(df["日付"], errors="coerce").dropna()
    return dates.max() if not dates.empty else None


def run():
    if not HF_TOKEN or not HF_REPO_ID:
        logger.error("HF_TOKEN / HF_REPO_ID が未設定です。")
        sys.exit(1)

    today = datetime.datetime.now(JST).date()
    # 今週末の土曜（実行日以前で直近の土曜）。日曜実行なら前日、土曜実行なら当日。
    saturday = today - datetime.timedelta(days=(today.weekday() - 5) % 7)
    logger.info(f"実行日={today} 今週末の土曜={saturday}")

    # ── daily 履歴の検査（本命）──────────────────────────────────────────
    try:
        latest = _latest_date(HF_REPO_ID, "ai_daily_history.csv")
    except Exception as e:
        logger.error(f"ai_daily_history.csv 取得失敗: {e}")
        _alert(
            f"🩺 **ヘルスチェック異常** | {today}\n"
            f"`ai_daily_history.csv` を HF Hub から取得できませんでした:\n"
            f"```{type(e).__name__}: {str(e)[:300]}```\n"
            f"HFトークン/リポジトリ、または保存処理を確認してください。"
        )
        return

    if latest is None:
        _alert(
            f"🩺 **ヘルスチェック異常** | {today}\n"
            f"`ai_daily_history.csv` に有効な日付行がありません。振り返り保存が壊れている可能性。"
        )
        return

    latest_d = latest.date()
    if latest_d < saturday:
        gap = (today - latest_d).days
        _alert(
            f"🩺 **今週末の振り返りが未蓄積** | {today}\n"
            f"`ai_daily_history.csv` の最新行は **{latest_d}**（{gap}日前）。"
            f"今週末（{saturday}〜）の行が見当たりません。\n"
            f"→ 振り返りが走っていない/0件return/クラッシュの可能性。"
            f"「自動振り返り」Actionsを確認してください。"
            f"（本当にJRA無開催の週末なら無視して構いません）"
        )
        logger.warning(f"未蓄積を検知: 最新={latest_d} < 土曜={saturday}")
        return

    # ── 正常（今週末の行あり）。race 明細も参考にチェックし、ズレていれば軽い注記。──
    note = ""
    try:
        r_latest = _latest_date(HF_REPO_ID, "ai_race_history.csv")
        if r_latest is not None and r_latest.date() < saturday:
            note = f"\n（注: ai_race_history.csv は {r_latest.date()} 止まりで明細が遅れています）"
    except Exception as e:
        note = f"\n（注: ai_race_history.csv の確認に失敗: {type(e).__name__}）"

    logger.info(f"正常: daily最新={latest_d} >= 土曜={saturday}{note}")
    # 正常時は Discord へ通知しない（ノイズ回避）。ログのみ。


if __name__ == "__main__":
    run()
