"""
開催日ヘルスチェック（GitHub Actions・毎晩実行）

目的:
  「振り返りが数ヶ月サイレントに蓄積されない」事故（2026-04〜09 に発生）の再発防止。
  その日の通知がすべて終わった後に、成績履歴 CSV に開催日の行が実際に書き込まれたかを
  検査し、抜けていれば Discord（振り返りチャンネル）に警告を出す「見張り番」。

★2026-09-25 の変更:
  旧実装は「今週の土曜以降の行があるか」だけを見ていたため、
  祝日の月曜開催・火曜開催（例: 2026-09-21/22）の振り返りが丸ごと欠けても
  土日の行さえあれば正常判定になってしまっていた。
  → netkeiba で直近7日間の「実際に開催があった日」を調べ、その各日について
     ai_daily_history.csv に行があるかを突き合わせる方式に変更。

検査対象（HF Hub / Dataset リポジトリ）:
  - ai_daily_history.csv  … 日次成績（auto_review.py が書く）
  - ai_race_history.csv   … レース×馬 明細（auto_review.py が書く・2026-08-29〜）

使い方:
  python health_check.py [--days 7]

必要な環境変数:
  HF_TOKEN                    - HuggingFace API トークン（read 権限）
  HF_REPO_ID                 - モデル保存先 Dataset リポジトリ ID
  DISCORD_REVIEW_WEBHOOK_URL - 振り返りチャンネル Webhook（無ければ DISCORD_WEBHOOK_URL）
"""

import os
import re
import sys
import argparse
import datetime
import logging

import pytz
import requests
import pandas as pd

from src.config import get_headers, safe_sleep

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


def _has_jra_races(date_str8: str) -> "bool | None":
    """その日に JRA の開催があったか。判定できなければ None を返す。

    src.scraper は streamlit に依存するためここでは使わず、レース一覧ページから
    race_id を正規表現で拾うだけの軽い実装にする（場コード01〜10がJRA）。
    """
    for url in (
        f"https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={date_str8}",
        f"https://db.netkeiba.com/race/list/{date_str8}/",
    ):
        try:
            res = requests.get(url, headers=get_headers(), timeout=10)
            if res.status_code != 200:
                continue
            try:
                html = res.content.decode("utf-8")
            except UnicodeDecodeError:
                html = res.content.decode("euc-jp", errors="replace")
            ids = re.findall(r"race_id=(\d{12})", html) or re.findall(r"/race/(\d{12})", html)
            jra = [i for i in ids if 1 <= int(i[4:6]) <= 10]
            if jra:
                return True
            # ページは取れたが JRA のレースが無い → 無開催（1本目のURLで確定させる）
            if url.startswith("https://race."):
                return False
        except Exception as e:
            logger.warning(f"開催判定の取得失敗 {date_str8} ({url}): {e}")
        finally:
            safe_sleep(0.8, 0.4)
    return None


def _history_dates(repo_id: str, filename: str) -> set:
    """HF Hub の CSV を取得し、'日付' 列に含まれる日付の集合を返す。失敗時は例外。"""
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id, filename, repo_type="dataset", token=HF_TOKEN)
    df = pd.read_csv(path)
    if "日付" not in df.columns or df.empty:
        return set()
    dates = pd.to_datetime(df["日付"], errors="coerce").dropna()
    return set(dates.dt.date)


def run(days: int = 7):
    if not HF_TOKEN or not HF_REPO_ID:
        logger.error("HF_TOKEN / HF_REPO_ID が未設定です。")
        sys.exit(1)

    today = datetime.datetime.now(JST).date()
    logger.info(f"実行日={today} 検査範囲=直近{days}日")

    # ── daily 履歴の取得 ──────────────────────────────────────────────
    try:
        have = _history_dates(HF_REPO_ID, "ai_daily_history.csv")
    except Exception as e:
        logger.error(f"ai_daily_history.csv 取得失敗: {e}")
        _alert(
            f"🩺 **ヘルスチェック異常** | {today}\n"
            f"`ai_daily_history.csv` を HF Hub から取得できませんでした:\n"
            f"```{type(e).__name__}: {str(e)[:300]}```\n"
            f"HFトークン/リポジトリ、または保存処理を確認してください。"
        )
        return

    if not have:
        _alert(
            f"🩺 **ヘルスチェック異常** | {today}\n"
            f"`ai_daily_history.csv` に有効な日付行がありません。振り返り保存が壊れている可能性。"
        )
        return

    # ── 直近 days 日の「実際に開催があった日」を調べ、履歴と突き合わせる ─────────
    missing, checked, undetermined = [], [], []
    for i in range(days):
        d = today - datetime.timedelta(days=i)
        held = _has_jra_races(d.strftime("%Y%m%d"))
        if held is None:
            undetermined.append(d)
            continue
        if not held:
            continue
        checked.append(d)
        if d not in have:
            missing.append(d)

    logger.info(f"開催日={[str(d) for d in checked]} / 未記録={[str(d) for d in missing]} "
                f"/ 判定不能={[str(d) for d in undetermined]}")

    if missing:
        _list = "・".join(str(d) for d in sorted(missing))
        _alert(
            f"🩺 **振り返りが未蓄積の開催日があります** | {today}\n"
            f"開催があったのに `ai_daily_history.csv` に行が無い日: **{_list}**\n"
            f"（直近{days}日の開催日 {len(checked)}日中 {len(missing)}日が欠落）\n"
            f"→ 振り返りが走っていない/0件return/クラッシュの可能性。"
            f"「自動振り返り」Actionsを確認し、"
            f"`python auto_review.py --date YYYYMMDD` で手動リカバリできます。"
        )
        return

    if not checked and undetermined:
        _alert(
            f"🩺 **ヘルスチェック: 開催判定ができません** | {today}\n"
            f"直近{days}日のレース一覧ページを1日も取得できませんでした"
            f"（{len(undetermined)}日が判定不能）。\n"
            f"netkeibaのブロック/構造変更の可能性があります。"
        )
        return

    # ── 正常。race 明細も参考にチェックし、ズレていればログに注記。────────────
    note = ""
    if checked:
        try:
            r_have = _history_dates(HF_REPO_ID, "ai_race_history.csv")
            r_missing = [d for d in checked if d not in r_have]
            if r_missing:
                note = f"（注: ai_race_history.csv に {[str(d) for d in r_missing]} の明細が無い）"
        except Exception as e:
            note = f"（注: ai_race_history.csv の確認に失敗: {type(e).__name__}）"

    logger.info(f"正常: 直近{days}日の開催 {len(checked)}日すべて記録済み {note}")
    # 正常時は Discord へ通知しない（ノイズ回避）。ログのみ。


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="開催日ヘルスチェック → Discord 警告")
    parser.add_argument("--days", type=int, default=7, help="何日前まで検査するか（既定7）")
    args = parser.parse_args()
    run(args.days)
