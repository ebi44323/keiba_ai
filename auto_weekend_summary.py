"""
土日まとめ振り返りスクリプト（GitHub Actions から呼び出し）

★2026-07-25 改修: 再計算をやめ、日次振り返り(auto_review.py)が保存した
  ai_daily_history.csv の土・日の行を「集計するだけ」に変更。
  旧実装は run_real_prediction を use_oikiri なしで再計算しており、日次(use_oikiri=True)
  との条件差で◎がズレ、日次Discordと週末まとめの数字が乖離していた。
  CSVを単一の真実の源にすることで構造的に一致させる。

使い方:
  python auto_weekend_summary.py [--sat YYYYMMDD] [--sun YYYYMMDD]
  （省略時は直近の土・日を自動判定）

必要な環境変数:
  HF_TOKEN        - HuggingFace API トークン
  HF_REPO_ID      - モデル保存先 Dataset リポジトリ ID
  DISCORD_WEBHOOK_URL / DISCORD_REVIEW_WEBHOOK_URL
"""

import os
import sys
import argparse
import datetime
import logging
import pytz
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("auto_weekend_summary")

HF_TOKEN                   = os.environ.get("HF_TOKEN", "")
HF_REPO_ID                 = os.environ.get("HF_REPO_ID", "")
DISCORD_WEBHOOK_URL        = os.environ.get("DISCORD_WEBHOOK_URL", "").strip()
DISCORD_REVIEW_WEBHOOK_URL = (os.environ.get("DISCORD_REVIEW_WEBHOOK_URL", "").strip()) or DISCORD_WEBHOOK_URL

if not HF_TOKEN or not HF_REPO_ID:
    logger.error("HF_TOKEN / HF_REPO_ID が未設定です。")
    sys.exit(1)

JST = pytz.timezone("Asia/Tokyo")


def get_last_weekend() -> tuple[str, str]:
    """直近の土曜・日曜の日付文字列を返す（YYYYMMDD形式）"""
    now = datetime.datetime.now(JST)
    wd = now.weekday()  # 月=0 ... 土=5, 日=6
    if wd == 6:          # 日曜
        sat = now - datetime.timedelta(days=1); sun = now
    elif wd == 5:        # 土曜
        sat = now; sun = now + datetime.timedelta(days=1)
    else:                # 平日
        sun = now - datetime.timedelta(days=wd + 1)
        sat = sun - datetime.timedelta(days=1)
    return sat.strftime("%Y%m%d"), sun.strftime("%Y%m%d")


def _f(row: dict, key: str) -> float:
    try:
        v = row.get(key, 0)
        return float(v) if v not in (None, "") else 0.0
    except (ValueError, TypeError):
        return 0.0


def _isum(rows, key) -> int:
    return int(sum(_f(r, key) for r in rows))


def _wavg(rows, rate_key, weight_key) -> float:
    """回収率を重み付き平均（combined = Σ(rate×weight)/Σweight）。
    rate% = 回収額/(weight×100)×100 = 回収額/weight なので、これがΣ回収額/Σ投資額と一致する。"""
    num = sum(_f(r, rate_key) * _f(r, weight_key) for r in rows)
    den = sum(_f(r, weight_key) for r in rows)
    return round(num / den, 1) if den > 0 else 0.0


def _load_daily_rows(sat_label: str, sun_label: str):
    """ai_daily_history.csv から土・日の行を取得。戻り値: (rows, found_dict)"""
    from huggingface_hub import hf_hub_download
    import pandas as pd
    path = hf_hub_download(HF_REPO_ID, "ai_daily_history.csv",
                           repo_type="dataset", token=HF_TOKEN)
    df = pd.read_csv(path, dtype=str)
    rows, found = [], {}
    for label in (sat_label, sun_label):
        m = df[df["日付"] == label]
        found[label] = not m.empty
        if not m.empty:
            rows.append(m.iloc[-1].to_dict())
    return rows, found


def build_message(rows, sat_label, sun_label, found) -> str:
    if not rows:
        return (f"📊 **keiba-ebye 週末まとめ** | {sat_label} & {sun_label}\n"
                f"日次振り返りデータがありません（auto_review が未実行の可能性）。")

    def _e(v):
        return "🔥" if v >= 150 else "✅" if v >= 100 else "🟡" if v >= 70 else "❌"

    races   = _isum(rows, "本命レース数")
    tan     = _wavg(rows, "本命単勝回収率", "本命レース数")
    fuku    = _wavg(rows, "本命複勝回収率", "本命レース数")
    tan_hi  = _isum(rows, "本命単勝的中数")
    fuku_hi = _isum(rows, "本命複勝的中数")

    buy_r     = _isum(rows, "買いレース数")
    miokuri_r = _isum(rows, "見送りレース数")
    buy_tan   = _wavg(rows, "買い本命単勝回収率", "買いレース数")
    buy_fuku  = _wavg(rows, "買い本命複勝回収率", "買いレース数")
    buy_tan_hi  = _isum(rows, "買い本命単勝的中数")
    buy_fuku_hi = _isum(rows, "買い本命複勝的中数")
    kachi_r   = _isum(rows, "勝負レース数")
    kachi_tan = _wavg(rows, "勝負本命単勝回収率", "勝負レース数")
    kachi_fuku = _wavg(rows, "勝負本命複勝回収率", "勝負レース数")

    uma      = _wavg(rows, "馬連回収率", "本命レース数")   # 馬連投資額は未保存のため本命R数で近似
    uma_hi   = _isum(rows, "馬連的中数")
    choko_n  = _isum(rows, "超狙い馬数")
    choko_t  = _wavg(rows, "超狙い馬単勝回収率", "超狙い馬数")
    choko_f  = _wavg(rows, "超狙い馬複勝回収率", "超狙い馬数")
    choko_th = _isum(rows, "超狙い馬単勝的中数")
    choko_fh = _isum(rows, "超狙い馬複勝的中数")
    ana_n    = _isum(rows, "穴馬数")
    ana_t    = _wavg(rows, "穴馬単勝回収率", "穴馬数")
    ana_f    = _wavg(rows, "穴馬複勝回収率", "穴馬数")
    ana_th   = _isum(rows, "穴馬単勝的中数")
    ana_fh   = _isum(rows, "穴馬複勝的中数")
    shiba_r  = _isum(rows, "芝レース数")
    dart_r   = _isum(rows, "ダートレース数")
    shiba    = _wavg(rows, "芝単勝回収率", "芝レース数")
    dart     = _wavg(rows, "ダート単勝回収率", "ダートレース数")

    missing = [lbl for lbl, ok in found.items() if not ok]

    lines = [
        f"📊 **keiba-ebye 週末まとめ** | {sat_label} & {sun_label}",
        f"対象 **{races}レース**（日次振り返りの合算）",
    ]
    if missing:
        lines.append(f"⚠️ {'・'.join(missing)} の日次データが無いため部分集計です")
    lines += [
        "",
        "**【本命(◎) 成績・全レース 2日合計】**",
        "```",
        f"単勝  {_e(tan)} {tan:6.1f}%   的中 {tan_hi}/{races}R",
        f"複勝  {_e(fuku)} {fuku:6.1f}%   的中 {fuku_hi}/{races}R",
        f"馬連  {_e(uma)} {uma:6.1f}%   的中 {uma_hi}回",
        "```",
        "",
        f"**【買うべきレース判定 2日合計】** 全{races}R → 🟢買い {buy_r}R / ⚠️見送り {miokuri_r}R",
        "```",
        f"買い◎単勝  {_e(buy_tan)} {buy_tan:6.1f}%   的中 {buy_tan_hi}/{buy_r}R",
        f"買い◎複勝  {_e(buy_fuku)} {buy_fuku:6.1f}%   的中 {buy_fuku_hi}/{buy_r}R",
        f"🔥勝負のみ  単{kachi_tan:.1f}% 複{kachi_fuku:.1f}%  ({kachi_r}R)",
        "```",
        "",
        "**【超狙い馬(AI上位5頭 EV1.5+) ベタ買い】**",
        "```",
        f"単勝  {_e(choko_t)} {choko_t:6.1f}%   的中 {choko_th}/{choko_n}頭",
        f"複勝  {_e(choko_f)} {choko_f:6.1f}%   的中 {choko_fh}/{choko_n}頭",
        "```",
        "",
        "**【穴馬(AI6位以下 EV1.5+) ベタ買い】**",
        "```",
        f"単勝  {_e(ana_t)} {ana_t:6.1f}%   的中 {ana_th}/{ana_n}頭",
        f"複勝  {_e(ana_f)} {ana_f:6.1f}%   的中 {ana_fh}/{ana_n}頭",
        "```",
        "",
        f"🌱 芝: {shiba:.1f}%  🏜️ ダート: {dart:.1f}%",
        "",
        "-# keiba-ebye 週末まとめ / 日次振り返りの合算・結果は参考情報です",
    ]
    return "\n".join(lines)


def run(sat_str: str = None, sun_str: str = None):
    if not sat_str or not sun_str:
        sat_str, sun_str = get_last_weekend()
    sat_label = datetime.datetime.strptime(sat_str, "%Y%m%d").strftime("%Y/%m/%d")
    sun_label = datetime.datetime.strptime(sun_str, "%Y%m%d").strftime("%Y/%m/%d")

    logger.info(f"週末まとめ（日次CSV集計）: {sat_label}（土）& {sun_label}（日）")
    try:
        rows, found = _load_daily_rows(sat_label, sun_label)
    except Exception as e:
        logger.error(f"ai_daily_history.csv 読み込み失敗: {e}")
        return
    logger.info(f"取得: 土={found.get(sat_label)} 日={found.get(sun_label)}")

    msg = build_message(rows, sat_label, sun_label, found)

    review_url = DISCORD_REVIEW_WEBHOOK_URL
    if not review_url:
        logger.error("DISCORD_WEBHOOK_URL / DISCORD_REVIEW_WEBHOOK_URL が未設定のため Discord 送信をスキップ")
        return
    try:
        resp = requests.post(
            review_url,
            json={"content": msg[:1990], "username": "keiba-ebye 📊週末"},
            timeout=15,
        )
        if resp.status_code in (200, 204):
            logger.info("Discord 送信成功")
        else:
            logger.error(f"Discord 送信失敗 HTTP {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        logger.error(f"Discord 送信エラー: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="土日まとめ振り返り（日次CSV集計）→ Discord")
    parser.add_argument("--sat", type=str, default=None, help="土曜日 YYYYMMDD")
    parser.add_argument("--sun", type=str, default=None, help="日曜日 YYYYMMDD")
    args = parser.parse_args()
    run(args.sat, args.sun)
