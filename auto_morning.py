"""
朝8時 全レース予想スクリプト（GitHub Actions から呼び出し）
- 当日の全開催レースを予想（EV優先）
- 結果を .txt と .html にフォーマットして Discord に直接投稿（ファイル添付）
- キュー経由でなく Webhook 直接送信（GitHub Actions は Discord への通信が可能）

使い方:
  python auto_morning.py [--date YYYYMMDD]

必要な環境変数:
  HF_TOKEN           - HuggingFace API トークン（モデルロード用）
  HF_REPO_ID         - モデル保存先 Dataset リポジトリ ID
  DISCORD_WEBHOOK_URL - 投稿先 Discord Webhook URL
"""

import os
import sys
import argparse
import datetime
import logging
import unittest.mock as mock
import json
import pytz
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("auto_morning")

HF_TOKEN            = os.environ.get("HF_TOKEN", "")
HF_REPO_ID          = os.environ.get("HF_REPO_ID", "")
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "").strip()

if not HF_TOKEN or not HF_REPO_ID:
    logger.error("HF_TOKEN / HF_REPO_ID が未設定です。")
    sys.exit(1)
if not DISCORD_WEBHOOK_URL:
    logger.error("DISCORD_WEBHOOK_URL が未設定です。")
    sys.exit(1)

# ── Streamlit デコレータをモックして src モジュールをインポート ──
def _passthrough(func=None, **kw):
    if callable(func): return func
    return lambda f: f

with mock.patch("streamlit.cache_resource", _passthrough), \
     mock.patch("streamlit.cache_data",     _passthrough), \
     mock.patch("streamlit.spinner",        lambda *a, **kw: mock.MagicMock()):
    from src.core_model import prepare_model_and_data
    from src.scraper import get_todays_races
    from src.inference import run_real_prediction

JST = pytz.timezone("Asia/Tokyo")


# ─────────────────────────────────────────────────────────────
# フォーマット関数
# ─────────────────────────────────────────────────────────────

def _emoji_ev(ev: float) -> str:
    if ev >= 2.0: return "🔥"
    if ev >= 1.5: return "⭐"
    if ev >= 1.0: return "✅"
    return ""


def format_race_txt(race: dict, res_df, reco: str, confidence_text: str, pace_text: str) -> str:
    """1レース分のテキストブロックを生成"""
    time_str = race["time"].strftime("%H:%M")
    lines = [
        f"{'━'*50}",
        f"【{race['place']} {race['num']}R】{race['title']}  {time_str}発走",
        f"{'━'*50}",
    ]
    for _, row in res_df.iterrows():
        mark   = str(row.get("印", "")).ljust(2)
        if not mark.strip():
            continue
        uban   = int(row.get("馬番", 0))
        name   = str(row.get("馬名", ""))
        odds   = float(row.get("単勝オッズ", 0))
        prob   = float(row.get("勝率(AI予測)", 0)) * 100
        ev     = float(row.get("期待値", 0) or 0)
        ana    = str(row.get("穴馬マーク", ""))
        ev_em  = _emoji_ev(ev)
        lines.append(
            f"{mark} {uban:2d}番 {name:<12} "
            f"オッズ{odds:5.1f}倍  勝率{prob:4.1f}%  EV{ev:4.2f} {ev_em}{ana}"
        )
    lines.append("")
    if confidence_text:
        lines.append(f"【判定】{confidence_text.replace('**','')}")
    if pace_text:
        lines.append(f"【展開】{pace_text.replace('**','')}")
    if reco:
        lines.append(f"【推奨】{reco.replace('**','').replace(chr(10),' ')[:120]}")
    lines.append("")
    return "\n".join(lines)


def format_race_html_row(race: dict, res_df, confidence_text: str) -> str:
    """1レース分のHTML<section>を生成"""
    time_str = race["time"].strftime("%H:%M")
    rows_html = ""
    for _, row in res_df.iterrows():
        mark = str(row.get("印", ""))
        if not mark.strip(): continue
        uban = int(row.get("馬番", 0))
        name = str(row.get("馬名", ""))
        odds = float(row.get("単勝オッズ", 0))
        prob = float(row.get("勝率(AI予測)", 0)) * 100
        ev   = float(row.get("期待値", 0) or 0)
        ana  = str(row.get("穴馬マーク", ""))
        bg   = "#fff3f3" if ev >= 1.5 else "#fffce8" if ev >= 1.0 else "white"
        rows_html += (
            f'<tr style="background:{bg}">'
            f'<td>{mark}</td><td>{uban}</td><td>{name}</td>'
            f'<td>{odds:.1f}</td><td>{prob:.1f}%</td>'
            f'<td><b>{ev:.2f}</b></td><td>{ana}</td></tr>\n'
        )
    color = "#c0392b" if "鉄板" in confidence_text else "#2471a3" if "波乱" in confidence_text else "#117a65"
    return f"""
<section style="margin:16px 0;border:1px solid #ddd;border-radius:8px;overflow:hidden">
  <div style="background:{color};color:white;padding:8px 12px;font-weight:bold">
    {race['place']} {race['num']}R　{race['title']}　{time_str}発走
  </div>
  <table style="width:100%;border-collapse:collapse;font-size:14px">
    <tr style="background:#f5f5f5;font-weight:bold">
      <th>印</th><th>馬番</th><th>馬名</th><th>オッズ</th><th>勝率</th><th>EV</th><th></th>
    </tr>
    {rows_html}
  </table>
  <div style="padding:6px 12px;font-size:13px;color:#555">{confidence_text.replace('**','')}</div>
</section>"""


def build_full_html(date_label: str, sections: list) -> str:
    body = "\n".join(sections)
    return f"""<!DOCTYPE html>
<html lang="ja"><head>
<meta charset="utf-8">
<title>keiba-ebye 予想 {date_label}</title>
<style>
  body {{ font-family: 'Hiragino Kaku Gothic Pro', Meiryo, sans-serif; max-width:900px; margin:auto; padding:16px; }}
  h1 {{ color:#2c3e50; border-bottom:3px solid #c0392b; padding-bottom:8px; }}
  td,th {{ padding:6px 10px; border-bottom:1px solid #eee; text-align:center; }}
</style>
</head><body>
<h1>🐴 keiba-ebye AI予想　{date_label}</h1>
<p style="color:#888;font-size:13px">⚠️ 馬券の購入は自己責任でお願いします。オッズは8時時点の参考値です。</p>
{body}
<hr><p style="color:#aaa;font-size:12px">keiba-ebye 自動生成 | EV=AI勝率×オッズ（1.0超が購入検討ライン）</p>
</body></html>"""


# ─────────────────────────────────────────────────────────────
# Discord への直接投稿（ファイル添付）
# ─────────────────────────────────────────────────────────────

def post_files_to_discord(webhook_url: str, summary_msg: str,
                           txt_content: str, html_content: str,
                           date_label: str) -> bool:
    """ファイルを Discord Webhook に直接 POST する（キュー経由ではない）"""
    try:
        files = {
            "files[0]": (
                f"keiba_{date_label.replace('/','')}.txt",
                txt_content.encode("utf-8"),
                "text/plain; charset=utf-8",
            ),
            "files[1]": (
                f"keiba_{date_label.replace('/','')}.html",
                html_content.encode("utf-8"),
                "text/html; charset=utf-8",
            ),
        }
        payload = {
            "payload_json": json.dumps({"content": summary_msg}, ensure_ascii=False)
        }
        resp = requests.post(webhook_url, data=payload, files=files, timeout=30)
        if resp.status_code in (200, 204):
            logger.info(f"Discord 投稿成功 HTTP {resp.status_code}")
            return True
        logger.warning(f"Discord 投稿失敗 HTTP {resp.status_code}: {resp.text[:200]}")
        return False
    except Exception as e:
        logger.error(f"Discord 投稿エラー: {e}")
        return False


def post_text_to_discord(webhook_url: str, content: str) -> bool:
    """テキストメッセージを Discord に直接送信"""
    try:
        resp = requests.post(webhook_url, json={"content": content[:1990]}, timeout=15)
        return resp.status_code in (200, 204)
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────
# メイン処理
# ─────────────────────────────────────────────────────────────

def run(date_str: str = None):
    now = datetime.datetime.now(JST)
    target_dt = datetime.datetime.strptime(date_str, "%Y%m%d").replace(tzinfo=JST) if date_str else now
    date_label = target_dt.strftime("%Y/%m/%d")
    date_str8  = target_dt.strftime("%Y%m%d")
    date_hf    = target_dt.strftime("%Y-%m-%d")

    logger.info(f"朝刊予想 対象日: {date_label}")

    races = get_todays_races(date_str8)
    if not races:
        logger.info("本日の開催なし。終了。")
        post_text_to_discord(
            DISCORD_WEBHOOK_URL,
            f"🐴 **keiba-ebye** {date_label} — 本日はJRAの開催がありません。"
        )
        return

    logger.info(f"{len(races)} レース取得。モデルロード中...")
    bundle = prepare_model_and_data(force_retrain=False)
    logger.info("モデルロード完了。推論開始...")

    # 全レース推論
    txt_blocks   = []
    html_sections = []
    ok_count = 0
    venues = sorted(set(r["place"] for r in races))

    for r in races:
        logger.info(f"  推論: {r['place']} {r['num']}R ({r['id']})")
        try:
            res_df, _, reco, pace_text, conf_text, _, _, _, err = run_real_prediction(
                r["id"], date_hf, bundle,
                skip_live_scrape=False,
                ev_first=True, ev_threshold=1.0, min_win_prob=0.10,
            )
        except Exception as e:
            logger.warning(f"  推論失敗: {e}")
            continue
        if res_df is None:
            logger.warning(f"  推論結果なし: {r['id']}")
            continue

        txt_blocks.append(format_race_txt(r, res_df, reco or "", conf_text or "", pace_text or ""))
        html_sections.append(format_race_html_row(r, res_df, conf_text or ""))
        ok_count += 1

    if ok_count == 0:
        logger.error("全レースで推論失敗。投稿スキップ。")
        return

    # テキスト全体
    header_txt = (
        f"keiba-ebye AI朝刊予想  {date_label}\n"
        f"開催: {' / '.join(venues)}  全{ok_count}レース\n"
        f"{'='*50}\n"
        f"EV=AI勝率×オッズ（1.0超が購入検討ライン）\n"
        f"{'='*50}\n\n"
    )
    full_txt  = header_txt + "\n".join(txt_blocks)
    full_html = build_full_html(date_label, html_sections)

    # Discord サマリーメッセージ（ファイルと一緒に投稿）
    summary = (
        f"🐴 **keiba-ebye AI朝刊予想** | {date_label}\n"
        f"開催: **{' / '.join(venues)}** 全**{ok_count}**レース\n"
        f"▼ 本日の全レース予想を .txt / .html で添付しました\n"
        f"⭐=EV1.5以上  🔥=EV2.0以上  🎯=穴馬マーク\n"
        f"-# keiba-ebye 自動予想 / 馬券は自己責任でお願いします"
    )

    ok = post_files_to_discord(DISCORD_WEBHOOK_URL, summary, full_txt, full_html, date_label)
    if ok:
        logger.info(f"Discord投稿完了 ({ok_count}レース分)")
    else:
        # ファイル投稿失敗時はテキストのみ送信
        logger.warning("ファイル投稿失敗。テキストのみ送信します。")
        post_text_to_discord(DISCORD_WEBHOOK_URL, summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="朝8時 全レース予想 → Discord")
    parser.add_argument("--date", type=str, default=None, help="対象日 YYYYMMDD（省略時は本日）")
    args = parser.parse_args()
    run(args.date)
