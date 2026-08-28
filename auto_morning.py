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
  GEMINI_API_KEY     - (任意) Google Gemini API キー（11R コメント生成）
"""

import os
import sys
import argparse
import datetime
import logging
import unittest.mock as mock
import json
import base64
import pytz
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("auto_morning")

HF_TOKEN            = os.environ.get("HF_TOKEN", "")
HF_REPO_ID          = os.environ.get("HF_REPO_ID", "")
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "").strip()
GEMINI_API_KEY      = os.environ.get("GEMINI_API_KEY", "")

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
    from src.reports import generate_pdf_report, generate_txt_report
    if GEMINI_API_KEY:
        from src.gemini_utils import generate_two_analysts

JST = pytz.timezone("Asia/Tokyo")


# ─────────────────────────────────────────────────────────────
# 馬券メモ読み込み
# ─────────────────────────────────────────────────────────────

def load_horse_memos() -> dict:
    """horse_memos.json をローカル優先→GitHub APIの順で読み込む"""
    memo_file = "horse_memos.json"
    if os.path.exists(memo_file):
        try:
            with open(memo_file, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"horse_memos.json読み込み失敗: {e}")
    gh_token = os.environ.get("GITHUB_TOKEN", "")
    gh_repo  = os.environ.get("GITHUB_REPO", "")
    if gh_token and gh_repo:
        try:
            headers = {
                "Authorization": f"token {gh_token}",
                "Accept": "application/vnd.github.v3+json",
            }
            resp = requests.get(
                f"https://api.github.com/repos/{gh_repo}/contents/{memo_file}",
                headers=headers, timeout=10,
            )
            if resp.status_code == 200:
                content = base64.b64decode(resp.json()["content"]).decode("utf-8")
                logger.info("horse_memos.json をGitHubから取得")
                return json.loads(content)
        except Exception as e:
            logger.warning(f"GitHub horse_memos.json取得失敗: {e}")
    return {}


# ─────────────────────────────────────────────────────────────
# フォーマット関数
# ─────────────────────────────────────────────────────────────

def _emoji_ev(ev: float) -> str:
    if ev >= 2.0: return "🔥"
    if ev >= 1.5: return "⭐"
    if ev >= 1.0: return "✅"
    return ""


def format_race_txt(race: dict, res_df, reco: str, confidence_text: str,
                    pace_text: str, topics: list = None,
                    all_memos: dict = None) -> str:
    """1レース分のテキストブロックを生成（全頭表示）"""
    time_str = race["time"].strftime("%H:%M")
    lines = [
        f"{'━'*52}",
        f"【{race['place']} {race['num']}R】{race['title']}  {time_str}発走",
        f"{'━'*52}",
        f"{'印':<2} {'馬番':>3} {'馬名':<12} {'オッズ':>6} {'勝率':>5} {'複勝率':>6} {'EV':>5} {'脚質':<5}",
        f"{'─'*52}",
    ]
    memo_lines = []
    for idx, row in res_df.iterrows():
        mark   = str(row.get("印", "")).ljust(2)
        uban   = int(row.get("馬番", 0))
        name   = str(row.get("馬名", ""))
        odds   = float(row.get("単勝オッズ", 0))
        prob   = float(row.get("勝率(AI予測)", 0)) * 100
        fprob  = float(row.get("複勝率(AI予測)", 0)) * 100
        ev     = float(row.get("期待値", 0) or 0)
        ana    = str(row.get("穴馬マーク", ""))
        style  = str(row.get("脚質カテゴリ", "")).replace("nan", "")
        ev_em  = _emoji_ev(ev)
        memo_flag = "📝" if all_memos and name in all_memos else ""
        lines.append(
            f"{mark} {uban:3d}番 {name:<12} "
            f"{odds:5.1f}倍 {prob:4.1f}% {fprob:5.1f}% "
            f"EV{ev:4.2f}{ev_em}{ana} {style}{memo_flag}"
        )
        # メモ内容をまとめておく
        if all_memos and name in all_memos:
            latest = sorted(all_memos[name], key=lambda x: x["日付"], reverse=True)[0]
            memo_lines.append(
                f"  📝 {name}({mark.strip()}): {latest['タグ']} {latest['日付']}"
                + (f" — {latest['メモ']}" if latest.get("メモ") else "")
            )
    lines.append("")
    if memo_lines:
        lines.append("【メモあり馬】")
        lines.extend(memo_lines)
        lines.append("")
    if confidence_text:
        lines.append(f"【判定】{confidence_text.replace('**','')}")
    if pace_text:
        lines.append(f"【展開】{pace_text.replace('**','')}")
    if reco:
        lines.append(f"【推奨】{reco.replace('**','').replace(chr(10),' ')[:180]}")
    # SHAP推し理由
    if topics:
        for t in topics:
            if "AIの推し理由" in t:
                lines.append(f"【AI推し理由】{t.replace(chr(10),' ')}")
                break
    lines.append("")
    return "\n".join(lines)


def _gemini_html_section(gemini_data: dict) -> str:
    """GeminiアナリストコメントのHTML断片"""
    if not gemini_data:
        return ""
    h = gemini_data.get("honmei", {})
    a = gemini_data.get("ana", {})
    model = gemini_data.get("model", "")
    return f"""
<div style="margin-top:10px;padding:10px;background:#f0f4ff;border:1px solid #b0c4ff;border-radius:6px">
  <div style="font-size:12px;font-weight:bold;color:#3a5bc7;margin-bottom:8px">🤖 AI思考モード（{model}）</div>
  <div style="display:flex;gap:10px">
    <div style="flex:1;padding:8px;background:#e8f4e8;border-left:3px solid #4caf50;border-radius:4px;font-size:12px">
      <div style="font-weight:bold;margin-bottom:4px">🎯 本命党「伊藤ホンメ」</div>
      <div>{h.get('comment','')}</div>
      <div style="margin-top:6px;font-size:11px;color:#555;font-weight:bold">💰 {h.get('bet','')}</div>
    </div>
    <div style="flex:1;padding:8px;background:#fff3e0;border-left:3px solid #ff9800;border-radius:4px;font-size:12px">
      <div style="font-weight:bold;margin-bottom:4px">💣 穴党「風穴あけるズ」</div>
      <div>{a.get('comment','')}</div>
      <div style="margin-top:6px;font-size:11px;color:#555;font-weight:bold">🎰 {a.get('bet','')}</div>
    </div>
  </div>
</div>"""


def format_race_html_row(race: dict, res_df, confidence_text: str,
                         topics: list = None, gemini_data: dict = None,
                         all_memos: dict = None) -> str:
    """1レース分のHTML<section>を生成（全頭表示・詳細情報付き）"""
    time_str = race["time"].strftime("%H:%M")
    rows_html = ""
    memo_html_items = []
    for idx, row in res_df.iterrows():
        mark   = str(row.get("印", ""))
        uban   = int(row.get("馬番", 0))
        name   = str(row.get("馬名", ""))
        odds   = float(row.get("単勝オッズ", 0))
        prob   = float(row.get("勝率(AI予測)", 0)) * 100
        fprob  = float(row.get("複勝率(AI予測)", 0)) * 100
        ev     = float(row.get("期待値", 0) or 0)
        ana    = str(row.get("穴馬マーク", ""))
        style  = str(row.get("脚質カテゴリ", "")).replace("nan", "")
        prev   = row.get("前走_着順", "")
        prev_s = f"{int(prev)}着" if str(prev) not in ("nan", "", "None") else "-"
        ev_em  = _emoji_ev(ev)
        has_memo = all_memos and name in all_memos
        # 背景色: メモあり馬は紫優先、その後EVによる色
        if has_memo:
            bg = "rgba(155,89,182,0.08)"
            row_border = "border-left:3px solid #9B59B6;"
        elif ev >= 1.5:
            bg, row_border = "#fff3f3", ""
        elif ev >= 1.0:
            bg, row_border = "#fffce8", ""
        else:
            bg, row_border = "white", ""
        # 印なし馬は薄めの文字色
        name_style = row_border + ("" if mark.strip() else "color:#999")
        memo_badge = ' <span style="font-size:11px">📝</span>' if has_memo else ""
        rows_html += (
            f'<tr style="background:{bg}">'
            f'<td style="font-size:16px">{mark}</td>'
            f'<td>{uban}</td>'
            f'<td style="text-align:left;{name_style}">{name}{memo_badge}</td>'
            f'<td>{style}</td>'
            f'<td>{odds:.1f}</td>'
            f'<td>{prob:.1f}%</td>'
            f'<td>{fprob:.1f}%</td>'
            f'<td><b>{ev:.2f}</b>{ev_em}</td>'
            f'<td>{prev_s}</td>'
            f'<td>{ana}</td>'
            f'</tr>\n'
        )
        # メモ内容を収集
        if has_memo:
            latest = sorted(all_memos[name], key=lambda x: x["日付"], reverse=True)[0]
            count  = len(all_memos[name])
            tag    = latest.get("タグ", "")
            date_  = latest.get("日付", "")
            memo_t = latest.get("メモ", "")
            count_s = f"（全{count}件）" if count > 1 else ""
            memo_html_items.append(
                f'<div style="padding:3px 8px;font-size:12px">'
                f'<b>{name}</b>（{mark.strip() or "印なし"}）{count_s} '
                f'<span style="color:#9b59b6">{tag}</span> '
                f'<span style="color:#888">{date_}</span>'
                + (f' — {memo_t}' if memo_t else '')
                + '</div>'
            )

    color = "#c0392b" if "鉄板" in confidence_text else "#2471a3" if "波乱" in confidence_text else "#117a65"

    # SHAP推し理由
    shap_html = ""
    if topics:
        for t in topics:
            if "AIの推し理由" in t:
                shap_html = f'<div style="padding:4px 12px;font-size:12px;color:#2471a3">🔍 {t.replace(chr(10),"　")}</div>'
                break

    # 馬券メモセクション
    memo_section_html = ""
    if memo_html_items:
        memo_section_html = (
            '<div style="margin:0;padding:6px 12px;background:#f5f0ff;'
            'border-top:1px solid #ddd;font-size:12px">'
            '<span style="color:#9b59b6;font-weight:bold">📝 馬券メモ</span>'
            + "".join(memo_html_items)
            + "</div>"
        )

    # Geminiコメント（11Rのみ）
    gemini_html = _gemini_html_section(gemini_data) if gemini_data else ""

    return f"""
<section style="margin:16px 0;border:1px solid #ddd;border-radius:8px;overflow:hidden">
  <div style="background:{color};color:white;padding:8px 12px;font-weight:bold">
    {race['place']} {race['num']}R　{race['title']}　{time_str}発走
  </div>
  <table style="width:100%;border-collapse:collapse;font-size:13px">
    <tr style="background:#f5f5f5;font-weight:bold">
      <th>印</th><th>馬番</th><th style="text-align:left">馬名</th><th>脚質</th>
      <th>オッズ</th><th>勝率</th><th>複勝率</th><th>EV</th><th>前走</th><th></th>
    </tr>
    {rows_html}
  </table>
  <div style="padding:6px 12px;font-size:13px;color:#555">{confidence_text.replace('**','')}</div>
  {shap_html}
  {memo_section_html}
  <div style="padding:6px 12px 10px;font-size:12px;color:#333;white-space:pre-wrap">{gemini_html}</div>
</section>"""


def build_full_html(date_label: str, sections: list) -> str:
    body = "\n".join(sections)
    return f"""<!DOCTYPE html>
<html lang="ja"><head>
<meta charset="utf-8">
<title>keiba-ebye 予想 {date_label}</title>
<style>
  body {{ font-family: 'Hiragino Kaku Gothic Pro', Meiryo, sans-serif; max-width:960px; margin:auto; padding:16px; }}
  h1 {{ color:#2c3e50; border-bottom:3px solid #c0392b; padding-bottom:8px; }}
  td,th {{ padding:5px 8px; border-bottom:1px solid #eee; text-align:center; }}
</style>
</head><body>
<h1>🐴 keiba-ebye AI予想　{date_label}</h1>
<p style="color:#888;font-size:13px">⚠️ 馬券の購入は自己責任でお願いします。オッズは8時時点の参考値です。</p>
<p style="font-size:12px;color:#555">
  EV=AI勝率×オッズ（✅=1.0以上・⭐=1.5以上・🔥=2.0以上）　🎯=穴馬マーク<br>
  印なし馬も含む全頭表示。EV1.5以上の行はピンク背景、1.0以上は黄背景。
</p>
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

    # 馬券メモ読み込み（ローカルまたはGitHub）
    all_memos = load_horse_memos()
    logger.info(f"馬券メモ読み込み: {len(all_memos)}頭分")

    # 全レース推論
    results_list = []
    ok_count = 0
    venues = sorted(set(r["place"] for r in races))

    for r in races:
        logger.info(f"  推論: {r['place']} {r['num']}R ({r['id']})")
        try:
            res_df, topics_list, reco, pace_text, conf_text, track_type, _, distance, err = run_real_prediction(
                r["id"], date_hf, bundle,
                skip_live_scrape=False,
                ev_first=True, ev_threshold=1.5, min_win_prob=0.18,
            )
        except Exception as e:
            logger.warning(f"  推論失敗: {e}")
            continue
        if res_df is None:
            logger.warning(f"  推論結果なし: {r['id']}")
            continue

        # 11R のみ Gemini アナリストコメントを生成
        gemini_data = None
        if r["num"] == 11 and GEMINI_API_KEY:
            logger.info(f"  Gemini生成中: {r['place']} 11R")
            try:
                gemini_data = generate_two_analysts(
                    res_df, pace_text or "", conf_text or "",
                    topics_list or [], reco or "",
                )
            except Exception as ge:
                logger.warning(f"  Gemini生成失敗: {ge}")

        ritem = {
            "df":         res_df,
            "place":      r["place"],
            "num":        r["num"],
            "title":      r.get("title", ""),
            "time":       r["time"].strftime("%H:%M") if r.get("time") else "",
            "track":      track_type or "",
            "dist":       distance or "",
            "confidence": conf_text or "",
            "pace":       pace_text or "",
            "topics":     topics_list or [],
            "reco":       reco or "",
            "date":       date_hf,
        }
        if gemini_data:
            ritem["gemini_honmei"] = gemini_data.get("honmei", {})
            ritem["gemini_ana"]    = gemini_data.get("ana", {})
            ritem["gemini_model"]  = gemini_data.get("model", "")
        results_list.append(ritem)
        ok_count += 1

    if ok_count == 0:
        logger.error("全レースで推論失敗。投稿スキップ。")
        return

    # src/reports.py のフォーマットで HTML / TXT 生成
    html_bytes = generate_pdf_report(results_list, ev_threshold=1.5, all_memos=all_memos)
    full_html  = html_bytes.decode("utf-8") if html_bytes else ""
    full_txt   = generate_txt_report(results_list, ev_threshold=1.5, all_memos=all_memos)

    # Discord サマリーメッセージ（ファイルと一緒に投稿）
    gemini_note = "🤖 11R Gemini AIコメント付き" if GEMINI_API_KEY else ""
    summary = (
        f"🐴 **keiba-ebye AI朝刊予想** | {date_label}\n"
        f"開催: **{' / '.join(venues)}** 全**{ok_count}**レース\n"
        f"▼ 本日の全レース予想を .txt / .html で添付しました（全頭表示）\n"
        f"⭐=EV1.5以上  🔥=EV2.0以上  🎯=穴馬マーク  {gemini_note}\n"
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
