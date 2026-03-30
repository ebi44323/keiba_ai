import os
import io
import json
import datetime
import logging
import requests

logger = logging.getLogger('keiba_ebye')

_HF_TOKEN   = os.environ.get("HF_TOKEN", "")
_HF_REPO_ID = os.environ.get("HF_REPO_ID", "")
_DISCORD_WEBHOOK_URL        = os.environ.get("DISCORD_WEBHOOK_URL", "").strip()
_DISCORD_REVIEW_WEBHOOK_URL = (os.environ.get("DISCORD_REVIEW_WEBHOOK_URL", "").strip()) or _DISCORD_WEBHOOK_URL

_DISCORD_QUEUE_FILE = "discord_queue.json"


# ==========================================
# HF Hub キュー書き込み
# Cloudflare Workers cron（毎分）が読み取って Discord に送信する
# ==========================================
def _push_discord_queue(content: str, channel: str = "prediction",
                        username: str = "keiba-ebye 🐴",
                        dedup_key: str = "") -> bool:
    """
    HF Hub の discord_queue.json にメッセージを追加する。
    Cloudflare Workers の cron trigger（毎分実行）が読み取って Discord に送信する。
    dedup_key: 重複防止キー（同一キーが未送信で存在する場合はスキップ）
    """
    if not _HF_TOKEN or not _HF_REPO_ID:
        logger.warning("HF_TOKEN/HF_REPO_IDが未設定のためDiscordキューに書き込めません")
        return False
    try:
        from huggingface_hub import HfApi, hf_hub_download
        api = HfApi(token=_HF_TOKEN)

        # 既存キューを取得
        queue = []
        try:
            qpath = hf_hub_download(
                repo_id=_HF_REPO_ID, filename=_DISCORD_QUEUE_FILE,
                repo_type="dataset", token=_HF_TOKEN, cache_dir="/tmp/hf_cache",
                force_download=True
            )
            with open(qpath, "r", encoding="utf-8") as f:
                queue = json.load(f)
        except Exception:
            queue = []

        # 重複防止チェック
        if dedup_key:
            already = any(
                q.get("dedup_key") == dedup_key and not q.get("sent", False)
                for q in queue
            )
            if already:
                logger.info(f"Discord重複スキップ: {dedup_key}")
                return True

        # 新規エントリを追加
        entry = {
            "id": f"{channel}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}",
            "dedup_key": dedup_key or "",
            "channel": channel,
            "content": content[:1990],
            "username": username,
            "timestamp": datetime.datetime.now().isoformat(),
            "sent": False,
        }
        queue.append(entry)

        # 古いエントリを整理（未送信50件・送信済み20件）
        queue = [q for q in queue if not q.get("sent", False)][-50:] + \
                [q for q in queue if q.get("sent", False)][-20:]

        # HF Hub に保存
        buf = io.BytesIO(json.dumps(queue, ensure_ascii=False, indent=2).encode("utf-8"))
        api.upload_file(
            path_or_fileobj=buf,
            path_in_repo=_DISCORD_QUEUE_FILE,
            repo_id=_HF_REPO_ID,
            repo_type="dataset",
            commit_message=f"Discord queue: {channel} {entry['id']}",
            token=_HF_TOKEN,
        )
        logger.info(f"Discord送信キューに追加: {entry['id']}")
        return True
    except Exception as _e:
        logger.warning(f"Discord送信キュー書き込み失敗: {_e}")
        return False


def send_discord_prediction(res_df, topics, reco, pace_text, conf_text,
                             race_info: dict, webhook_url: str = "",
                             gemini_data: dict = None) -> bool:
    """予想結果を Discord キュー経由で送信する（Cloudflare Workers cron が配送）"""
    try:
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
                lines.append(f"{imp:<3} {num_:>3} {name:<12} {odds:>5.1f}倍 {wp:>5.1f}% {ev:>4.2f}{ev_mark}")
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
        race_id = race_info.get('race_id', '')
        ok = _push_discord_queue(content, channel="prediction",
                                  username="keiba-ebye 🐴", dedup_key=race_id)

        # AIアナリストコメントがあれば続けて別メッセージで投稿
        if ok and gemini_data and isinstance(gemini_data, dict):
            honmei = gemini_data.get('honmei', {})
            ana    = gemini_data.get('ana', {})
            model  = gemini_data.get('model', '')
            if honmei or ana:
                alines = [
                    f"🤖 **AIアナリスト解説** ({model}) | {place} {num}R",
                    "",
                ]
                if honmei.get('comment'):
                    alines.append(f"🎯 **伊藤ホンメ（本命党）**")
                    alines.append(f"> {honmei['comment'][:200]}")
                    if honmei.get('bet'):
                        alines.append(f"> 💰 {honmei['bet'][:80]}")
                    alines.append("")
                if ana.get('comment'):
                    alines.append(f"💣 **風穴あけるズ（穴党）**")
                    alines.append(f"> {ana['comment'][:200]}")
                    if ana.get('bet'):
                        alines.append(f"> 🎰 {ana['bet'][:80]}")
                alines.append("")
                alines.append("-# keiba-ebye AI思考モード")
                _push_discord_queue(
                    "\n".join(alines), channel="prediction",
                    username="keiba-ebye 🤖",
                    dedup_key=f"{race_id}_gemini",
                )
        return ok
    except Exception as _e:
        logger.warning(f"send_discord_prediction エラー: {_e}")
        return False


def send_discord_review(stats: dict, rates: dict, target_date_str: str,
                        webhook_url: str = "") -> bool:
    """振り返り結果を Discord キュー経由で送信する（Cloudflare Workers cron が配送）"""
    try:
        tan_rate      = rates.get('tan_rate', 0)
        fuku_rate     = rates.get('fuku_rate', 0)
        choko_tan     = rates.get('choko_tan_rate', 0)
        choko_fuku    = rates.get('choko_fuku_rate', 0)
        ana_tan       = rates.get('ana_tan_rate', 0)
        ana_fuku      = rates.get('ana_fuku_rate', 0)
        uma_rate      = rates.get('uma_rate', 0)
        shiba_rate    = rates.get('shiba_rate', 0)
        dart_rate     = rates.get('dart_rate', 0)
        races         = stats.get('honmei_races', 0)

        def _emoji(v):
            if v >= 150: return "🔥"
            if v >= 100: return "✅"
            if v >= 70:  return "🟡"
            return "❌"

        lines = [
            f"📊 **keiba-ebye 振り返りレポート** | {target_date_str}",
            f"対象 {races}レース",
            "",
            "**【本命(◎) 成績】**",
            "```",
            f"単勝  {_emoji(tan_rate)} {tan_rate:6.1f}%   的中 {stats.get('honmei_tan_hits',0)}/{races}R",
            f"複勝  {_emoji(fuku_rate)} {fuku_rate:6.1f}%   的中 {stats.get('honmei_fuku_hits',0)}/{races}R",
            "```",
            "",
            "**【超狙い馬(AI上位5頭 EV1.5+) ベタ買い】**",
            "```",
            f"単勝  {_emoji(choko_tan)} {choko_tan:6.1f}%   的中 {stats.get('choko_tan_hits',0)}/{int(stats.get('choko_invest',0)//100)}頭",
            f"複勝  {_emoji(choko_fuku)} {choko_fuku:6.1f}%   的中 {stats.get('choko_fuku_hits',0)}/{int(stats.get('choko_invest',0)//100)}頭",
            "```",
            "",
            "**【穴馬(AI6位以下 EV1.5+) ベタ買い】**",
            "```",
            f"単勝  {_emoji(ana_tan)} {ana_tan:6.1f}%   的中 {stats.get('ana_tan_hits',0)}/{int(stats.get('ana_invest',0)//100)}頭",
            f"複勝  {_emoji(ana_fuku)} {ana_fuku:6.1f}%   的中 {stats.get('ana_fuku_hits',0)}/{int(stats.get('ana_invest',0)//100)}頭",
            "```",
            "",
            f"🌱 芝: {shiba_rate:.1f}%  🏜️ ダート: {dart_rate:.1f}%  🔗 馬連: {uma_rate:.1f}%",
            "",
            "-# keiba-ebye / 結果は参考情報です",
        ]
        content = "\n".join(lines)
        return _push_discord_queue(content, channel="review",
                                   username="keiba-ebye 📊", dedup_key=target_date_str)
    except Exception as _e:
        logger.warning(f"send_discord_review エラー: {_e}")
        return False


def _test_discord_webhook(webhook_url: str, label: str = "テスト") -> tuple[bool, str]:
    """テストメッセージを Discord キューに追加する"""
    channel = "review" if label == "振り返り" else "prediction"
    msg = f"🔌 **keiba-ebye** 接続テスト！ ({label}チャンネル) — Cloudflare Workers cron が1分以内に配送します ✅"
    ok = _push_discord_queue(msg, channel=channel, username="keiba-ebye 🔌")
    if ok:
        return True, "✅ キューに追加しました！1分以内に Discord に届きます"
    return False, "❌ キュー書き込み失敗（HF_TOKEN / HF_REPO_ID を確認してください）"
