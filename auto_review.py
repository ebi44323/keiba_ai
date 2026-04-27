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
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("auto_review")

HF_TOKEN                = os.environ.get("HF_TOKEN", "")
HF_REPO_ID              = os.environ.get("HF_REPO_ID", "")
DISCORD_WEBHOOK_URL     = os.environ.get("DISCORD_WEBHOOK_URL", "").strip()
DISCORD_REVIEW_WEBHOOK_URL = (os.environ.get("DISCORD_REVIEW_WEBHOOK_URL", "").strip()) or DISCORD_WEBHOOK_URL

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
    from src.utils import classify_race_class

JST = pytz.timezone("Asia/Tokyo")


def _send_review_direct(stats: dict, rates: dict, date_label: str) -> bool:
    """振り返り結果を Discord Webhook に直接送信する（GitHub Actions 用）"""
    webhook_url = DISCORD_REVIEW_WEBHOOK_URL
    if not webhook_url:
        logger.error("DISCORD_WEBHOOK_URL / DISCORD_REVIEW_WEBHOOK_URL が未設定のため Discord 送信をスキップ")
        return False

    def _emoji(v):
        if v >= 150: return "🔥"
        if v >= 100: return "✅"
        if v >= 70:  return "🟡"
        return "❌"

    tan_rate   = rates.get('tan_rate', 0)
    fuku_rate  = rates.get('fuku_rate', 0)
    choko_tan  = rates.get('choko_tan_rate', 0)
    choko_fuku = rates.get('choko_fuku_rate', 0)
    ana_tan    = rates.get('ana_tan_rate', 0)
    ana_fuku   = rates.get('ana_fuku_rate', 0)
    uma_rate   = rates.get('uma_rate', 0)
    shiba_rate = rates.get('shiba_rate', 0)
    dart_rate  = rates.get('dart_rate', 0)
    races      = stats.get('honmei_races', 0)

    lines = [
        f"📊 **keiba-ebye 振り返りレポート** | {date_label}",
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
        f"🌱 芝: {shiba_rate:.1f}%  🏜️ ダート: {dart_rate:.1f}%",
        f"🔗 馬連: {uma_rate:.1f}%  📐 穴馬ワイド: {rates.get('wide_rate', 0):.1f}%",
        "",
        "-# keiba-ebye / 結果は参考情報です",
    ]
    content = "\n".join(lines)
    try:
        resp = requests.post(
            webhook_url,
            json={"content": content[:1990], "username": "keiba-ebye 📊"},
            timeout=15,
        )
        if resp.status_code in (200, 204):
            return True
        logger.warning(f"Discord送信失敗 HTTP {resp.status_code}: {resp.text[:200]}")
        return False
    except Exception as e:
        logger.error(f"Discord送信エラー: {e}")
        return False


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

    _ds_init = lambda: {"R数": 0, "的中": 0, "単勝回収": 0}
    stats = {
        # 本命◎
        "honmei_races": 0, "honmei_tan_hits": 0, "honmei_tan_return": 0,
        "honmei_fuku_hits": 0, "honmei_fuku_return": 0,
        # 馬連
        "umaren_races": 0, "umaren_invest": 0, "umaren_hits": 0, "umaren_return": 0,
        # 超狙い馬: AI上位5頭(index<5) かつ EV>=1.5
        "choko_invest": 0, "choko_tan_hits": 0, "choko_tan_return": 0,
        "choko_fuku_hits": 0, "choko_fuku_return": 0,
        # 穴馬: AI6位以下(index>=5) かつ EV>=1.5
        "ana_invest": 0, "ana_tan_hits": 0, "ana_tan_return": 0,
        "ana_fuku_hits": 0, "ana_fuku_return": 0,
        # 穴馬ワイド流し
        "wide_ana_races": 0, "wide_ana_invest": 0, "wide_ana_hits": 0, "wide_ana_return": 0,
        # 三連複◎〇▲ボックス
        "sanrenpuku_invest": 0, "sanrenpuku_hits": 0, "sanrenpuku_return": 0,
        # 芝/ダート
        "shiba_races": 0, "shiba_return": 0, "dart_races": 0, "dart_return": 0,
        # Calibration用リスト
        "honmei_ai_probs": [],   # ◎のAI勝率
        "winner_ai_probs": [],   # 実際の勝者のAI勝率
        # 競馬場別 / 距離帯別 / クラス別（dict）
        "venue_stats": {},
        "dist_stats": {"短距離": _ds_init(), "マイル": _ds_init(),
                       "中距離": _ds_init(), "長距離": _ds_init()},
        "class_stats": {"低クラス": _ds_init(), "高クラス": _ds_init()},
    }

    for r in races:
        try:
            res_df, _, _, _, _, track_type, place, dist, err_log = run_real_prediction(
                r["id"], date_hf, bundle,
                skip_live_scrape=True,  # 振り返りは高速モード（前走データは取得しない）
                ev_first=True,
                ev_threshold=1.5,
                min_win_prob=0.18,
                use_oikiri=True,  # 当日振り返り: 直近レースの調教データは残っているため取得
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
        honmei_ai_prob = float(res_df.iloc[0]["勝率(AI予測)"])
        stats["honmei_races"] += 1
        stats["honmei_ai_probs"].append(honmei_ai_prob)

        # 実際の勝者のAI勝率（Calibration）
        winner_nums = list(payouts["tansho"].keys())
        if winner_nums:
            winner_row = res_df[res_df["馬番"] == winner_nums[0]]
            if not winner_row.empty:
                stats["winner_ai_probs"].append(float(winner_row.iloc[0]["勝率(AI予測)"]))

        # 芝/ダート
        if track_type == "芝":
            stats["shiba_races"] += 1
        elif track_type == "ダート":
            stats["dart_races"] += 1

        # 競馬場別
        venue = place or r.get("place", "不明")
        vs = stats["venue_stats"].setdefault(venue, {"R数": 0, "的中": 0, "単勝回収": 0})
        vs["R数"] += 1

        # 距離帯別
        try:
            dist_val = int(dist) if dist else 0
        except (ValueError, TypeError):
            dist_val = 0
        if dist_val <= 1400:   dist_key = "短距離"
        elif dist_val <= 1800: dist_key = "マイル"
        elif dist_val <= 2200: dist_key = "中距離"
        else:                  dist_key = "長距離"
        ds = stats["dist_stats"][dist_key]
        ds["R数"] += 1

        # クラス別
        race_class_code = float(res_df.iloc[0].get("レースクラスコード", 5)) \
            if "レースクラスコード" in res_df.columns else float(classify_race_class(r.get("title", "")))
        class_key = "低クラス" if race_class_code <= 2 else "高クラス"
        cs = stats["class_stats"][class_key]
        cs["R数"] += 1

        # 本命◎ 結果
        if honmei in payouts["tansho"]:
            stats["honmei_tan_hits"] += 1
            pay = payouts["tansho"][honmei]
            stats["honmei_tan_return"] += pay
            if track_type == "芝":   stats["shiba_return"] += pay
            elif track_type == "ダート": stats["dart_return"] += pay
            vs["的中"] += 1; vs["単勝回収"] += pay
            ds["的中"] += 1; ds["単勝回収"] += pay
            cs["的中"] += 1; cs["単勝回収"] += pay

        if honmei in payouts["fukusho"]:
            stats["honmei_fuku_hits"] += 1
            stats["honmei_fuku_return"] += payouts["fukusho"][honmei]

        # 馬連（◎ → 〇〜☆流し）
        if len(res_df) >= 5:
            himo_list = res_df.iloc[1:5]["馬番"].tolist()
            stats["umaren_races"] += 1
            stats["umaren_invest"] += len(himo_list) * 100
            for himo in himo_list:
                key = tuple(sorted([honmei, himo]))
                if key in payouts.get("umaren", {}):
                    stats["umaren_hits"] += 1
                    stats["umaren_return"] += payouts["umaren"][key]

        # 三連複 ◎ → 2〜5位ながし (◎軸1頭 × 相手4頭からC(4,2)=6点)
        if len(res_df) >= 5:
            honmei_num3 = res_df.iloc[0]["馬番"]
            himo4 = res_df.iloc[1:5]["馬番"].tolist()
            for _i in range(len(himo4)):
                for _j in range(_i + 1, len(himo4)):
                    key3 = tuple(sorted([honmei_num3, himo4[_i], himo4[_j]]))
                    stats["sanrenpuku_invest"] += 100
                    if key3 in payouts.get("sanrenpuku", {}):
                        stats["sanrenpuku_hits"] += 1
                        stats["sanrenpuku_return"] += payouts["sanrenpuku"][key3]

        # 穴馬ワイド流し: ◎ → AI6位以下(index>=5) かつ EV>=1.5
        ana_list = res_df[(res_df.index >= 5) & (res_df["期待値"] >= 1.5)]["馬番"].tolist()
        if ana_list:
            stats["wide_ana_races"] += 1
            stats["wide_ana_invest"] += len(ana_list) * 100
            for ana in ana_list:
                key = tuple(sorted([honmei, ana]))
                if key in payouts.get("wide", {}):
                    stats["wide_ana_hits"] += 1
                    stats["wide_ana_return"] += payouts["wide"][key]

        # 超狙い馬: AI上位5頭(index<5) かつ EV>=1.5
        for _, row in res_df[(res_df.index < 5) & (res_df["期待値"] >= 1.5)].iterrows():
            uban = row["馬番"]
            stats["choko_invest"] += 100
            if uban in payouts["tansho"]:
                stats["choko_tan_hits"] += 1
                stats["choko_tan_return"] += payouts["tansho"][uban]
            if uban in payouts["fukusho"]:
                stats["choko_fuku_hits"] += 1
                stats["choko_fuku_return"] += payouts["fukusho"][uban]

        # 穴馬ベタ買い: AI6位以下(index>=5) かつ EV>=1.5
        for _, row in res_df[(res_df.index >= 5) & (res_df["期待値"] >= 1.5)].iterrows():
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

    def _dist_rate(key):
        ds = stats["dist_stats"][key]
        return _rate(ds["単勝回収"], max(ds["R数"] * 100, 1))
    def _class_rate(key):
        cs = stats["class_stats"][key]
        return _rate(cs["単勝回収"], max(cs["R数"] * 100, 1))

    rates = {
        "tan_rate":        _rate(stats["honmei_tan_return"],  races_n * 100),
        "fuku_rate":       _rate(stats["honmei_fuku_return"], races_n * 100),
        "choko_tan_rate":  _rate(stats["choko_tan_return"], max(stats["choko_invest"], 1)),
        "choko_fuku_rate": _rate(stats["choko_fuku_return"], max(stats["choko_invest"], 1)),
        "ana_tan_rate":    _rate(stats["ana_tan_return"],  max(stats["ana_invest"], 1)),
        "ana_fuku_rate":   _rate(stats["ana_fuku_return"], max(stats["ana_invest"], 1)),
        "uma_rate":        _rate(stats["umaren_return"], max(stats["umaren_invest"], 1)),
        "wide_rate":       _rate(stats["wide_ana_return"], max(stats["wide_ana_invest"], 1)),
        "sanrenpuku_rate": _rate(stats["sanrenpuku_return"], max(stats["sanrenpuku_invest"], 1)),
        "shiba_rate":      _rate(stats["shiba_return"], max(stats["shiba_races"] * 100, 1)),
        "dart_rate":       _rate(stats["dart_return"],  max(stats["dart_races"] * 100, 1)),
        "tandist_rate":    {k: _dist_rate(k) for k in stats["dist_stats"]},
        "class_rate":      {k: _class_rate(k) for k in stats["class_stats"]},
        "honmei_avg_ai":   round(sum(stats["honmei_ai_probs"]) / len(stats["honmei_ai_probs"]) * 100, 1) if stats["honmei_ai_probs"] else 0.0,
        "winner_avg_ai":   round(sum(stats["winner_ai_probs"]) / len(stats["winner_ai_probs"]) * 100, 1) if stats["winner_ai_probs"] else 0.0,
    }

    logger.info(
        f"集計完了 {races_n}R: 本命単勝{rates['tan_rate']}% 複勝{rates['fuku_rate']}%"
    )

    ok = _send_review_direct(stats, rates, date_label)
    if ok:
        logger.info("Discord 送信成功")
    else:
        logger.error("Discord 送信失敗")

    # ── HF Hub に ai_daily_history.csv を保存 ────────────────────────────
    try:
        import io
        import pandas as pd
        from huggingface_hub import HfApi, hf_hub_download

        import json
        daily_row = {
            "日付":              date_hf.replace("-", "/"),
            # 本命◎
            "本命レース数":       races_n,
            "本命単勝的中数":     stats["honmei_tan_hits"],
            "本命単勝回収率":     rates["tan_rate"],
            "本命複勝的中数":     stats["honmei_fuku_hits"],
            "本命複勝回収率":     rates["fuku_rate"],
            # 馬連
            "馬連的中数":         stats["umaren_hits"],
            "馬連回収率":         rates["uma_rate"],
            # 三連複
            "三連複的中数":       stats["sanrenpuku_hits"],
            "三連複回収率":       rates["sanrenpuku_rate"],
            # 超狙い馬
            "超狙い馬数":         int(stats["choko_invest"] // 100),
            "超狙い馬単勝的中数": stats["choko_tan_hits"],
            "超狙い馬単勝回収率": rates["choko_tan_rate"],
            "超狙い馬複勝的中数": stats["choko_fuku_hits"],
            "超狙い馬複勝回収率": rates["choko_fuku_rate"],
            # 穴馬
            "穴馬数":             int(stats["ana_invest"] // 100),
            "穴馬単勝的中数":     stats["ana_tan_hits"],
            "穴馬単勝回収率":     rates["ana_tan_rate"],
            "穴馬複勝的中数":     stats["ana_fuku_hits"],
            "穴馬複勝回収率":     rates["ana_fuku_rate"],
            # 穴馬ワイド
            "穴馬ワイド対象R":    stats["wide_ana_races"],
            "穴馬ワイド的中数":   stats["wide_ana_hits"],
            "穴馬ワイド回収率":   rates["wide_rate"],
            # 芝/ダート
            "芝レース数":         stats["shiba_races"],
            "芝単勝回収率":       rates["shiba_rate"],
            "ダートレース数":     stats["dart_races"],
            "ダート単勝回収率":   rates["dart_rate"],
            # 距離帯別
            "短距離_R数":   stats["dist_stats"]["短距離"]["R数"],
            "短距離_的中":  stats["dist_stats"]["短距離"]["的中"],
            "短距離_単勝回収率": rates["tandist_rate"]["短距離"],
            "マイル_R数":   stats["dist_stats"]["マイル"]["R数"],
            "マイル_的中":  stats["dist_stats"]["マイル"]["的中"],
            "マイル_単勝回収率": rates["tandist_rate"]["マイル"],
            "中距離_R数":   stats["dist_stats"]["中距離"]["R数"],
            "中距離_的中":  stats["dist_stats"]["中距離"]["的中"],
            "中距離_単勝回収率": rates["tandist_rate"]["中距離"],
            "長距離_R数":   stats["dist_stats"]["長距離"]["R数"],
            "長距離_的中":  stats["dist_stats"]["長距離"]["的中"],
            "長距離_単勝回収率": rates["tandist_rate"]["長距離"],
            # クラス別
            "低クラス_R数":   stats["class_stats"]["低クラス"]["R数"],
            "低クラス_的中":  stats["class_stats"]["低クラス"]["的中"],
            "低クラス_単勝回収率": rates["class_rate"]["低クラス"],
            "高クラス_R数":   stats["class_stats"]["高クラス"]["R数"],
            "高クラス_的中":  stats["class_stats"]["高クラス"]["的中"],
            "高クラス_単勝回収率": rates["class_rate"]["高クラス"],
            # Calibration / AIスコア
            "本命平均AIスコア":     rates["honmei_avg_ai"],
            "実際勝者の平均AI勝率": rates["winner_avg_ai"],
            # 競馬場別（JSON）
            "競馬場別": json.dumps(stats["venue_stats"], ensure_ascii=False),
        }
        new_df = pd.DataFrame([daily_row])

        # 既存CSVを取得してマージ
        try:
            csv_path = hf_hub_download(HF_REPO_ID, "ai_daily_history.csv",
                                        repo_type="dataset", token=HF_TOKEN)
            existing_df = pd.read_csv(csv_path)
            existing_df = existing_df[existing_df["日付"] != daily_row["日付"]]
            merged_df = pd.concat([existing_df, new_df], ignore_index=True)
        except Exception:
            merged_df = new_df

        buf = io.BytesIO()
        merged_df.to_csv(buf, index=False)
        buf.seek(0)
        HfApi(token=HF_TOKEN).upload_file(
            path_or_fileobj=buf,
            path_in_repo="ai_daily_history.csv",
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            commit_message=f"成績履歴更新 {date_hf}",
        )
        logger.info("ai_daily_history.csv を HF Hub に保存しました")
    except Exception as e:
        logger.warning(f"ai_daily_history.csv 保存失敗（スキップ）: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="振り返り自動集計 → Discord 通知")
    parser.add_argument("--date", type=str, default=None, help="対象日 YYYYMMDD（省略時は本日）")
    args = parser.parse_args()
    run(args.date)
