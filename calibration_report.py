"""
勝率キャリブレーション診断レポート（Phase 2 の測定基盤）

目的:
  本命◎のAI勝率が慢性的に約2倍過信（予測~22-24% vs 実~11-13%）という積年の課題を、
  **本番のOOSデータ**（auto_review.py が蓄積する ai_race_history.csv）で定量化する。
  勝負/回避ラベルのマジックナンバーを廃止し「実ROIから閾値を導く」ための材料も出す。

  ⚠️ このスクリプトは**測定専用**。モデルも推論も一切変更しない。
     ここで出た数字を見てから inference 側の補正やラベル閾値を決めること。

使い方:
  python calibration_report.py [--days 120] [--discord] [--save-csv out.csv]

  GitHub Actions からは「キャリブレーション診断」ワークフローを手動実行する
  （HF_TOKEN を持っているのは Actions / HF Space 側だけなのでローカルでは動かない）。

必要な環境変数:
  HF_TOKEN   - HuggingFace API トークン（read 権限）
  HF_REPO_ID - モデル保存先 Dataset リポジトリ ID
  DISCORD_REVIEW_WEBHOOK_URL / DISCORD_WEBHOOK_URL - --discord 指定時のみ
"""

import os
import sys
import math
import argparse
import logging

import numpy as np
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("calibration_report")

HF_TOKEN   = os.environ.get("HF_TOKEN", "")
HF_REPO_ID = os.environ.get("HF_REPO_ID", "")
WEBHOOK    = (os.environ.get("DISCORD_REVIEW_WEBHOOK_URL", "").strip()
              or os.environ.get("DISCORD_WEBHOOK_URL", "").strip())

EPS = 1e-9


# ──────────────────────────────────────────────────────────────
# データ取得
# ──────────────────────────────────────────────────────────────
def load_history(days: int) -> pd.DataFrame:
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(HF_REPO_ID, "ai_race_history.csv",
                           repo_type="dataset", token=HF_TOKEN)
    df = pd.read_csv(path, dtype={"レースID": str})
    df["日付"] = pd.to_datetime(df["日付"], errors="coerce")
    df = df.dropna(subset=["日付"])
    if days:
        cutoff = df["日付"].max() - pd.Timedelta(days=days)
        df = df[df["日付"] >= cutoff]
    for c in ("AI勝率", "複勝率", "単勝オッズ", "EV", "市場勝率"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("1着", "複勝内", "頭数", "AI順位", "人気"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    return df.reset_index(drop=True)


# ──────────────────────────────────────────────────────────────
# 指標
# ──────────────────────────────────────────────────────────────
def ece(pred: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> tuple:
    """Expected Calibration Error と ビンごとの内訳を返す（分位ビン）。"""
    if len(pred) == 0:
        return 0.0, []
    qs = np.unique(np.quantile(pred, np.linspace(0, 1, n_bins + 1)))
    if len(qs) < 3:
        qs = np.array([pred.min() - EPS, pred.mean(), pred.max() + EPS])
    idx = np.clip(np.digitize(pred, qs[1:-1]), 0, len(qs) - 2)
    rows, total = [], len(pred)
    err = 0.0
    for b in range(len(qs) - 1):
        m = idx == b
        n = int(m.sum())
        if n == 0:
            continue
        p, a = float(pred[m].mean()), float(actual[m].mean())
        err += n / total * abs(p - a)
        rows.append((qs[b], qs[b + 1], n, p, a))
    return err, rows


def brier(pred: np.ndarray, actual: np.ndarray) -> float:
    return float(np.mean((pred - actual) ** 2)) if len(pred) else 0.0


def race_logloss(df: pd.DataFrame, col: str) -> float:
    """レース単位の多クラス logloss（= -log(勝ち馬に付けた確率)の平均）。

    「どの馬が勝つか」に対する厳密な proper scoring rule。温度の最適化はこれで行う。
    """
    lls = []
    for _, g in df.groupby("レースID"):
        p = g[col].to_numpy(dtype=float)
        s = p.sum()
        if s <= 0:
            continue
        p = p / s
        win = g["1着"].to_numpy() == 1
        if not win.any():
            continue
        lls.append(-math.log(max(float(p[win][0]), EPS)))
    return float(np.mean(lls)) if lls else float("nan")


def apply_temperature(df: pd.DataFrame, t: float, src: str = "AI勝率") -> np.ndarray:
    """レース内で p^(1/T) 再正規化した確率を返す。

    T>1 = なだらかにする（過信の是正）/ T<1 = 尖らせる。
    レース内の順位は変わらないため、◎の選定には一切影響しない。
    """
    out = np.zeros(len(df), dtype=float)
    for _, g in df.groupby("レースID"):
        p = np.clip(g[src].to_numpy(dtype=float), EPS, None) ** (1.0 / t)
        out[g.index.to_numpy()] = p / max(p.sum(), EPS)
    return out


def roi(sub: pd.DataFrame, kind: str = "tan") -> tuple:
    """(回収率%, 的中数, 件数)。払戻列があれば使い、無ければオッズから復元する。"""
    n = len(sub)
    if n == 0:
        return 0.0, 0, 0
    if kind == "tan":
        hit = (sub["1着"] == 1)
        if "単勝払戻" in sub.columns and pd.to_numeric(sub["単勝払戻"], errors="coerce").fillna(0).sum() > 0:
            ret = pd.to_numeric(sub["単勝払戻"], errors="coerce").fillna(0).sum()
        else:
            ret = (sub.loc[hit, "単勝オッズ"].fillna(0) * 100).sum()
    else:
        hit = (sub["複勝内"] == 1)
        if "複勝払戻" in sub.columns and pd.to_numeric(sub["複勝払戻"], errors="coerce").fillna(0).sum() > 0:
            ret = pd.to_numeric(sub["複勝払戻"], errors="coerce").fillna(0).sum()
        else:
            return float("nan"), int(hit.sum()), n   # 複勝は払戻列が無いと復元不可
    return round(float(ret) / (n * 100) * 100, 1), int(hit.sum()), n


# ──────────────────────────────────────────────────────────────
# レポート本体
# ──────────────────────────────────────────────────────────────
def build_report(df: pd.DataFrame) -> str:
    L = []
    add = L.append

    df = df.reset_index(drop=True)
    if not pd.api.types.is_datetime64_any_dtype(df["日付"]):
        df["日付"] = pd.to_datetime(df["日付"], errors="coerce")
    races = df["レースID"].nunique()
    d0, d1 = df["日付"].min().date(), df["日付"].max().date()
    add("=" * 62)
    add("🎯 勝率キャリブレーション診断（本番OOS / ai_race_history.csv）")
    add("=" * 62)
    add(f"期間: {d0} 〜 {d1}   レース数: {races}   のべ出走頭数: {len(df)}")
    if races < 100:
        add(f"⚠️ レース数が少ないため参考値です（目安: 300R以上で安定。現在 {races}R）")
    add("")

    pred = df["AI勝率"].to_numpy(dtype=float)
    act  = (df["1着"] == 1).to_numpy(dtype=float)

    # ── 1. 全馬のキャリブレーション ─────────────────────────────
    e, rows = ece(pred, act)
    add("【1. 全馬 AI勝率のキャリブレーション】")
    add(f"  ECE {e*100:.2f}pp / Brier {brier(pred, act):.4f} / "
        f"予測平均 {pred.mean()*100:.1f}% vs 実勝率 {act.mean()*100:.1f}%")
    add("  " + "-" * 56)
    add(f"  {'予測レンジ':>16} {'頭数':>6} {'予測':>7} {'実績':>7} {'乖離':>8}")
    for lo, hi, n, p, a in rows:
        add(f"  {lo*100:6.1f}%〜{hi*100:5.1f}% {n:6d} {p*100:6.1f}% {a*100:6.1f}% {(p-a)*100:+7.1f}pp")
    add("")

    # ── 2. ◎(AI順位1)のキャリブレーション・頭数帯別 ───────────────
    h = df[df["AI順位"] == 1]
    add("【2. 本命◎ の予測 vs 実績（頭数帯別）】")
    add("  ※ 小頭数ほど softmax が薄まり ◎勝率が下がる『頭数バイアス』の実測")
    add("  " + "-" * 56)
    add(f"  {'頭数帯':>8} {'R数':>5} {'予測勝率':>9} {'実勝率':>8} {'乖離':>9} {'単ROI':>8}")
    bands = [("〜9頭", 0, 9), ("10-13頭", 10, 13), ("14-15頭", 14, 15), ("16頭〜", 16, 99)]
    for name, lo, hi in bands:
        s = h[(h["頭数"] >= lo) & (h["頭数"] <= hi)]
        if len(s) == 0:
            continue
        p, a = s["AI勝率"].mean(), (s["1着"] == 1).mean()
        r, _, n = roi(s, "tan")
        add(f"  {name:>8} {n:5d} {p*100:8.1f}% {a*100:7.1f}% {(p-a)*100:+8.1f}pp {r:7.1f}%")
    p, a = h["AI勝率"].mean(), (h["1着"] == 1).mean()
    r, hit, n = roi(h, "tan")
    add(f"  {'全体':>8} {n:5d} {p*100:8.1f}% {a*100:7.1f}% {(p-a)*100:+8.1f}pp {r:7.1f}%")
    if a > 0:
        add(f"  → ◎の過信倍率: **{p/max(a, EPS):.2f}倍**（1.0が理想）")
    add("")

    # ── 3. 温度補正の推定 ──────────────────────────────────────
    add("【3. 事後の温度補正 T* の推定】")
    add("  レース内で p^(1/T) 再正規化。T>1で過信を緩和。順位は不変＝◎選定に影響しない。")
    base_ll = race_logloss(df, "AI勝率")
    best_t, best_ll = 1.0, base_ll
    grid = [round(x, 2) for x in np.arange(0.7, 3.01, 0.05)]
    curve = []
    work = df.reset_index(drop=True)
    for t in grid:
        work["_q"] = apply_temperature(work, t)
        ll = race_logloss(work, "_q")
        curve.append((t, ll))
        if ll < best_ll:
            best_t, best_ll = t, ll
    work["_q"] = apply_temperature(work, best_t)
    e_after, _ = ece(work["_q"].to_numpy(), act)
    h_after = work[work["AI順位"] == 1]
    add(f"  現状 T=1.00: レースlogloss {base_ll:.4f} / ECE {e*100:.2f}pp")
    add(f"  最適 T={best_t:.2f}: レースlogloss {best_ll:.4f} / ECE {e_after*100:.2f}pp")
    add(f"  補正後の◎: 予測 {h_after['_q'].mean()*100:.1f}% vs 実 "
        f"{(h_after['1着']==1).mean()*100:.1f}%")
    if best_t > 1.05:
        add(f"  → **過信を確認**。T={best_t:.2f} 相当のなだらか化が必要。")
    elif best_t < 0.95:
        add(f"  → 逆に過小評価。T={best_t:.2f} 相当で尖らせるべき。")
    else:
        add("  → 現状でおおむね適正（追加補正は不要）。")
    add("")

    # ── 4. 複勝率のキャリブレーション ────────────────────────────
    if "複勝率" in df.columns and df["複勝内"].sum() > 0:
        fp = df["複勝率"].to_numpy(dtype=float)
        fa = (df["複勝内"] == 1).to_numpy(dtype=float)
        fe, _ = ece(fp, fa)
        hf_ = df[df["AI順位"] == 1]
        add("【4. 複勝率のキャリブレーション】")
        add(f"  全馬 ECE {fe*100:.2f}pp / 予測平均 {fp.mean()*100:.1f}% vs 実 {fa.mean()*100:.1f}%")
        add(f"  ◎    予測 {hf_['複勝率'].mean()*100:.1f}% vs 実 "
            f"{(hf_['複勝内']==1).mean()*100:.1f}%")
        add("")

    # ── 5. ラベル閾値の材料: p1/p2比 と ◎勝率 のバケット別ROI ──────
    add("【5. ラベル再設計の材料（◎の実ROI）】")
    add("  ※ 現行ラベルは絶対値マジックナンバー。ここの実ROIから閾値を引き直す。")
    top2 = (df[df["AI順位"] <= 2]
            .sort_values(["レースID", "AI順位"])
            .groupby("レースID")["AI勝率"].apply(list))
    ratio = {rid: (v[0] / max(v[1], EPS)) for rid, v in top2.items() if len(v) >= 2}
    h = h.copy()
    h["p1p2"] = h["レースID"].map(ratio)

    add("  ● p1/p2 比 別")
    add(f"  {'比':>10} {'R数':>5} {'実勝率':>8} {'単ROI':>8}")
    for lbl, lo, hi in [("〜1.1", 0, 1.1), ("1.1-1.25", 1.1, 1.25),
                        ("1.25-1.5", 1.25, 1.5), ("1.5-2.0", 1.5, 2.0), ("2.0〜", 2.0, 99)]:
        s = h[(h["p1p2"] > lo) & (h["p1p2"] <= hi)]
        if len(s) == 0:
            continue
        r, hit, n = roi(s, "tan")
        add(f"  {lbl:>10} {n:5d} {(s['1着']==1).mean()*100:7.1f}% {r:7.1f}%")

    add("  ● ◎のAI勝率 別")
    add(f"  {'勝率':>10} {'R数':>5} {'実勝率':>8} {'単ROI':>8}")
    for lbl, lo, hi in [("〜12%", 0, .12), ("12-16%", .12, .16), ("16-20%", .16, .20),
                        ("20-25%", .20, .25), ("25%〜", .25, 1.0)]:
        s = h[(h["AI勝率"] > lo) & (h["AI勝率"] <= hi)]
        if len(s) == 0:
            continue
        r, hit, n = roi(s, "tan")
        add(f"  {lbl:>10} {n:5d} {(s['1着']==1).mean()*100:7.1f}% {r:7.1f}%")

    # ── 6. 現行ラベル別ROI（判定列がある日以降のみ）───────────────
    if "判定" in df.columns and df["判定"].notna().any():
        add("  ● 現行ラベル別（判定列があるレースのみ）")
        add(f"  {'判定':>10} {'R数':>5} {'実勝率':>8} {'単ROI':>8}")
        for lbl, s in h.groupby(h["判定"].fillna("不明")):
            r, hit, n = roi(s, "tan")
            add(f"  {str(lbl):>10} {n:5d} {(s['1着']==1).mean()*100:7.1f}% {r:7.1f}%")
    else:
        add("  ● 現行ラベル別: 『判定』列がまだありません（2026-09-25以降の振り返りから記録）")
    add("")

    # ── 7. 市場エッジ（ログ専用・予想には使わない）─────────────────
    if "市場勝率" in df.columns and df["市場勝率"].notna().any():
        m = df.dropna(subset=["市場勝率"])
        add("【6. 参考: 市場勝率との比較（ログ専用・予想には使わない）】")
        add(f"  全馬 市場 {m['市場勝率'].mean()*100:.1f}% vs AI {m['AI勝率'].mean()*100:.1f}% "
            f"vs 実 {(m['1着']==1).mean()*100:.1f}%")
        mh = m[m["AI順位"] == 1]
        if len(mh):
            add(f"  ◎    市場 {mh['市場勝率'].mean()*100:.1f}% vs AI {mh['AI勝率'].mean()*100:.1f}% "
                f"vs 実 {(mh['1着']==1).mean()*100:.1f}%")
        add("")

    add("=" * 62)
    add("【次のアクション】")
    if best_t > 1.05:
        add(f"  1. 事後温度 T={best_t:.2f} を推論に入れるか検討（順位不変＝再学習不要）。")
    add("  2. 上の p1/p2 比・AI勝率バケットの実ROIから 🔥勝負/⚠️回避 の閾値を引き直す。")
    add("  3. レース数が300を超えてから本採用すること（少数だと偶然を拾う）。")
    add("=" * 62)
    return "\n".join(L)


def post_discord(text: str):
    if not WEBHOOK:
        logger.warning("Webhook 未設定のため Discord 送信をスキップ")
        return
    # コードブロックで囲って 1900 字ずつ送る
    lines, buf = text.split("\n"), ""
    chunks = []
    for ln in lines:
        if len(buf) + len(ln) + 1 > 1850:
            chunks.append(buf); buf = ln
        else:
            buf = f"{buf}\n{ln}" if buf else ln
    if buf:
        chunks.append(buf)
    for i, c in enumerate(chunks):
        try:
            requests.post(WEBHOOK,
                          json={"content": f"```\n{c}\n```",
                                "username": "keiba-ebye 🎯キャリブ"},
                          timeout=20)
        except Exception as e:
            logger.error(f"Discord送信失敗 (chunk {i+1}): {e}")


def main():
    ap = argparse.ArgumentParser(description="勝率キャリブレーション診断（測定専用）")
    ap.add_argument("--days", type=int, default=120, help="直近何日分を対象にするか（0=全期間）")
    ap.add_argument("--discord", action="store_true", help="結果を Discord に投稿する")
    ap.add_argument("--save-csv", type=str, default="", help="対象データをCSVに保存")
    args = ap.parse_args()

    if not HF_TOKEN or not HF_REPO_ID:
        logger.error("HF_TOKEN / HF_REPO_ID が未設定です。"
                     "このスクリプトは GitHub Actions から実行してください。")
        sys.exit(1)

    df = load_history(args.days)
    if df.empty:
        logger.error("ai_race_history.csv にデータがありません。")
        sys.exit(1)
    if args.save_csv:
        df.to_csv(args.save_csv, index=False)
        logger.info(f"{args.save_csv} に保存しました")

    report = build_report(df)
    print(report)
    if args.discord:
        post_discord(report)


if __name__ == "__main__":
    main()
