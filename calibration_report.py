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


def _race_normalize(df: pd.DataFrame, values: np.ndarray) -> np.ndarray:
    """レース内で合計1になるよう正規化（ベクトル化）。"""
    s = pd.Series(values).groupby(df["レースID"].to_numpy()).transform("sum").to_numpy()
    return values / np.maximum(s, EPS)


def race_logloss(df: pd.DataFrame, col: str) -> float:
    """レース単位の多クラス logloss（= -log(勝ち馬に付けた確率)の平均）。

    「どの馬が勝つか」に対する厳密な proper scoring rule。温度の最適化はこれで行う。
    """
    q = _race_normalize(df, np.clip(df[col].to_numpy(dtype=float), 0, None))
    win = df["1着"].to_numpy() == 1
    if not win.any():
        return float("nan")
    return float(-np.log(np.clip(q[win], EPS, None)).mean())


def apply_temperature(df: pd.DataFrame, t: float, src: str = "AI勝率") -> np.ndarray:
    """レース内で p^(1/T) 再正規化した確率を返す。

    T>1 = なだらかにする（過信の是正）/ T<1 = 尖らせる。
    レース内の順位は変わらないため、◎の選定には一切影響しない。
    """
    p = np.clip(df[src].to_numpy(dtype=float), EPS, None) ** (1.0 / t)
    return _race_normalize(df, p)


# ──────────────────────────────────────────────────────────────
# 有意性の判定（2026-09-25 追加）
# ------------------------------------------------------------
# 初版は誤差範囲を出していなかったため、たった51レースの「当たりすぎた帯」に
# 全体が引っ張られて『過小評価・T*=0.70』という結論に見えてしまった。
# 少サンプルで誤った結論に飛ばないよう、z値と信頼区間を必ず併記する。
# ──────────────────────────────────────────────────────────────
def pb_z(obs_wins: float, probs: np.ndarray) -> float:
    """ポアソン二項の z 値 = (観測勝利数 − Σp) / sqrt(Σp(1−p))。

    各レースで勝つ確率が異なるので、単純な二項ではなくこれを使う。
    |z| < 2 は「誤差の範囲」、|z| >= 3 は「この帯だけ何かが起きている」と読む。
    """
    probs = np.asarray(probs, dtype=float)
    if len(probs) == 0:
        return 0.0
    var = float((probs * (1 - probs)).sum())
    if var <= 0:
        return 0.0
    return (float(obs_wins) - float(probs.sum())) / math.sqrt(var)


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple:
    """二項比率の Wilson 信頼区間（少サンプルでも破綻しない）。"""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def _ci_str(k: int, n: int) -> str:
    lo, hi = wilson_ci(k, n)
    return f"[{lo*100:4.1f},{hi*100:5.1f}]"


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
    h = df[df["AI順位"] == 1].copy()
    add("【2. 本命◎ の予測 vs 実績（頭数帯別）】")
    add("  z = (実際の勝ち数 − AI予測の期待勝ち数) / 標準偏差。|z|<2 は誤差の範囲、")
    add("  |z|>=3 は『この帯だけ何かが起きている』外れ帯として ★ を付け、全体から除いて再計算する。")
    add("  " + "-" * 70)
    add(f"  {'頭数帯':>8} {'R数':>5} {'◎勝':>4} {'AI予測':>7} {'期待':>6} {'zAI':>6} "
        f"{'市場':>6} {'z市場':>6} {'実勝率':>7} {'95%CI':>13} {'単ROI':>7}")
    bands = [("〜9頭", 0, 9), ("10-13頭", 10, 13), ("14-15頭", 14, 15), ("16頭〜", 16, 99)]
    outliers = []
    for name, lo, hi in bands:
        s = h[(h["頭数"] >= lo) & (h["頭数"] <= hi)]
        n = len(s)
        if n == 0:
            continue
        k = int((s["1着"] == 1).sum())
        p_ai = s["AI勝率"].to_numpy(dtype=float)
        z_ai = pb_z(k, p_ai)
        mkt = s["市場勝率"].dropna().to_numpy(dtype=float) if "市場勝率" in s.columns else np.array([])
        z_mk = pb_z(k * len(mkt) / max(n, 1), mkt) if len(mkt) else float("nan")
        r, _, _ = roi(s, "tan")
        flag = "★" if abs(z_ai) >= 3 else " "
        if abs(z_ai) >= 3:
            outliers.append((name, lo, hi))
        add(f" {flag}{name:>8} {n:5d} {k:4d} {p_ai.mean()*100:6.1f}% {p_ai.sum():6.1f} {z_ai:+6.2f} "
            f"{(mkt.mean()*100 if len(mkt) else float('nan')):5.1f}% {z_mk:+6.2f} "
            f"{k/n*100:6.1f}% {_ci_str(k, n):>13} {r:6.1f}%")

    def _summary_line(label, s):
        n = len(s)
        k = int((s["1着"] == 1).sum())
        p_ai = s["AI勝率"].to_numpy(dtype=float)
        mkt = s["市場勝率"].dropna().to_numpy(dtype=float) if "市場勝率" in s.columns else np.array([])
        r, _, _ = roi(s, "tan")
        add(f"  {label:>8} {n:5d} {k:4d} {p_ai.mean()*100:6.1f}% {p_ai.sum():6.1f} "
            f"{pb_z(k, p_ai):+6.2f} {(mkt.mean()*100 if len(mkt) else float('nan')):5.1f}% "
            f"{(pb_z(k*len(mkt)/max(n,1), mkt) if len(mkt) else float('nan')):+6.2f} "
            f"{k/n*100:6.1f}% {_ci_str(k, n):>13} {r:6.1f}%")
        return n, k, p_ai.mean(), (mkt.mean() if len(mkt) else float("nan"))

    add("  " + "-" * 70)
    n_all, k_all, p_all, m_all = _summary_line("全体", h)

    h_trim = h
    if outliers:
        mask = np.ones(len(h), dtype=bool)
        for _, lo, hi in outliers:
            mask &= ~((h["頭数"] >= lo) & (h["頭数"] <= hi)).to_numpy()
        h_trim = h[mask]
        if len(h_trim) > 0:
            n_t, k_t, p_t, m_t = _summary_line("★除く", h_trim)
            add("")
            add(f"  ⚠️ 外れ帯 {'・'.join(o[0] for o in outliers)} を除くと "
                f"実勝率 {k_all/max(n_all,1)*100:.1f}% → {k_t/max(n_t,1)*100:.1f}%。")
            add(f"     全体の数字はこの帯に強く引っ張られている。結論は『★除く』側で判断すること。")
    add("")
    if k_all:
        add(f"  → ◎の過信倍率（全体）: {p_all/max(k_all/max(n_all,1), EPS):.2f}倍"
            f"（1.0が理想・1未満=過小評価）")
        if outliers and len(h_trim):
            _a_t = int((h_trim['1着'] == 1).sum()) / max(len(h_trim), 1)
            add(f"  → ◎の過信倍率（★除く）: "
                f"{h_trim['AI勝率'].mean()/max(_a_t, EPS):.2f}倍  ← **こちらを見る**")
    add("")

    # ── 2b. 市場を物差しにした較正（少サンプルではこちらが頑健）──────────
    # 実績（当たり外れ）は少レースだと揺れるが、「AIの値付け vs 市場の値付け」は
    # 同じ馬への評価の比較なので、その日の当たり外れに左右されない。
    if "市場勝率" in df.columns and df["市場勝率"].notna().any():
        add("【2b. ★市場を物差しにした較正（少サンプルではこちらが頑健）】")
        add("  市場（オッズ）はよく較正されている。同じ馬にAIがいくら付けたかを比べる。")
        add("  実績の当たり外れに依存しないため、レース数が少ないうちはこの比を信用する。")
        add("  " + "-" * 56)
        add(f"  {'AI順位':>7} {'頭数':>6} {'AI予測':>8} {'市場':>8} {'AI/市場':>9} {'実勝率':>8}")
        for rk in (1, 2, 3, 4, 5):
            s = df[(df["AI順位"] == rk)].dropna(subset=["市場勝率"])
            if len(s) == 0:
                continue
            ai_m, mk_m = s["AI勝率"].mean(), s["市場勝率"].mean()
            add(f"  {rk:>7} {len(s):6d} {ai_m*100:7.1f}% {mk_m*100:7.1f}% "
                f"{ai_m/max(mk_m, EPS):8.2f}倍 {(s['1着']==1).mean()*100:7.1f}%")
        s1 = df[df["AI順位"] == 1].dropna(subset=["市場勝率"])
        if len(s1):
            ratio = s1["AI勝率"].mean() / max(s1["市場勝率"].mean(), EPS)
            add("")
            if ratio < 0.85:
                add(f"  → ◎で AI/市場 = {ratio:.2f}倍。**AIは自分の◎を市場より低く見積もっている（過小評価）**。")
                add(f"     補正するなら T ≈ {max(0.5, ratio):.2f} 付近が上限の目安（市場に一致させる量）。")
            elif ratio > 1.15:
                add(f"  → ◎で AI/市場 = {ratio:.2f}倍。**AIは自分の◎を市場より高く見積もっている（過信）**。")
            else:
                add(f"  → ◎で AI/市場 = {ratio:.2f}倍。市場とおおむね同水準。")
        add("")

    # ── 3. 温度補正の推定（＋安定性チェック）──────────────────────
    add("【3. 事後の温度補正 T* の推定】")
    add("  レース内で p^(1/T) 再正規化。T>1で過信を緩和。順位は不変＝◎選定に影響しない。")

    grid = [round(x, 2) for x in np.arange(0.6, 3.01, 0.05)]

    def _fit_t(sub: pd.DataFrame) -> tuple:
        """そのデータでの最適 T と logloss を返す。"""
        sub = sub.reset_index(drop=True)
        if sub.empty or (sub["1着"] == 1).sum() < 5:
            return float("nan"), float("nan")
        best_t, best_ll = 1.0, race_logloss(sub, "AI勝率")
        for t in grid:
            sub["_q"] = apply_temperature(sub, t)
            ll = race_logloss(sub, "_q")
            if ll < best_ll:
                best_t, best_ll = t, ll
        return best_t, best_ll

    work = df.reset_index(drop=True)
    base_ll = race_logloss(work, "AI勝率")
    best_t, best_ll = _fit_t(work)
    work["_q"] = apply_temperature(work, best_t)
    e_after, _ = ece(work["_q"].to_numpy(), act)
    h_after = work[work["AI順位"] == 1]
    add(f"  現状 T=1.00 : レースlogloss {base_ll:.4f} / ECE {e*100:.2f}pp")
    add(f"  最適 T={best_t:.2f} : レースlogloss {best_ll:.4f} / ECE {e_after*100:.2f}pp "
        f"（改善 {(base_ll-best_ll)/max(base_ll,EPS)*100:.1f}%）")
    add(f"  補正後の◎: 予測 {h_after['_q'].mean()*100:.1f}% vs 実 "
        f"{(h_after['1着']==1).mean()*100:.1f}%")
    add("")

    # 安定性チェック: 同じ結論が別の切り口でも出るか。ばらつけば「まだ決められない」。
    add("  ● 安定性チェック（この3つが揃わないうちは採用しない）")
    checks = []
    if outliers:
        ids = set(h_trim["レースID"])
        t_trim, _ = _fit_t(df[df["レースID"].isin(ids)])
        checks.append(("外れ帯★を除く", t_trim))
    mid = df["日付"].quantile(0.5)
    t_1st, _ = _fit_t(df[df["日付"] <= mid])
    t_2nd, _ = _fit_t(df[df["日付"] > mid])
    checks.append(("前半期間のみ", t_1st))
    checks.append(("後半期間のみ", t_2nd))
    add(f"  {'切り口':>14} {'T*':>7}")
    add(f"  {'全データ':>14} {best_t:7.2f}")
    for lbl, t in checks:
        add(f"  {lbl:>14} {t:7.2f}" if not math.isnan(t) else f"  {lbl:>14}    データ不足")
    vals = [best_t] + [t for _, t in checks if not math.isnan(t)]
    spread = max(vals) - min(vals) if len(vals) > 1 else 0.0
    add("")
    if spread > 0.3:
        add(f"  ⚠️ **T* が切り口によって {min(vals):.2f}〜{max(vals):.2f} とばらついている（幅 {spread:.2f}）。**")
        add(f"     まだ推定が安定していない。この段階で推論に入れてはいけない。")
    elif best_t > 1.05:
        add(f"  → 過信を確認（T*={best_t:.2f}）。切り口によらず安定している。")
    elif best_t < 0.95:
        add(f"  → 過小評価（T*={best_t:.2f}）。切り口によらず安定している。")
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

    def _bucket_table(title, col, cuts):
        add(f"  ● {title}")
        add(f"  {'区間':>10} {'R数':>5} {'実勝率':>8} {'95%CI':>14} {'単ROI':>8} {'判定':>6}")
        seen = []
        for lbl, lo, hi in cuts:
            s = h[(h[col] > lo) & (h[col] <= hi)]
            n = len(s)
            if n == 0:
                continue
            k = int((s["1着"] == 1).sum())
            r, _, _ = roi(s, "tan")
            # 30R未満、または95%CIの幅が15pp超なら「ノイズ」扱い
            ci_lo, ci_hi = wilson_ci(k, n)
            noisy = (n < 30) or ((ci_hi - ci_lo) * 100 > 15)
            add(f"  {lbl:>10} {n:5d} {k/n*100:7.1f}% {_ci_str(k, n):>14} {r:7.1f}% "
                f"{'ノイズ' if noisy else 'OK':>6}")
            seen.append((lbl, k / n, noisy))
        # 単調性チェック: 区間が上がるほど良くなるはずが逆転していればノイズの証拠
        solid = [x for x in seen if not x[2]]
        rates = [x[1] for x in seen]
        if len(rates) >= 3 and any(rates[i] > rates[i + 1] for i in range(len(rates) - 1)):
            add("     ⚠️ 区間をまたいで実勝率が逆転している＝ノイズの特徴。ここから閾値を引かないこと。")
        if len(solid) < 2:
            add("     ⚠️ 十分なサンプルの区間が2つ未満。閾値の根拠にはまだ使えない。")

    _bucket_table("p1/p2 比 別", "p1p2",
                  [("〜1.1", 0, 1.1), ("1.1-1.25", 1.1, 1.25), ("1.25-1.5", 1.25, 1.5),
                   ("1.5-2.0", 1.5, 2.0), ("2.0〜", 2.0, 99)])
    _bucket_table("◎のAI勝率 別", "AI勝率",
                  [("〜12%", 0, .12), ("12-16%", .12, .16), ("16-20%", .16, .20),
                   ("20-25%", .20, .25), ("25%〜", .25, 1.0)])

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
    add("【判定】")
    blockers = []
    if races < 300:
        blockers.append(f"レース数が {races}R（採用の目安 300R に未達）")
    if spread > 0.3:
        blockers.append(f"T* が切り口で {min(vals):.2f}〜{max(vals):.2f} とばらついている")
    if outliers:
        blockers.append(f"外れ帯 {'・'.join(o[0] for o in outliers)} が全体を歪めている")
    if blockers:
        add("  ⛔ **まだ推論に反映しないこと**。理由:")
        for b in blockers:
            add(f"     ・{b}")
        add("  → やること: データを貯める。上の【2b】市場との比（実績の当たり外れに依存しない）")
        add("     だけは今でも読める指標なので、方向性の確認にはそちらを使う。")
    else:
        add(f"  ✅ 採用条件を満たしている。事後温度 T={best_t:.2f} の導入を検討してよい。")
        add("     （順位は不変なので◎選定は変わらず、再学習も不要）")
        add("  → 次に p1/p2 比・AI勝率バケットの実ROIから 🔥勝負/⚠️回避 の閾値を引き直す。")
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
