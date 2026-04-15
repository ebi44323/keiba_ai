"""
Temperature Scaling の最適 T 値を探索するスクリプト

【方針】
  backtest.py と同じカテゴリ処理で全データのスコアを計算し、
  最新 holdout% 期間だけで T を評価する。

【使い方】
  python check_temperature.py [--holdout 0.2]
"""

import os, sys, argparse, zipfile, io, unittest.mock as mock
import numpy as np
import pandas as pd

def _passthrough(func=None, **kw):
    if callable(func): return func
    return lambda f: f

with mock.patch("streamlit.cache_resource", _passthrough), \
     mock.patch("streamlit.cache_data",     _passthrough), \
     mock.patch("streamlit.spinner",        lambda *a, **kw: mock.MagicMock()):
    from src.core_model import prepare_model_and_data
    from src.features_engine import TE_COLS, create_features


def load_csv():
    if os.path.exists("learning_data_perfect_tier.csv"):
        print("CSV読み込み中...")
        return pd.read_csv("learning_data_perfect_tier.csv",
                           encoding="utf-8-sig", low_memory=False)
    if os.path.exists("learning_data_perfect_tier.zip"):
        print("ZIP読み込み中...")
        with zipfile.ZipFile("learning_data_perfect_tier.zip") as z:
            with z.open(z.namelist()[0]) as f:
                return pd.read_csv(io.BytesIO(f.read()),
                                   encoding="utf-8-sig", low_memory=False)
    print("学習データが見つかりません"); sys.exit(1)


def _norm(s):
    mn, mx = s.min(), s.max()
    return (s - mn) / (mx - mn + 1e-9)


def main():
    parser = argparse.ArgumentParser(description="Temperature T値探索")
    parser.add_argument("--holdout", type=float, default=0.2,
                        help="評価に使う末尾割合 default=0.2")
    args = parser.parse_args()

    print("=" * 60)
    print(f"Temperature Scaling T値探索  holdout={args.holdout*100:.0f}%%")
    print("=" * 60)

    # ── モデルロード ──────────────────────────────────────────────
    print("\nモデルロード中...")
    bundle = prepare_model_and_data(force_retrain=False)
    (model, model_win, model_reg, features, cat_features, _num, _catd,
     _lhd, _hcd, _ped, _kj, _kt, te_dicts, global_mean, *_rest) = bundle
    print("完了")

    # ── CSV 読み込み・特徴量生成（core_model.py と同じ前処理）────
    df_raw = load_csv()
    # core_model.py と同様に調教師名を正規化
    if "調教師" in df_raw.columns:
        df_raw["調教師"] = df_raw["調教師"].str.replace(r"^\[.+?\]\s*", "", regex=True)
    print("特徴量生成中（create_features）...")
    df, _ = create_features(df_raw)
    print("完了")

    df["日付"] = pd.to_datetime(df["日付"], errors="coerce")
    df["着順"] = pd.to_numeric(df["着順"], errors="coerce")
    df["単勝"] = pd.to_numeric(df["単勝"], errors="coerce")
    df = df.dropna(subset=["日付", "着順", "単勝", "レースID"])
    try:
        df = df[df["レースID"].astype(str).str[4:6].astype(int) < 11]
    except Exception:
        pass
    print(f"全データ: {df['レースID'].nunique()}レース  {len(df)}馬")

    # ── TE 適用 ──────────────────────────────────────────────────
    for col in TE_COLS:
        if col in df.columns:
            df[f"{col}_TE"] = df[col].map(te_dicts.get(col, {})).fillna(global_mean)

    # ── カテゴリ・数値変換（backtest.py と完全に同じ処理）─────────
    for col in cat_features:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("不明").astype("category")

    avail = [f for f in features if f in df.columns]
    for col in avail:
        if col not in cat_features:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    X = df[avail].copy()
    for col in X.columns:
        if hasattr(X[col], "cat"):
            if "不明" not in X[col].cat.categories:
                X[col] = X[col].cat.add_categories(["不明"])
            X[col] = X[col].fillna("不明")
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

    # ── スコア計算（全データ）────────────────────────────────────
    print("スコア計算中（しばらくかかります）...")
    sa = pd.Series(_norm(model.predict(X)),           index=df.index)
    sb = pd.Series(_norm(model_win.predict(X)),       index=df.index)
    sc = pd.Series(_norm(1.0 - model_reg.predict(X)), index=df.index)
    df["_score"] = sa * 0.0581 + sb * 0.8159 + sc * 0.1261
    print("完了")

    # ── ホールドアウト期間だけで評価 ─────────────────────────────
    cutoff = df["日付"].quantile(1.0 - args.holdout)
    df_eval = df[df["日付"] > cutoff].copy()
    n_races = df_eval["レースID"].nunique()
    print(f"\nホールドアウト期間: {df_eval['日付'].min().date()} 〜 {df_eval['日付'].max().date()}")
    print(f"評価レース数: {n_races}  馬数: {len(df_eval)}")
    if n_races < 50:
        print("レース数不足。--holdout を大きくしてください"); sys.exit(1)

    # ── T ごとに評価 ─────────────────────────────────────────────
    T_values = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0]
    rows = []
    for T in T_values:
        records = []
        for _, rdf in df_eval.groupby("レースID"):
            if len(rdf) < 3:
                continue
            raw  = rdf["_score"].values
            expv = np.exp((raw - raw.max()) / T)
            prob = expv / expv.sum()
            i    = prob.argmax()
            row  = rdf.iloc[i]
            won  = int(row["着順"]) == 1
            records.append({
                "ai_prob": prob[i],
                "won":     won,
                "placed":  int(row["着順"]) <= 3,
                "pay_tan": float(row["単勝"]) * 100 if won else 0.0,
            })

        r        = pd.DataFrame(records)
        n        = len(r)
        avg_ai   = r["ai_prob"].mean()
        win_rate = r["won"].mean()
        gap      = avg_ai - win_rate
        tan_rate = r["pay_tan"].sum() / (n * 100) * 100
        plc_rate = r["placed"].mean() * 100
        rows.append({
            "T":               T,
            "本命平均AI勝率":  f"{avg_ai*100:.1f}%%",
            "本命実勝率":      f"{win_rate*100:.1f}%%",
            "ギャップ":        f"{gap*100:+.1f}%%",
            "単勝回収率":      f"{tan_rate:.1f}%%",
            "本命複勝率(参考)":f"{plc_rate:.1f}%%",
            "レース数":        n,
            "_gap_abs":        abs(gap),
            "_tan_rate":       tan_rate,
        })

    result = pd.DataFrame(rows)
    print("\n" + "=" * 60)
    print(result[["T","本命平均AI勝率","本命実勝率","ギャップ",
                  "単勝回収率","本命複勝率(参考)","レース数"]].to_string(index=False))

    best_c = result.loc[result["_gap_abs"].idxmin()]
    best_r = result.loc[result["_tan_rate"].idxmax()]
    print("\n" + "=" * 60)
    print(f"▶ キャリブレーション最良（ギャップ最小）: T = {best_c['T']}")
    print(f"  ギャップ {best_c['ギャップ']}  単勝回収率 {best_c['単勝回収率']}")
    print(f"▶ 単勝回収率最高: T = {best_r['T']}")
    print(f"  ギャップ {best_r['ギャップ']}  単勝回収率 {best_r['単勝回収率']}")
    print("=" * 60)
    print("""
次のステップ:
  src/inference.py の TEMPERATURE = 1.0 を選んだ値に変更
  （再学習不要・push だけで反映）
""")


if __name__ == "__main__":
    main()
