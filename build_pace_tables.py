"""
build_pace_tables.py — 展開予想テーブルの生成（手動実行・ローカル）

学習データから「想定ペース」と「想定隊列」に必要な統計テーブルを作り、
src/pace_tables.json に保存する。推論側はこのJSONを lookup するだけなので
**モデルの再学習は不要**（draw_course_dict 等と同じ方式）。

使い方:
  python build_pace_tables.py

実測にもとづく設計判断（2026-09-25）:
  - ペース: 想定先行馬数が増えるほど前半が速くなる（相関 -0.227・芝ダ両方で単調）。
    ただし時系列OOSの R² は 芝0.12 / ダート0.057 と弱い。
    → 秒数を断言せず**5段階ラベル**までに留める（精度のない数字を出さない）。
  - 隊列: 前走の位置率 → 今走の位置率 の相関 +0.441（R²0.194）。
    ただし各馬のばらつき SD≈0.27（14頭立てで±3.6頭分）。
    → 点ではなく**帯（中央値±SD）**で提示する。
  - 枠順: 全体では相関 +0.007 とほぼ無効。芝/ダートで符号が逆なうえ、**同じ競馬場でも
    距離で符号が反転する**（東京ダ1600m -0.036＝芝スタートで外が前 / 東京ダ1300m +0.037）。
    → **(競馬場×芝ダ×距離)** を主テーブルにし、不足時のみ (競馬場×芝ダ) へフォールバック。
    効果量は最大 0.187（新潟芝1000＝直線競馬）で、14頭立てなら約2.5頭分と無視できない。
"""

import json
import zipfile
import datetime
import numpy as np
import pandas as pd

ZIP = "learning_data_perfect_tier.zip"
CSV = "learning_data_perfect_tier.csv"
OUT = "src/pace_tables.json"

MIN_RACES_PER_COND = 20     # コース条件の最低レース数
MIN_ROWS_PER_DRAW      = 3000   # 枠効果(競馬場×芝ダ)を採用する最低サンプル
MIN_ROWS_PER_DRAW_DIST = 700    # 枠効果(競馬場×芝ダ×距離)を採用する最低サンプル


def _load():
    z = zipfile.ZipFile(ZIP)
    use = ['レースID', '日付', '競馬場', '芝/ダート', '距離', '馬場', '前半3F',
           '出走頭数', '前走コーナー順位', '最初のコーナー順位', '馬番', '馬ID']
    df = pd.read_csv(z.open(CSV), usecols=use, dtype=str, low_memory=False)
    for c in ['前半3F', '距離', '出走頭数', '前走コーナー順位', '最初のコーナー順位', '馬番']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['日付'] = pd.to_datetime(df['日付'], errors='coerce')
    return df.dropna(subset=['日付'])


def build_pace(df):
    """想定ペース: コース条件のベースライン＋(先行馬数,頭数)の線形補正。"""
    df = df.copy()
    df['cand'] = np.where(df['前走コーナー順位'].isna(), np.nan,
                          (df['前走コーナー順位'] <= 3).astype(float))
    r = df.groupby('レースID').agg(
        pace=('前半3F', 'first'), venue=('競馬場', 'first'), track=('芝/ダート', 'first'),
        dist=('距離', 'first'), n=('出走頭数', 'first'), date=('日付', 'first'),
        lead=('cand', 'sum'), known=('cand', 'count')).reset_index()
    r = r.dropna(subset=['pace', 'dist', 'n'])
    r = r[(r.pace > 9) & (r.pace < 15) & (r.known >= 5)]

    # ベースライン（競馬場×芝ダ×距離）と、カバーできない条件用の段階フォールバック
    base = r.groupby(['venue', 'track', 'dist'])['pace'].agg(['median', 'size'])
    base = base[base['size'] >= MIN_RACES_PER_COND]['median']
    fb_vt = r.groupby(['venue', 'track'])['pace'].median()
    fb_td = r.groupby(['track', 'dist'])['pace'].median()
    fb_t = r.groupby('track')['pace'].median()

    r['mu'] = r.set_index(['venue', 'track', 'dist']).index.map(base)
    fit = r.dropna(subset=['mu']).copy()
    fit['resid'] = fit['pace'] - fit['mu']

    coef, pred_sd, resid_sd = {}, {}, {}
    for td in ('芝', 'ダート'):
        s = fit[fit.track == td]
        if len(s) < 500:
            continue
        X = np.column_stack([np.ones(len(s)), s['lead'], s['n']])
        b = np.linalg.lstsq(X, s['resid'].to_numpy(), rcond=None)[0]
        coef[td] = [round(float(x), 6) for x in b]
        pred_sd[td] = round(float((X @ b).std()), 5)     # ラベル分割の基準
        resid_sd[td] = round(float(s['resid'].std()), 5)
    return dict(
        pace_base={f"{v}|{t}|{int(d)}": round(float(m), 4) for (v, t, d), m in base.items()},
        pace_fb_venue_track={f"{v}|{t}": round(float(m), 4) for (v, t), m in fb_vt.items()},
        pace_fb_track_dist={f"{t}|{int(d)}": round(float(m), 4) for (t, d), m in fb_td.items()},
        pace_fb_track={t: round(float(m), 4) for t, m in fb_t.items()},
        pace_coef=coef, pace_pred_sd=pred_sd, pace_resid_sd=resid_sd,
        pace_races=int(len(fit)),
    )


def build_positions(df):
    """想定隊列: 前走位置率→今走位置率の回帰＋(競馬場×芝ダ×距離)別の枠効果。"""
    d = df.dropna(subset=['馬番', '出走頭数', '最初のコーナー順位',
                          '前走コーナー順位']).copy()
    d = d[d['出走頭数'] >= 8].sort_values(['馬ID', '日付'])
    d['前走頭数'] = d.groupby('馬ID')['出走頭数'].shift(1)
    d = d.dropna(subset=['前走頭数'])
    d = d[d['前走頭数'] >= 8]
    d['y'] = (d['最初のコーナー順位'] - 1) / (d['出走頭数'] - 1)
    d['prev'] = (d['前走コーナー順位'] - 1) / (d['前走頭数'] - 1)
    d['draw'] = (d['馬番'] - 1) / (d['出走頭数'] - 1)
    d = d[d.y.between(0, 1) & d.prev.between(0, 1)]

    X = np.column_stack([np.ones(len(d)), d['prev']])
    b = np.linalg.lstsq(X, d.y.to_numpy(), rcond=None)[0]
    resid = d.y.to_numpy() - X @ b

    # 枠効果: 内枠と外枠の残差の差（正=外枠ほど後方 / 負=外枠ほど前）。
    #
    # ★2026-09-25 修正: 当初は (競馬場×芝ダ) 別だったが、**距離別に符号が反転する**ため
    #   平均すると情報が消えていた。実測:
    #     東京ダ1600m -0.036（外ほど前・芝スタート） vs 東京ダ1300m +0.037（外ほど後方）
    #     → (競馬場×芝ダ) にまとめると 東京ダート +0.007 ≒ ゼロ になっていた。
    #   外枠有利なダートは**芝スタートのコース**（東京ダ1600・中京ダ1400・阪神ダ2000 等）に
    #   集中しており、ダート発走のコースは逆に外枠が後方になる。
    #   芝も距離差が大きい（中京芝1600 +0.117 / 新潟芝1000 -0.187＝直線競馬で外有利）。
    #   → (競馬場×芝ダ×距離) を主テーブルにし、サンプル不足時のみ (競馬場×芝ダ) へ落とす。
    d['_resid'] = resid
    d['_dist'] = d['距離'].astype('Int64')

    def _eff(s):
        inn = s[s.draw <= .25]['_resid'].mean()
        out = s[s.draw >= .75]['_resid'].mean()
        if np.isnan(inn) or np.isnan(out):
            return None
        return round(float(out - inn), 4)

    draw_dist = {}
    for (v, t, dd), s in d.groupby(['競馬場', '芝/ダート', '_dist']):
        if len(s) < MIN_ROWS_PER_DRAW_DIST:
            continue
        e = _eff(s)
        if e is not None:
            draw_dist[f"{v}|{t}|{int(dd)}"] = e

    draw = {}
    for (v, t), s in d.groupby(['競馬場', '芝/ダート']):
        if len(s) < MIN_ROWS_PER_DRAW:
            continue
        e = _eff(s)
        if e is not None:
            draw[f"{v}|{t}"] = e

    return dict(
        pos_intercept=round(float(b[0]), 5),
        pos_slope=round(float(b[1]), 5),
        pos_sd=round(float(resid.std()), 5),
        pos_rows=int(len(d)),
        draw_effect_dist=draw_dist,
        draw_effect=draw,
    )


def main():
    print("学習データ読込中...")
    df = _load()
    print(f"  {len(df):,} 行")
    tables = {"generated": datetime.date.today().isoformat()}
    print("想定ペースのテーブル生成...")
    tables.update(build_pace(df))
    print(f"  コース条件 {len(tables['pace_base'])} 件 / 係数 {tables['pace_coef']}")
    print("想定隊列のテーブル生成...")
    tables.update(build_positions(df))
    print(f"  位置回帰: y = {tables['pos_intercept']:.3f} + {tables['pos_slope']:.3f}×前走位置率 "
          f"(SD {tables['pos_sd']:.3f})")
    print(f"  枠効果: 距離別 {len(tables['draw_effect_dist'])} 条件 / "
          f"フォールバック {len(tables['draw_effect'])} コース")
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(tables, f, ensure_ascii=False, indent=1)
    print(f"✅ {OUT} を保存しました")


if __name__ == "__main__":
    main()
