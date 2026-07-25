# -*- coding: utf-8 -*-
"""
validate_model.py — 新モデル（絶対スコア勝率）のデータドリブン検証

本番と同じ特徴量パイプライン（create_features + 本番モデルパラメータ）を使い、
リークなしの時系列ウォークフォワードで「旧（レース内min-max・T=1.5）」と
「新（絶対スコア・T=1.0）」を同一データ上で比較する。

出力する指標:
  1. キャリブレーション: 予測勝率ビンごとの予測 vs 実際の勝率（+ ECE）
  2. 頭数別: 本命(top1)の的中率・単勝回収率（小頭数の過大評価を直接検証）
  3. 全体: 本命的中率・単勝回収率・複勝的中率・本命平均AI勝率 vs 実勝率（過信度）

使い方:
  python validate_model.py                 # 既定: 3fold × 直近30日、全履歴で学習
  python validate_model.py --splits 3 --test-days 30 --max-train-years 3
"""
import argparse
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from src.features_engine import NUM_FEATURES, CAT_FEATURES, TE_COLS, create_features

# ── 本番(core_model.py)と一致するモデルパラメータ ──────────────────────────
PARAMS_A = dict(n_estimators=500, learning_rate=0.01, num_leaves=63, max_bin=255,
                cat_smooth=10, random_state=42, importance_type='gain',
                colsample_bytree=0.7, subsample=0.8)
PARAMS_B = dict(n_estimators=685, learning_rate=0.029074, num_leaves=15, max_bin=228,
                cat_smooth=39.9527, colsample_bytree=0.8848, subsample=0.7033,
                min_child_samples=70, random_state=123, importance_type='gain')
PARAMS_C = dict(n_estimators=300, learning_rate=0.03, num_leaves=31, max_bin=255,
                random_state=777, colsample_bytree=0.7, subsample=0.8)
W_A, W_B, W_C = 0.0581, 0.8159, 0.1261


def _load_features():
    """学習データを読み込み create_features 適用済みの df を返す。"""
    try:
        df = pd.read_csv('learning_data_perfect_tier.zip', compression='zip', dtype=str)
    except FileNotFoundError:
        df = pd.read_csv('learning_data_perfect_tier.csv', dtype=str)
    if '調教師' in df.columns:
        df['調教師'] = df['調教師'].str.replace(r'^\[.+?\]\s*', '', regex=True)
    df, _ = create_features(df)
    return df


def _prep(df):
    """数値/カテゴリ型変換・ラベル生成。"""
    df = df.dropna(subset=['着順', '単勝', 'レースID', '日付']).copy()
    df['日付'] = pd.to_datetime(df['日付'], errors='coerce')
    df['着順'] = pd.to_numeric(df['着順'], errors='coerce')
    df['単勝'] = pd.to_numeric(df['単勝'], errors='coerce')
    df = df.dropna(subset=['着順', '単勝', '日付'])
    df['馬券内'] = (df['着順'] <= 3).astype(int)
    df['win_label'] = (df['着順'] == 1).astype(int)
    for col in NUM_FEATURES:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors='coerce')
    for col in CAT_FEATURES:
        if col not in df.columns:
            df[col] = '不明'
        df[col] = df[col].astype(str).fillna('不明').astype('category')
    return df


def _fit_predict_fold(train_df, test_df, features, cat_features):
    """1foldぶんの A/B/C を学習し、train/test の生スコアを返す。"""
    tr_groups = train_df.groupby('レースID', sort=False).size().values
    te_groups = test_df.groupby('レースID', sort=False).size().values
    catf = [f for f in cat_features if f in features]

    mA = lgb.LGBMRanker(**PARAMS_A)
    mA.fit(train_df[features], train_df['馬券内'], group=tr_groups,
           categorical_feature=catf,
           eval_set=[(test_df[features], test_df['馬券内'])], eval_group=[te_groups])
    mB = lgb.LGBMRanker(**PARAMS_B)
    mB.fit(train_df[features], train_df['win_label'], group=tr_groups,
           categorical_feature=catf,
           eval_set=[(test_df[features], test_df['win_label'])], eval_group=[te_groups])
    mC = lgb.LGBMRegressor(**PARAMS_C)
    mC.fit(train_df[features], train_df['着順パーセント'].fillna(0.5),
           categorical_feature=catf)

    def _raw(d):
        return (np.asarray(mA.predict(d[features]), dtype=float),
                np.asarray(mB.predict(d[features]), dtype=float),
                1.0 - np.asarray(mC.predict(d[features]), dtype=float))
    return _raw(train_df), _raw(test_df)


def _softmax_by_race(score, race_ids, temp):
    s = pd.Series(score, index=race_ids.index)
    g = s.groupby(race_ids)
    e = np.exp((s - g.transform('max')) / temp)
    return (e / e.groupby(race_ids).transform('sum')).values


def _ai_winprob(raw_tr, raw_te, test_df, mode):
    """正規化モードに応じて test の AI勝率(レース内softmax)を返す。"""
    (a_tr, b_tr, c_tr), (a_te, b_te, c_te) = raw_tr, raw_te
    rid = test_df['レースID']
    if mode == 'absolute':
        def norm(x_tr, x_te):
            lo, hi = np.percentile(x_tr, 1), np.percentile(x_tr, 99)
            return np.clip((x_te - lo) / (hi - lo + 1e-9), 0.0, 1.0)
        sa, sb, sc = norm(a_tr, a_te), norm(b_tr, b_te), norm(c_tr, c_te)
        temp = 1.0
    else:  # race_minmax（旧・推論はレース内min-max + T=1.5）
        def norm_race(x):
            s = pd.Series(x, index=rid.index)
            g = s.groupby(rid)
            return ((s - g.transform('min')) / (g.transform('max') - g.transform('min') + 1e-9)).values
        sa, sb, sc = norm_race(a_te), norm_race(b_te), norm_race(c_te)
        temp = 1.5
    ens = sa * W_A + sb * W_B + sc * W_C
    return _softmax_by_race(ens, rid, temp)


def _walk_forward(df, features, cat_features, n_splits, test_days, max_train_years):
    df = df.sort_values('日付').reset_index(drop=True)
    max_date = df['日付'].max()
    rows = []
    for i in range(n_splits):
        test_end = max_date - pd.Timedelta(days=test_days * i)
        test_start = test_end - pd.Timedelta(days=test_days)
        train_mask = df['日付'] < test_start
        if max_train_years:
            train_mask &= df['日付'] >= (test_start - pd.Timedelta(days=365 * max_train_years))
        test_mask = (df['日付'] >= test_start) & (df['日付'] <= test_end)
        train_df = df[train_mask].copy()
        test_df = df[test_mask].copy()
        if len(train_df) == 0 or len(test_df) == 0:
            continue
        print(f"[fold {i+1}] train={len(train_df):,} / test={len(test_df):,} "
              f"({test_start.date()}〜{test_end.date()})", flush=True)

        # TE（train のみで算出）
        local_features = list(features)
        gm = train_df['馬券内'].mean()
        for col in TE_COLS:
            if col in train_df.columns:
                te = train_df.groupby(col)['馬券内'].mean().to_dict()
                train_df[f'{col}_TE'] = train_df[col].map(te).fillna(gm)
                test_df[f'{col}_TE'] = test_df[col].map(te).fillna(gm)
                if f'{col}_TE' not in local_features:
                    local_features.append(f'{col}_TE')

        raw_tr, raw_te = _fit_predict_fold(train_df, test_df, local_features, cat_features)
        out = test_df[['レースID', '着順', '単勝']].copy()
        out['出走頭数'] = out.groupby('レースID')['着順'].transform('size')
        out['p_old'] = _ai_winprob(raw_tr, raw_te, test_df, 'race_minmax')
        out['p_new'] = _ai_winprob(raw_tr, raw_te, test_df, 'absolute')
        out['fold'] = i
        rows.append(out)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ── 指標 ────────────────────────────────────────────────────────────────────
def _calibration(res, pcol, bins=10):
    r = res.dropna(subset=[pcol]).copy()
    r['win'] = (r['着順'] == 1).astype(int)
    r['bin'] = pd.qcut(r[pcol], q=bins, duplicates='drop')
    g = r.groupby('bin', observed=True)
    tbl = g.agg(pred=(pcol, 'mean'), actual=('win', 'mean'), n=(pcol, 'size'))
    ece = (np.abs(tbl['pred'] - tbl['actual']) * tbl['n']).sum() / tbl['n'].sum()
    return tbl, ece


def _top1(res, pcol):
    idx = res.groupby('レースID')[pcol].idxmax()
    t = res.loc[idx].copy()
    t['win'] = (t['着順'] == 1).astype(int)
    t['place'] = (t['着順'] <= 3).astype(int)
    t['payout'] = np.where(t['win'] == 1, t['単勝'] * 100, 0.0)
    return t


def _by_fieldsize(res, pcol):
    t = _top1(res, pcol)
    def bucket(n):
        if n <= 7: return '① 5-7頭'
        if n <= 12: return '② 8-12頭'
        return '③ 13頭+'
    t['層'] = t['出走頭数'].map(bucket)
    g = t.groupby('層')
    return g.agg(R数=('win', 'size'), 的中率=('win', 'mean'),
                 単勝回収率=('payout', lambda s: s.mean()),
                 平均本命AI勝率=(pcol, 'mean'))


def _ev_top1(res, pcol, floor_mode, ev_threshold=1.5):
    """EV優先◎を選ぶ（inference.pyと同じ: 候補なしは標準◎にフォールバック）。
    floor_mode='flat'(旧: 0.18固定) / 'dynamic'(新: max(0.25,1.4/N))。
    単勝(結果オッズ)を live オッズの代理として EV=AI勝率×単勝 を使う。"""
    r = res.dropna(subset=[pcol, '単勝']).copy()
    r['EV'] = r[pcol] * r['単勝']
    if floor_mode == 'flat':
        floor = 0.18
    else:
        floor = np.maximum(0.25, 1.4 / r['出走頭数'].clip(lower=1))
    r['ok'] = (r['EV'] >= ev_threshold) & (r[pcol] >= floor)
    picks = []
    for rid, g in r.groupby('レースID'):
        cand = g[g['ok']]
        row = cand.loc[cand['EV'].idxmax()] if not cand.empty else g.loc[g[pcol].idxmax()]
        picks.append(row)
    t = pd.DataFrame(picks)
    t['win'] = (t['着順'] == 1).astype(int)
    t['payout'] = np.where(t['win'] == 1, t['単勝'] * 100, 0.0)
    t['人気薄'] = (t['単勝'] >= 10.0).astype(int)   # 10倍以上を人気薄と定義
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--splits', type=int, default=3)
    ap.add_argument('--test-days', type=int, default=30)
    ap.add_argument('--max-train-years', type=int, default=None,
                    help='学習を直近N年に制限（未指定=全履歴）')
    ap.add_argument('--out', default='validation_report.csv')
    args = ap.parse_args()

    print('学習データ読込 + 特徴量生成 ...', flush=True)
    df = _prep(_load_features())
    features = [f for f in (CAT_FEATURES + NUM_FEATURES) if f in df.columns]
    print(f'  総レコード={len(df):,} / 特徴量={len(features)}', flush=True)

    res = _walk_forward(df, features, CAT_FEATURES, args.splits, args.test_days,
                        args.max_train_years)
    if res.empty:
        print('検証データが空です。'); return
    res.to_csv(args.out, index=False, encoding='utf-8-sig')

    n_races = res['レースID'].nunique()
    print('\n' + '=' * 68)
    print(f'  検証結果  （{n_races:,}レース / {len(res):,}頭 のアウトオブサンプル）')
    print('=' * 68)

    for label, pcol in [('旧: レース内min-max + T=1.5', 'p_old'),
                        ('新: 絶対スコア + T=1.0', 'p_new')]:
        t = _top1(res, pcol)
        tbl, ece = _calibration(res, pcol)
        print(f'\n■ {label}')
        print(f'  本命的中率        : {t["win"].mean()*100:5.1f}%')
        print(f'  本命単勝回収率    : {t["payout"].mean():5.1f}%')
        print(f'  本命複勝的中率    : {t["place"].mean()*100:5.1f}%')
        print(f'  本命平均AI勝率    : {t[pcol].mean()*100:5.1f}%  (実際の勝率 {t["win"].mean()*100:.1f}% ← 差が小さいほど良い=過信が少ない)')
        print(f'  キャリブレーション誤差(ECE): {ece*100:.2f}%p  (小さいほど良い)')
        print('  頭数別:')
        fb = _by_fieldsize(res, pcol)
        for 層, row in fb.iterrows():
            print(f'    {層}: R数={int(row["R数"]):4d}  的中率={row["的中率"]*100:5.1f}%  '
                  f'単勝回収={row["単勝回収率"]:6.1f}%  平均本命AI勝率={row["平均本命AI勝率"]*100:5.1f}%')

    # ── EV優先◎の新旧比較（#1/#6の核心）──────────────────────────────
    print('\n' + '=' * 68)
    print('  EV優先◎の比較（#1 小頭数で人気薄◎ / #6 の効き所）')
    print('=' * 68)
    for label, pcol, fmode in [('旧EV: p=min-max, フロア0.18固定', 'p_old', 'flat'),
                               ('新EV: p=絶対, フロア=max(0.25,1.4/N)', 'p_new', 'dynamic')]:
        t = _ev_top1(res, pcol, fmode)
        small = t[t['出走頭数'] <= 9]
        print(f'\n■ {label}')
        print(f'  EV◎的中率        : {t["win"].mean()*100:5.1f}%')
        print(f'  EV◎単勝回収率    : {t["payout"].mean():6.1f}%')
        print(f'  EV◎が人気薄(10倍+): {t["人気薄"].mean()*100:5.1f}%  '
              f'(全{len(t)}R中 {int(t["人気薄"].sum())}R)')
        print(f'  └ 小頭数(≤9頭)での人気薄◎率: {small["人気薄"].mean()*100:5.1f}%  '
              f'(全{len(small)}R中 {int(small["人気薄"].sum())}R) ← #1の直接指標')

    print(f'\n詳細は {args.out} に保存しました。')
    print('※ 回収率はサンプル依存でぶれます。ECE(校正)・頭数別AI勝率・小頭数の人気薄◎率を主指標に。')


if __name__ == '__main__':
    main()
