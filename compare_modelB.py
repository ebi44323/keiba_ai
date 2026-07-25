# -*- coding: utf-8 -*-
"""
compare_modelB.py — モデルB パラメータの直接対決（同一ウォークフォワードfold）

CVスコアはデータ期間に依存し比較不能なため、現行パラメータと Optuna 新パラメータを
「同じfold」で学習・評価し、どちらが本当に良いかを客観的に判定する。

指標:
  - モデルB単体AUC（Optunaが最適化している値・リーク無しOOS）
  - アンサンブル◎の的中率 / 単勝回収率 / キャリブレーション誤差(ECE)

使い方:
  python compare_modelB.py --splits 4 --test-days 30 --max-train-years 3
"""
import argparse
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import validate_model as V  # _load_features, _prep, PARAMS_A, PARAMS_C, W_A/B/C, _softmax_by_race, _top1, _calibration
from src.features_engine import CAT_FEATURES, NUM_FEATURES, TE_COLS

# ── 対決するモデルBパラメータ ────────────────────────────────────────────────
B_CURRENT = dict(n_estimators=685, learning_rate=0.029074, num_leaves=15, max_bin=228,
                 cat_smooth=39.9527, colsample_bytree=0.8848, subsample=0.7033,
                 min_child_samples=70, random_state=123, importance_type='gain')
B_NEW = dict(n_estimators=344, learning_rate=0.025325, num_leaves=37, max_bin=217,
             cat_smooth=46.1425, colsample_bytree=0.8668, subsample=0.7016,
             min_child_samples=73, random_state=123, importance_type='gain')


def _norm_abs(x_tr, x_te):
    lo, hi = np.percentile(x_tr, 1), np.percentile(x_tr, 99)
    return np.clip((x_te - lo) / (hi - lo + 1e-9), 0.0, 1.0)


def _winprob(raw, rid):
    (a_tr, a_te), (b_tr, b_te), (c_tr, c_te) = raw
    sa, sb, sc = _norm_abs(a_tr, a_te), _norm_abs(b_tr, b_te), _norm_abs(c_tr, c_te)
    ens = sa * V.W_A + sb * V.W_B + sc * V.W_C
    return V._softmax_by_race(ens, rid, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--splits', type=int, default=4)
    ap.add_argument('--test-days', type=int, default=30)
    ap.add_argument('--max-train-years', type=int, default=3)
    args = ap.parse_args()

    print('学習データ読込 + 特徴量生成 ...', flush=True)
    df = V._prep(V._load_features())
    df = df.sort_values('日付').reset_index(drop=True)
    features = [f for f in (CAT_FEATURES + NUM_FEATURES) if f in df.columns]
    catf = [f for f in CAT_FEATURES if f in features]
    max_date = df['日付'].max()

    rows = []
    aucB = {'current': [], 'new': []}   # (auc, n)
    for i in range(args.splits):
        test_end = max_date - pd.Timedelta(days=args.test_days * i)
        test_start = test_end - pd.Timedelta(days=args.test_days)
        tr_mask = (df['日付'] < test_start) & (df['日付'] >= test_start - pd.Timedelta(days=365 * args.max_train_years))
        te_mask = (df['日付'] >= test_start) & (df['日付'] <= test_end)
        train_df, test_df = df[tr_mask].copy(), df[te_mask].copy()
        if train_df.empty or test_df.empty:
            continue
        print(f"[fold {i+1}] train={len(train_df):,} test={len(test_df):,} "
              f"({test_start.date()}〜{test_end.date()})", flush=True)

        local = list(features)
        gm = train_df['馬券内'].mean()
        for col in TE_COLS:
            if col in train_df.columns:
                te = train_df.groupby(col)['馬券内'].mean().to_dict()
                train_df[f'{col}_TE'] = train_df[col].map(te).fillna(gm)
                test_df[f'{col}_TE'] = test_df[col].map(te).fillna(gm)
                if f'{col}_TE' not in local:
                    local.append(f'{col}_TE')

        tg = train_df.groupby('レースID', sort=False).size().values
        eg = test_df.groupby('レースID', sort=False).size().values
        cf = [f for f in catf if f in local]

        # A, C は共通（1回だけ学習）
        mA = lgb.LGBMRanker(**V.PARAMS_A)
        mA.fit(train_df[local], train_df['馬券内'], group=tg, categorical_feature=cf,
               eval_set=[(test_df[local], test_df['馬券内'])], eval_group=[eg])
        mC = lgb.LGBMRegressor(**V.PARAMS_C)
        mC.fit(train_df[local], train_df['着順パーセント'].fillna(0.5), categorical_feature=cf)
        a_tr, a_te = mA.predict(train_df[local]), mA.predict(test_df[local])
        c_tr = 1.0 - mC.predict(train_df[local]); c_te = 1.0 - mC.predict(test_df[local])

        out = test_df[['レースID', '着順', '単勝', 'win_label']].copy()
        out['出走頭数'] = out.groupby('レースID')['着順'].transform('size')
        rid = test_df['レースID']

        for tag, params in [('current', B_CURRENT), ('new', B_NEW)]:
            mB = lgb.LGBMRanker(**params)
            mB.fit(train_df[local], train_df['win_label'], group=tg, categorical_feature=cf,
                   eval_set=[(test_df[local], test_df['win_label'])], eval_group=[eg])
            b_tr = np.asarray(mB.predict(train_df[local]), float)
            b_te = np.asarray(mB.predict(test_df[local]), float)
            out[f'p_{tag}'] = _winprob(((a_tr, a_te), (b_tr, b_te), (c_tr, c_te)), rid)
            try:
                aucB[tag].append((roc_auc_score(test_df['win_label'], b_te), len(test_df)))
            except Exception:
                pass
        rows.append(out)

    res = pd.concat(rows, ignore_index=True)
    n_races = res['レースID'].nunique()

    def _wauc(pairs):
        num = sum(a * n for a, n in pairs); den = sum(n for _, n in pairs)
        return num / den if den else 0.0

    print('\n' + '=' * 66)
    print(f'  モデルB パラメータ直接対決  （{n_races:,}レース / {len(res):,}頭 OOS・同一fold）')
    print('=' * 66)
    for tag, label in [('current', '現行 (n=685, leaves=15)'),
                       ('new', 'Optuna新 (n=344, leaves=37)')]:
        pcol = f'p_{tag}'
        t = V._top1(res, pcol)
        _, ece = V._calibration(res, pcol)
        print(f'\n■ {label}')
        print(f'  モデルB AUC(OOS) : {_wauc(aucB[tag]):.4f}   ← Optunaが最適化する値')
        print(f'  ◎的中率          : {t["win"].mean()*100:5.1f}%')
        print(f'  ◎単勝回収率      : {t["payout"].mean():6.1f}%')
        print(f'  ◎複勝的中率      : {t["place"].mean()*100:5.1f}%')
        print(f'  キャリブレーション誤差(ECE): {ece*100:.2f}%p')

    # 勝敗サマリ
    ac, an = _wauc(aucB['current']), _wauc(aucB['new'])
    roi_c = V._top1(res, 'p_current')['payout'].mean()
    roi_n = V._top1(res, 'p_new')['payout'].mean()
    auc_win = '新の勝ち' if an > ac else ('現行の勝ち' if ac > an else '引分')
    roi_win = '新の勝ち' if roi_n > roi_c else ('現行の勝ち' if roi_c > roi_n else '引分')
    print('\n' + '-' * 66)
    print('判定（同一foldなので直接比較可能）:')
    print(f'  AUC        : 現行 {ac:.4f}  vs  新 {an:.4f}  → {auc_win}')
    print(f'  単勝回収率 : 現行 {roi_c:.1f}%  vs  新 {roi_n:.1f}%  → {roi_win}')
    print('※ AUCと回収率の両方で新が明確に上回るなら貼り替え価値あり。拮抗/劣後なら現行据え置き。')


if __name__ == '__main__':
    main()
