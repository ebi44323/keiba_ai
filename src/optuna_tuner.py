"""
optuna_tuner.py - リーク防止版 Optunaハイパーパラメータ最適化
================================================================
【修正内容】
  - コース統計は features_engine の expanding window 版を使用（リーク消滅）
  - ウォークフォワードCV (3分割) で汎化性を評価
  - TE は各foldのtrainデータのみから計算
  - 評価指標: 回収率のCV平均（単一期間の過適合を防ぐ）
================================================================
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import logging

logger = logging.getLogger('keiba_ebye')


def _compute_te_for_fold(train_df, test_df, te_cols, target_col='馬券内'):
    """foldごとにTarget Encodingを計算（訓練データのみから）"""
    global_mean = train_df[target_col].mean()
    for col in te_cols:
        if col not in train_df.columns:
            continue
        te_dict = train_df.groupby(col)[target_col].mean().to_dict()
        train_df[f'{col}_TE'] = train_df[col].map(te_dict).fillna(global_mean)
        test_df[f'{col}_TE']  = test_df[col].map(te_dict).fillna(global_mean)
    return train_df, test_df


def _calc_auc(test_df, pred_col='optuna_pred'):
    """AUC（1着予測精度）を計算。0.5=ランダム、0.7以上で有効、0.8以上で優秀。"""
    from sklearn.metrics import roc_auc_score
    try:
        y_true = (pd.to_numeric(test_df['着順'], errors='coerce') == 1).astype(int)
        if y_true.sum() == 0:
            return 0.5
        return float(roc_auc_score(y_true, test_df[pred_col]))
    except Exception:
        return 0.5


def _calc_return_rate(test_df, pred_col='optuna_pred'):
    """EV優先◎の単勝回収率を計算（0〜200%程度、100%=トントン）。
    EV=予測スコア×単勝オッズ が最大の馬を◎とする。
    """
    try:
        df = test_df.copy()
        df['_odds'] = pd.to_numeric(df.get('単勝', None), errors='coerce').fillna(10.0)
        df['_ev']   = df[pred_col] * df['_odds']
        df['_rank'] = df['着順'].apply(lambda x: pd.to_numeric(x, errors='coerce'))
        # レースごとに EV最大の馬を◎として単勝回収率を計算
        total_invest = 0
        total_return = 0
        for _, grp in df.groupby('レースID'):
            if grp.empty: continue
            honmei_idx = grp['_ev'].idxmax()
            honmei = grp.loc[honmei_idx]
            total_invest += 100
            if honmei['_rank'] == 1:
                total_return += honmei['_odds'] * 100
        if total_invest == 0:
            return 0.0
        return total_return / total_invest  # 1.0 = 100%
    except Exception:
        return 0.0


def run_optuna_tuning(df, features, cat_features, te_cols,
                     n_trials=50, n_folds=3, fold_days=60,
                     exclude_features=None,
                     auc_weight=0.7, return_weight=0.3):
    """
    ウォークフォワードCV + リーク防止版 Optunaチューニング。

    Parameters
    ----------
    n_trials      : int    試行回数（多いほど良い結果、時間もかかる）
    n_folds       : int    交差検証の分割数（3〜5推奨）
    fold_days     : int    各検証期間の日数
    auc_weight    : float  目的関数内のAUCウェイト (0〜1)
    return_weight : float  目的関数内の単勝回収率ウェイト (0〜1)
                           auc_weight + return_weight は 1.0 が標準

    Returns
    -------
    (best_params, summary_str, cv_results_df)
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        return None, "Optunaがインストールされていません (`pip install optuna`)", None

    # exclude_features: チューニングから除外する特徴量リスト
    # 例: ['市場勝率'] → オッズ依存を排除した真の予測力でチューニング
    if exclude_features:
        features = [f for f in features if f not in exclude_features]

    df = df.copy()
    if '馬券内' not in df.columns:
        df['馬券内'] = (df['着順'] <= 3).astype(int)
    if 'win_label' not in df.columns:
        df['win_label'] = (df['着順'] == 1).astype(int)

    for col in cat_features:
        if col not in df.columns:
            df[col] = '不明'
        df[col] = df[col].astype(str).fillna('不明').astype('category')

    for f in features:
        if f not in cat_features and f in df.columns:
            df[f] = pd.to_numeric(df[f], errors='coerce')

    df = df.sort_values('日付').reset_index(drop=True)
    max_date = df['日付'].max()

    # ── ウォークフォワードCV の fold 定義 ────────────────────────
    folds = []
    for i in range(n_folds):
        test_end   = max_date - pd.Timedelta(days=fold_days * i)
        test_start = test_end  - pd.Timedelta(days=fold_days)
        train_mask = df['日付'] < test_start
        test_mask  = (df['日付'] >= test_start) & (df['日付'] <= test_end)
        train_df = df[train_mask].copy()
        test_df  = df[test_mask].copy()
        if len(train_df) < 500 or len(test_df) < 50:
            logger.info(f'Optuna fold {i+1}: データ不足のためスキップ')
            continue
        folds.append((i, train_df, test_df))

    if not folds:
        return None, "有効なfoldが作れませんでした（データが少なすぎます）", None

    logger.info(f'Optuna: {len(folds)}fold x {n_trials}試行 で最適化開始')

    def objective(trial):
        params = {
            'n_estimators':     trial.suggest_int('n_estimators', 100, 700),
            'learning_rate':    trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
            'num_leaves':       trial.suggest_int('num_leaves', 10, 100),
            'max_bin':          trial.suggest_int('max_bin', 100, 255),
            'cat_smooth':       trial.suggest_float('cat_smooth', 1.0, 50.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.95),
            'subsample':        trial.suggest_float('subsample', 0.5, 0.95),
            'min_child_samples':trial.suggest_int('min_child_samples', 10, 100),
            'random_state': 42,
            'verbose': -1,
        }

        fold_scores = []
        for fold_idx, train_df, test_df in folds:
            _local_feat = [f for f in features if f in train_df.columns]
            needed_cols = list(set(_local_feat + te_cols +
                                   ['レースID', '馬券内', 'win_label', '着順', '単勝']))
            needed_cols = [c for c in needed_cols if c in train_df.columns]

            tr = train_df[needed_cols].copy()
            te = test_df[[c for c in needed_cols if c in test_df.columns]].copy()

            # TE をこのfoldのtrainのみから計算
            tr, te = _compute_te_for_fold(tr, te, te_cols)

            te_feat = [f'{c}_TE' for c in te_cols if f'{c}_TE' in tr.columns]
            local_features = _local_feat + [f for f in te_feat if f not in _local_feat]
            local_features = [f for f in local_features if f in tr.columns and f in te.columns]
            cat_feats = [f for f in cat_features if f in local_features]

            for col in cat_feats:
                tr[col] = tr[col].astype('category')
                te[col] = pd.Categorical(te[col].astype(str),
                                         categories=tr[col].cat.categories)

            tr_groups = tr.groupby('レースID', sort=False).size().values
            te_groups = te.groupby('レースID', sort=False).size().values

            try:
                m = lgb.LGBMRanker(**params)
                m.fit(
                    tr[local_features], tr['win_label'],
                    group=tr_groups,
                    categorical_feature=cat_feats,
                    eval_set=[(te[local_features], te['win_label'])],
                    eval_group=[te_groups],
                    callbacks=[lgb.early_stopping(30, verbose=False),
                               lgb.log_evaluation(-1)]
                )
                preds = m.predict(te[local_features])
                te = te.copy()
                te['optuna_pred'] = preds
                auc  = _calc_auc(te)
                rr   = _calc_return_rate(te)
                # rr を 0〜1 に正規化（想定レンジ 0〜2.0）してブレンド
                rr_norm = min(rr / 2.0, 1.0)
                fold_scores.append(auc_weight * auc + return_weight * rr_norm)
            except Exception as e:
                logger.debug(f'Optuna trial fold {fold_idx} エラー: {e}')
                fold_scores.append(0.0)

        return float(np.mean(fold_scores)) if fold_scores else 0.0

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_score  = study.best_value

    cv_df = pd.DataFrame([
        {'trial': t.number, 'cv_score': t.value, **t.params}
        for t in study.trials if t.value is not None
    ]).sort_values('cv_score', ascending=False).reset_index(drop=True)

    excluded_str = f"  除外特徴量: {exclude_features}\n" if exclude_features else ""
    summary = (
        f"チューニング完了 ({len(folds)}fold ウォークフォワードCV)\n"
        f"  最適CVスコア: {best_score:.4f}  (AUC×{auc_weight} + 回収率×{return_weight}, 試行数: {n_trials})\n"
        f"{excluded_str}"
        f"  n_estimators:      {best_params.get('n_estimators')}\n"
        f"  learning_rate:     {best_params.get('learning_rate'):.6f}\n"
        f"  num_leaves:        {best_params.get('num_leaves')}\n"
        f"  max_bin:           {best_params.get('max_bin')}\n"
        f"  cat_smooth:        {best_params.get('cat_smooth'):.2f}\n"
        f"  colsample_bytree:  {best_params.get('colsample_bytree'):.4f}\n"
        f"  subsample:         {best_params.get('subsample'):.4f}\n"
        f"  min_child_samples: {best_params.get('min_child_samples')}\n"
        f"  ※ スコア = AUC×{auc_weight} + 正規化回収率×{return_weight} / AUC 0.7以上で有効"
    )

    logger.info(summary)
    return best_params, summary, cv_df


def run_weight_optimization(df_scored, n_trials=200):
    """
    アンサンブル重み (wa, wb, wc) の最適化。

    Parameters
    ----------
    df_scored : DataFrame
        以下の列を含む: レースID, 着順, 単勝,
        score_a_norm, score_b_norm, score_c_norm
    n_trials  : int  試行回数

    Returns
    -------
    (best_weights_dict, summary_str, trials_df)
      best_weights_dict = {'wa': float, 'wb': float, 'wc': float}
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        return None, "Optunaがインストールされていません", None

    required = ['レースID', '着順', '単勝', 'score_a_norm', 'score_b_norm', 'score_c_norm']
    missing = [c for c in required if c not in df_scored.columns]
    if missing:
        return None, f"必要列が不足: {missing}", None

    df = df_scored[required].copy()
    df['着順'] = pd.to_numeric(df['着順'], errors='coerce')
    df['単勝'] = pd.to_numeric(df['単勝'], errors='coerce').fillna(10.0)

    def objective(trial):
        wa = trial.suggest_float('wa', 0.0, 1.0)
        wb = trial.suggest_float('wb', 0.0, 1.0 - wa)
        wc = 1.0 - wa - wb
        if wc < 0:
            return 0.0

        df2 = df.copy()
        df2['combo'] = wa * df2['score_a_norm'] + wb * df2['score_b_norm'] + wc * df2['score_c_norm']

        # EV優先◎ の単勝回収率を計算
        total_invest = 0
        total_return = 0
        for _, grp in df2.groupby('レースID'):
            if grp.empty:
                continue
            odds = grp['単勝'].fillna(10.0)
            ev   = grp['combo'] * odds
            best_idx = ev.idxmax()
            total_invest += 100
            if grp.loc[best_idx, '着順'] == 1:
                total_return += grp.loc[best_idx, '単勝'] * 100

        return total_return / total_invest if total_invest > 0 else 0.0

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    bp = study.best_params
    wa = bp['wa']
    wb = bp['wb']
    wc = 1.0 - wa - wb
    best_score = study.best_value

    best_weights = {'wa': round(wa, 4), 'wb': round(wb, 4), 'wc': round(max(wc, 0.0), 4)}

    trials_df = pd.DataFrame([
        {'trial': t.number, '回収率': t.value, 'wa': t.params.get('wa'), 'wb': t.params.get('wb'),
         'wc': 1.0 - t.params.get('wa', 0) - t.params.get('wb', 0)}
        for t in study.trials if t.value is not None
    ]).sort_values('回収率', ascending=False).reset_index(drop=True)

    summary = (
        f"アンサンブル重み最適化完了 ({n_trials}試行)\n"
        f"  最適EV回収率: {best_score*100:.1f}%\n"
        f"  wa (モデルA 複勝): {wa:.4f}\n"
        f"  wb (モデルB 1着):  {wb:.4f}\n"
        f"  wc (モデルC 着順回帰): {wc:.4f}\n"
        f"  ※ core_model.py L331 の 0.35/0.50/0.15 を上記に変更して再学習してください"
    )

    logger.info(summary)
    return best_weights, summary, trials_df
