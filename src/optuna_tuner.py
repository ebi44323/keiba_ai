import pandas as pd
import numpy as np
import lightgbm as lgb
import logging

def run_optuna_tuning(df, features, cat_features, te_cols, n_trials=30):
    """
    Optunaを使ってRankerおよびRegressorのハイパーパラメータを自動探索する。
    探索した最適なパラメータは辞書として返す。
    ※ 実行から終了まで時間がかかるためバックグラウンドやStreamlitから長時間のwaitが必要。
    """
    try:
        import optuna
    except ImportError:
        return None, "Optunaがインストールされていません(`pip install optuna`)"

    df = df.copy()
    if '馬券内' not in df.columns:
        df['馬券内'] = (df['着順'] <= 3).astype(int)
    if 'win_label' not in df.columns:
        df['win_label'] = (df['着順'] == 1).astype(int)

    for col in cat_features:
        if col not in df.columns: df[col] = '不明'
        df[col] = df[col].astype(str).fillna('不明').astype('category')

    for f in features:
        if f not in cat_features and f in df.columns:
            df[f] = pd.to_numeric(df[f], errors='coerce')

    df = df.sort_values('日付').reset_index(drop=True)
    max_date = df['日付'].max()
    test_start = max_date - pd.Timedelta(days=30)
    
    train_df = df[df['日付'] < test_start].copy()
    test_df  = df[df['日付'] >= test_start].copy()
    
    # TE
    global_mean = train_df['馬券内'].mean()
    local_features = features.copy()
    for col in te_cols:
        if col in train_df.columns:
            te = train_df.groupby(col)['馬券内'].mean().to_dict()
            train_df[f'{col}_TE'] = train_df[col].map(te).fillna(global_mean)
            test_df[f'{col}_TE']  = test_df[col].map(te).fillna(global_mean)
            if f'{col}_TE' not in local_features:
                local_features.append(f'{col}_TE')

    train_groups = train_df.groupby('レースID', sort=False).size().values
    test_groups  = test_df.groupby('レースID', sort=False).size().values
    cat_feats = [f for f in cat_features if f in local_features]

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 600),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 15, 127),
            'max_bin': trial.suggest_int('max_bin', 127, 255),
            'cat_smooth': trial.suggest_float('cat_smooth', 1.0, 50.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
            'subsample': trial.suggest_float('subsample', 0.5, 0.9),
            'random_state': 42,
        }
        
        # 1着モデル(Ranker)の最適化を目標にする
        model = lgb.LGBMRanker(**params)
        model.fit(
            train_df[local_features], 
            train_df['win_label'], 
            group=train_groups,
            categorical_feature=cat_feats,
            eval_set=[(test_df[local_features], test_df['win_label'])], 
            eval_group=[test_groups]
        )
        
        preds = model.predict(test_df[local_features])
        test_df['optuna_pred'] = preds
        
        # 評価指標：単勝1番手評価の回収率
        top_preds = test_df.sort_values(['レースID', 'optuna_pred'], ascending=[True, False]).groupby('レースID').head(1)
        win_hits = top_preds[pd.to_numeric(top_preds['着順'], errors='coerce') == 1]
        invest = len(top_preds) * 100
        win_ret = (pd.to_numeric(win_hits['単勝'], errors='coerce') * 100).sum()
        
        ret_rate = (win_ret / invest * 100) if invest > 0 else 0
        return ret_rate

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    return study.best_params, f"チューニング完了: 最適回収率={study.best_value:.1f}%, 試行回数={n_trials}"
