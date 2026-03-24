import pandas as pd
import numpy as np
import lightgbm as lgb
import logging

logger = logging.getLogger('keiba_ebye')

def run_timeseries_backtest(df, features, cat_features, te_cols, n_splits=3, test_days=30):
    """
    リーク防止版の時系列バックテスト（Time-Series Split）
    n_splits: 分割回数
    test_days: 各検証期間の日数
    
    戻り値: (overall_return_rate, results_df)
    """
    df = df.copy()
    if '馬券内' not in df.columns:
        df['馬券内'] = (df['着順'] <= 3).astype(int)
    if 'win_label' not in df.columns:
        df['win_label'] = (df['着順'] == 1).astype(int)
        
    df = df.sort_values('日付').reset_index(drop=True)
    max_date = df['日付'].max()
    
    results = []
    
    for i in range(n_splits):
        # 過去から現在に向かってウィンドウをスライド
        test_end = max_date - pd.Timedelta(days=test_days * i)
        test_start = test_end - pd.Timedelta(days=test_days)
        
        # 訓練データは完全にテスト期間より前（未来の情報のリークを100%防止する）
        train_mask = df['日付'] < test_start
        test_mask = (df['日付'] >= test_start) & (df['日付'] <= test_end)
        
        train_df = df[train_mask].copy()
        test_df = df[test_mask].copy()
        
        if len(train_df) == 0 or len(test_df) == 0:
            continue
            
        logger.info(f"バックテスト分割 {i+1}: 訓練={len(train_df)}件, テスト={len(test_df)}件 ({test_start.date()} ~ {test_end.date()})")
        
        # ── 1. Target Encoding (訓練データのみで計算) ──
        te_dicts = {}
        global_mean = train_df['馬券内'].mean()
        local_features = features.copy()
        for col in te_cols:
            if col in train_df.columns:
                te_dicts[col] = train_df.groupby(col)['馬券内'].mean().to_dict()
                train_df[f'{col}_TE'] = train_df[col].map(te_dicts[col]).fillna(global_mean)
                test_df[f'{col}_TE']  = test_df[col].map(te_dicts[col]).fillna(global_mean)
                if f'{col}_TE' not in local_features:
                    local_features.append(f'{col}_TE')

        train_groups = train_df.groupby('レースID', sort=False).size().values
        test_groups  = test_df.groupby('レースID', sort=False).size().values
        
        # ── 2. モデルA (複勝Ranker) ──
        model_a = lgb.LGBMRanker(n_estimators=300, learning_rate=0.01, num_leaves=63, random_state=42)
        model_a.fit(train_df[local_features], train_df['馬券内'], group=train_groups,
                    categorical_feature=[f for f in cat_features if f in local_features],
                    eval_set=[(test_df[local_features], test_df['馬券内'])], eval_group=[test_groups])
        
        # ── 3. モデルB (1着Ranker) ──
        model_b = lgb.LGBMRanker(n_estimators=300, learning_rate=0.02, num_leaves=48, random_state=123)
        model_b.fit(train_df[local_features], train_df['win_label'], group=train_groups,
                    categorical_feature=[f for f in cat_features if f in local_features],
                    eval_set=[(test_df[local_features], test_df['win_label'])], eval_group=[test_groups])

        # ── 4. モデルC (着順パーセント Regressor: 実装済みなら) ──
        model_c = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.03, num_leaves=31, random_state=777)
        model_c.fit(train_df[local_features], train_df['着順パーセント'].fillna(0.5),
                    categorical_feature=[f for f in cat_features if f in local_features])

        # ── スコア計算 ──
        score_a = model_a.predict(test_df[local_features])
        score_b = model_b.predict(test_df[local_features])
        # Regressorは値が「低いほど良い（着順が上）」なので反転
        score_c = 1.0 - model_c.predict(test_df[local_features]) 

        def _norm(s):
            return (s - s.min()) / (s.max() - s.min() + 1e-9)
        
        test_df['予測スコア'] = _norm(score_a)*0.35 + _norm(score_b)*0.50 + _norm(score_c)*0.15
        test_df['exp_score'] = np.exp(test_df['予測スコア'] - test_df.groupby('レースID')['予測スコア'].transform('max'))
        test_df['AI勝率'] = test_df['exp_score'] / test_df.groupby('レースID')['exp_score'].transform('sum')
        
        test_df['fold'] = i
        results.append(test_df)
        
    if not results:
        return 0, pd.DataFrame()
        
    res_df = pd.concat(results, ignore_index=True)
    
    # 回収率計算（1番手評価の単勝を買った場合）
    top_preds = res_df.sort_values(['レースID', 'AI勝率'], ascending=[True, False]).groupby('レースID').head(1)
    win_hits = top_preds[pd.to_numeric(top_preds['着順'], errors='coerce') == 1]
    invest = len(top_preds) * 100
    win_ret = (pd.to_numeric(win_hits['単勝'], errors='coerce') * 100).sum()
    
    ret_rate = (win_ret / invest * 100) if invest > 0 else 0
    return ret_rate, res_df
