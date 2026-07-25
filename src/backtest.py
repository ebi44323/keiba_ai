import pandas as pd
import numpy as np
import lightgbm as lgb
import logging

logger = logging.getLogger('keiba_ebye')


def run_longterm_ev_backtest(df, bundle, ev_threshold=1.0, min_win_prob=0.10,
                              date_from=None, date_to=None):
    """
    学習済みbundleを全期間データに適用し、EV優先◎ vs 標準◎ の回収率を比較する。

    ⚠️ 注意: 学習データ上での評価のため回収率は過楽観になる可能性あり。
             EV優先と標準◎の「相対比較」が主目的。

    引数:
      df           : create_features適用済みのDataFrame
      bundle       : 学習済みモデルbundle
      ev_threshold : EV優先◎に採用する最低EV閾値（デフォルト1.0）
      min_win_prob : EV優先◎に採用する最低AI勝率（デフォルト0.10）
      date_from/to : 集計対象期間の絞り込み（None=全期間）

    戻り値: race_df（レースごとの集計DataFrame）
    """
    (model, model_win, model_reg, features, cat_features, num_features, cat_categories_dict,
     latest_horse_data, horse_course_dict, ped_dict,
     known_jockeys, known_trainers, te_dicts, global_mean, *_rest) = bundle
    # _rest[4]=calibrator, [11]=score_norms, [12]=SOFTMAX_TEMPERATURE（無ければ旧挙動にフォールバック）
    calibrator  = _rest[4]  if len(_rest) > 4  else None
    score_norms = _rest[11] if len(_rest) > 11 else None
    softmax_t   = _rest[12] if len(_rest) > 12 else None

    df = df.copy()

    # 日付・数値フィルタ
    df['日付'] = pd.to_datetime(df['日付'], errors='coerce')
    for col in ['着順', '単勝']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['日付', '着順', '単勝', 'レースID'])

    if date_from:
        df = df[df['日付'] >= pd.Timestamp(date_from)]
    if date_to:
        df = df[df['日付'] <= pd.Timestamp(date_to)]

    if df.empty:
        return pd.DataFrame()

    # TE適用（学習済み te_dicts を使用）
    from src.features_engine import TE_COLS
    for col in list(TE_COLS):
        if col in df.columns:
            df[f'{col}_TE'] = df[col].map(te_dicts.get(col, {})).fillna(global_mean)

    # カテゴリ・数値型変換
    for col in cat_features:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna('不明').astype('category')
    avail = [f for f in features if f in df.columns]
    for col in avail:
        if col not in cat_features:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # カテゴリ列は '不明' で、数値列は 0 で fillna
    X = df[avail].copy()
    for col in X.columns:
        if hasattr(X[col], 'cat'):
            if '不明' not in X[col].cat.categories:
                X[col] = X[col].cat.add_categories(['不明'])
            X[col] = X[col].fillna('不明')
        else:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)

    score_a = np.asarray(model.predict(X), dtype=float)
    score_b = np.asarray(model_win.predict(X), dtype=float)
    score_c = 1.0 - np.asarray(model_reg.predict(X), dtype=float)

    # ── 正規化: bundleに絶対スコア定数があれば本番と同じ絶対正規化、無ければ旧min-max ──
    if score_norms is not None:
        def _apply(s, lohi):
            lo, hi = lohi
            return np.clip((s - lo) / (hi - lo + 1e-9), 0.0, 1.0)
        sa = _apply(score_a, score_norms[0])
        sb = _apply(score_b, score_norms[1])
        sc = _apply(score_c, score_norms[2])
        temp = float(softmax_t) if softmax_t else 1.0
    else:
        def _norm(s):
            return (s - s.min()) / (s.max() - s.min() + 1e-9)
        sa, sb, sc = _norm(score_a), _norm(score_b), _norm(score_c)
        temp = 1.5
    df['予測スコア'] = sa * 0.0581 + sb * 0.8159 + sc * 0.1261  # アンサンブル重み最適化 @ 2026-03-30

    grp = df.groupby('レースID')
    df['exp_s']  = np.exp((df['予測スコア'] - grp['予測スコア'].transform('max')) / temp)
    df['AI勝率'] = df['exp_s'] / grp['exp_s'].transform('sum')
    # Isotonic校正（本番と同じくsoftmax後に適用）
    if calibrator is not None:
        try:
            df['AI勝率'] = np.clip(calibrator.predict(df['AI勝率'].values), 1e-6, 1.0)
        except Exception:
            pass

    # 推定オッズ (市場勝率の逆数; 0除算回避)
    if '市場勝率' in df.columns:
        mw = pd.to_numeric(df['市場勝率'], errors='coerce').fillna(0)
        df['推定オッズ'] = np.where(mw > 0.01, 1.0 / mw, 0.0)
    else:
        df['推定オッズ'] = 0.0
    df['EV'] = df['AI勝率'] * df['推定オッズ']

    # ── レースごとに ◎ を決定 ──────────────────────────────────
    records = []
    for race_id, rdf in grp:
        if len(rdf) < 2:
            continue

        # 標準◎
        std_idx = rdf['AI勝率'].idxmax()
        std_row = rdf.loc[std_idx]
        # 単勝はCSV内で小数倍率（例: 3.5）で格納 → ×100 で100円あたりの払戻額に変換
        std_win  = (int(std_row['着順']) == 1) if pd.notna(std_row['着順']) else False
        std_pay  = float(std_row['単勝']) * 100 if std_win else 0.0

        # EV優先◎
        cands = rdf[(rdf['EV'] >= ev_threshold) & (rdf['AI勝率'] >= min_win_prob)]
        if not cands.empty:
            ev_row  = cands.loc[cands['EV'].idxmax()]
            ev_mode = 'EV優先'
        else:
            ev_row  = std_row
            ev_mode = '標準fallback'
        ev_win  = (int(ev_row['着順']) == 1) if pd.notna(ev_row['着順']) else False
        ev_pay  = float(ev_row['単勝']) * 100 if ev_win else 0.0

        records.append({
            '日付':          rdf['日付'].iloc[0],
            'レースID':      race_id,
            '標準_AI勝率':   round(float(std_row['AI勝率']), 4),
            '標準_払戻':     std_pay,
            'EV_AI勝率':     round(float(ev_row['AI勝率']), 4),
            'EV_EV値':       round(float(ev_row['EV']), 3),
            'EV_払戻':       ev_pay,
            'EV_モード':     ev_mode,
        })

    if not records:
        return pd.DataFrame()

    result_df = pd.DataFrame(records).sort_values('日付').reset_index(drop=True)
    result_df['年月'] = result_df['日付'].dt.to_period('M')

    logger.info(f"超長期バックテスト完了: {len(result_df)}レース")
    return result_df

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

    for col in cat_features:
        if col not in df.columns: df[col] = '不明'
        df[col] = df[col].astype(str).fillna('不明').astype('category')
        
    for f in features:
        if f not in cat_features and f in df.columns:
            df[f] = pd.to_numeric(df[f], errors='coerce')
        
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
        
        test_df['予測スコア'] = _norm(score_a)*0.0581 + _norm(score_b)*0.8159 + _norm(score_c)*0.1261  # アンサンブル重み最適化 @ 2026-03-30
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
