import re

# 1. Update core_model.py
with open("src/core_model.py", "r", encoding="utf-8") as f:
    core_text = f.read()

regressor_code = """
    # ── モデルC: 着順パーセント予測Regressor（アンサンブル用）────────────────
    model_reg = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=31, max_bin=255,
                                  random_state=777,
                                  colsample_bytree=0.7, subsample=0.8)
    model_reg.fit(train_df[features], train_df['着順パーセント'].fillna(0.5),
                  categorical_feature=[f for f in cat_features if f in features])
"""

if "model_reg =" not in core_text:
    core_text = core_text.replace("    # ── アンサンブルスコア ────────────────", regressor_code + "\n    # ── アンサンブルスコア ────────────────")

ensemble_code = """
    score_a = model.predict(test_df[features])
    score_b = model_win.predict(test_df[features])
    score_c = 1.0 - model_reg.predict(test_df[features])  # 低い方が上位なので反転

    def _norm_scores(s):
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn + 1e-9)
    _sa_norm = _norm_scores(score_a)
    _sb_norm = _norm_scores(score_b)
    _sc_norm = _norm_scores(score_c)

    # 複勝0.35, 1着0.5, 着順回帰0.15
    test_df['予測スコア'] = _sa_norm * 0.35 + _sb_norm * 0.50 + _sc_norm * 0.15
"""
# Replace the old ensemble logic
old_ensemble = r"    score_a = model\.predict\(test_df\[features\]\).*?test_df\['予測スコア'\] = _sa_norm \* best_weight \+ _sb_norm \* \(1 - best_weight\)"
core_text = re.sub(old_ensemble, ensemble_code.strip('\n'), core_text, flags=re.DOTALL)

# Update bundle packing limit
bundle_old = "bundle = (model, model_win, features, cat_features,"
bundle_new = "bundle = (model, model_win, model_reg, features, cat_features,"
core_text = core_text.replace(bundle_old, bundle_new)

with open("src/core_model.py", "w", encoding="utf-8") as f:
    f.write(core_text)

# 2. Update inference.py
with open("src/inference.py", "r", encoding="utf-8") as f:
    inf_text = f.read()

unpack_old = "    (model, model_win, features, cat_features, num_features, cat_categories_dict,\n     latest_horse_data"
unpack_new = "    (model, model_win, model_reg, features, cat_features, num_features, cat_categories_dict,\n     latest_horse_data"
inf_text = inf_text.replace(unpack_old, unpack_new)

inf_ensemble = """
        # アンサンブル: 3モデルの予測を結合
        _sa = model.predict(df_test[features]).astype(float)
        _sa = (_sa - _sa.min()) / (_sa.max() - _sa.min() + 1e-9)
        try:
            _sb = model_win.predict(df_test[features]).astype(float)
            _sb = (_sb - _sb.min()) / (_sb.max() - _sb.min() + 1e-9)
            
            _sc = 1.0 - model_reg.predict(df_test[features]).astype(float)
            _sc = (_sc - _sc.min()) / (_sc.max() - _sc.min() + 1e-9)
            
            raw_scores = _sa * 0.35 + _sb * 0.50 + _sc * 0.15
        except Exception as _e:
            logger.warning(f'model_win/reg予測失敗、model_aのみ使用: {_e}')
            raw_scores = _sa  # フォールバック
"""
old_inf_ensemble = r"        # アンサンブル: 最適化済み重みを使用.*?raw_scores = _sa  # model_win失敗時はmodel_aのみ"
inf_text = re.sub(old_inf_ensemble, inf_ensemble.strip('\n'), inf_text, flags=re.DOTALL)

with open("src/inference.py", "w", encoding="utf-8") as f:
    f.write(inf_text)

print("Patched core_model.py and inference.py")
