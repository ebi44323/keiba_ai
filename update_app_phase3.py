import re

with open("app.py", "r", encoding="utf-8") as f:
    app_text = f.read()

new_tab_code = """
# ==========================================
# ② 新・モデル検証＆AIチューニング (Phase 3実装)
# ==========================================
elif action == "📊 モデル検証 (ウォークフォワード)":
    st.subheader("📊 AIチューニング & バックテスト (時系列分割)")
    st.info("過去のデータを時系列に分割し、未来の情報が一切混入しない（リーク防止）厳密なバックテストを行います。\\nまた、Optunaを用いたAIのハイパーパラメータ自動最適化も実行可能です。")

    tab_bt, tab_op = st.tabs(["🧪 厳密バックテスト", "🔧 Optuna自動チューニング"])

    with tab_bt:
        st.markdown("#### リーク防止版 Time-Series Split バックテスト")
        bt_splits = st.slider("検証を遡る回数 (n_splits)", 1, 5, 3)
        bt_days = st.slider("1回あたりの検証日数", 7, 60, 30)

        if st.button("🚀 バックテスト実行", type="primary"):
            with st.spinner(f"過去 {bt_splits} 期間分のモデル学習と推論を行っています... (数分かかります)"):
                try:
                    df_bt = pd.read_csv('learning_data_perfect_tier.zip', compression='zip', dtype=str)
                    df_bt['日付'] = pd.to_datetime(df_bt['日付'], format='mixed', errors='coerce')
                    df_bt = df_bt.dropna(subset=['日付', '着順', '単勝'])
                    for col in ['着順', '単勝', '人気']: df_bt[col] = pd.to_numeric(df_bt[col], errors='coerce')

                    from src.backtest import run_timeseries_backtest
                    ret_rate, res_df = run_timeseries_backtest(df_bt, features, cat_features, te_cols, n_splits=bt_splits, test_days=bt_days)

                    st.success(f"✅ 全期間テスト完了！ 総合単勝回収率: **{ret_rate:.1f}%**")
                    if not res_df.empty:
                        st.dataframe(res_df.groupby('fold').apply(lambda x: x.sort_values('AI勝率', ascending=False).groupby('レースID').head(1)).reset_index(drop=True)[['日付','レースID','馬券内','着順','単勝','AI勝率']])
                except Exception as e:
                    import traceback
                    st.error(f"バックテストエラー: {e}")
                    st.code(traceback.format_exc())

    with tab_op:
        st.markdown("#### Optuna 超パラメータ自動最適化")
        st.caption("AIの予測精度を最大限に引き出すためのパラメータ探索を行います。実行には時間がかかります。")
        n_trials = st.number_input("探索回数 (Trials)", min_value=10, max_value=200, value=30, step=10)

        if st.button("🔧 チューニング開始", type="primary"):
            with st.spinner(f"Optunaによる探索中 ({n_trials} trials)..."):
                try:
                    df_op = pd.read_csv('learning_data_perfect_tier.zip', compression='zip', dtype=str)
                    df_op['日付'] = pd.to_datetime(df_op['日付'], format='mixed', errors='coerce')
                    df_op = df_op.dropna(subset=['日付', '着順', '単勝'])

                    from src.optuna_tuner import run_optuna_tuning
                    best_p, msg = run_optuna_tuning(df_op, features, cat_features, te_cols, n_trials=n_trials)

                    st.success(msg)
                    st.json(best_p)
                    st.warning("⚠️ 新しいパラメータをシステムに適用するには、`src/core_model.py` の `lgb.LGBMRanker` の引数を書き換えてください。")
                except Exception as e:
                    import traceback
                    st.error(f"Optunaチューニングエラー: {e}")
                    st.code(traceback.format_exc())

"""

# Regex replacement: from 'elif action == "📊 モデル検証 (ウォークフォワード)":' down to but excluding 'elif action == "🏇 騎手・調教師フォーム分析":'
pattern = r"# ==========================================\n# ② ウォークフォワード検証 \(モデル精度の安定性確認\)\n# ==========================================\nelif action == \"📊 モデル検証 \(ウォークフォワード\)\":.*?elif action == \"🏇 騎手・調教師フォーム分析\":"
replacement = new_tab_code + "elif action == \"🏇 騎手・調教師フォーム分析\":"

app_text_new = re.sub(pattern, replacement.strip('\n') + '\n', app_text, flags=re.DOTALL)

with open("app.py", "w", encoding="utf-8") as f:
    f.write(app_text_new)

print("app.py updated with Phase 3 UI.")
