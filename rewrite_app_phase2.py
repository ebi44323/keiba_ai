import re

with open('app.py', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Fix the strftime bug
text = text.replace("now.strftime('%Y-%m-%d', bundle))", "now.strftime('%Y-%m-%d'), bundle)")
text = text.replace("test_date.strftime('%Y-%m-%d', bundle)", "test_date.strftime('%Y-%m-%d'), bundle")
text = text.replace("target_date.strftime('%Y-%m-%d', bundle)", "target_date.strftime('%Y-%m-%d'), bundle")


# 2. Inject get_morning_prediction at the top
morning_cache_func = """
@st.cache_data(ttl=3600*12, show_spinner=False)
def get_morning_prediction(race_id, race_date_str, _bundle):
    # 朝版（直前スクレイピングなし）
    return run_real_prediction(race_id, race_date_str, _bundle, skip_live_scrape=True)
"""
if "def get_morning_prediction" not in text:
    text = text.replace("from src.inference import run_real_prediction", "from src.inference import run_real_prediction\n" + morning_cache_func)

# 3. Add mobile toggle to `display_result`
# Around line 200, we find `def display_result(df_res, topics, reco, pace_text, confidence_text, show_change_table=True):`
# We want to change the `st.dataframe` to conditionally display a card UI.

card_ui_code = """
        mobile_mode = st.toggle("📱 スマホ（カード）表示", value=True, help="不要な列を隠し縦長に最適化します")
        if mobile_mode:
            for idx, r in show_df.iterrows():
                with st.container(border=True):
                    cols = st.columns([1, 4, 3])
                    cols[0].markdown(f"**{r['馬番']}**")
                    cols[1].markdown(f"**{r['印']} {r['馬名']}**")
                    cols[2].markdown(f"オッズ: **{r['オッズ']}**")
                    st.caption(f"推奨: **{r['💰推奨']}** | 勝率: {r['勝率']} / 複勝率: {r['複勝率']} / EV: {r.get('期待値', 0)}")
        else:
            st.dataframe(
                show_df.style.apply(highlight_row, axis=1)
                       .format({'期待値':'{:.2f}','オッズ':'{:.1f}','枠番':'{:.0f}','馬番':'{:.0f}'}),
                use_container_width=True, hide_index=True
            )
"""

if "mobile_mode = st.toggle(" not in text:
    # Replace the dataframe rendering with the toggle.
    old_df_render = """        st.dataframe(
            show_df.style.apply(highlight_row, axis=1)
                   .format({'期待値':'{:.2f}','オッズ':'{:.1f}','枠番':'{:.0f}','馬番':'{:.0f}'}),
            use_container_width=True, hide_index=True
        )"""
    text = text.replace(old_df_render, card_ui_code)

# 4. Modify 'run_real_prediction' calls to use morning cache where appropriate
# In app.py around "⏩ 次のレースを予想":
live_logic_600 = """
                live_update = st.button("🔄 直前オッズ・馬体重で最新情報を取得し再予測", use_container_width=True)
                if manual_run or force_refresh or auto_triggered or discord_triggered or live_update:
                    with st.spinner('AIが推論中（最新オッズ取得含む）...'):
                        res_df, topics, reco, pace_text, conf_text, _, _, _, err_log = run_real_prediction(next_race['id'], now.strftime('%Y-%m-%d'), bundle, skip_live_scrape=False)
                else:
                    res_df, topics, reco, pace_text, conf_text, _, _, _, err_log = get_morning_prediction(next_race['id'], now.strftime('%Y-%m-%d'), bundle)
"""
# Replace the original block inside `if run_flag:` or near `manual_run`
old_call_block_1 = """                if manual_run or force_refresh or auto_triggered or discord_triggered:
                    with st.spinner('AIが推論中（最新オッズ取得含む）...'):
                        res_df, topics, reco, pace_text, conf_text, _, _, _, err_log = run_real_prediction(next_race['id'], now.strftime('%Y-%m-%d'), bundle)"""
text = text.replace(old_call_block_1, live_logic_600)

live_logic_640 = """
            live_update = st.button("🔄 直前オッズ・馬体重で最新情報を取得し再推論", use_container_width=True)
            if st.button("🚀 朝版 予想開始", type="primary") or live_update:
                with st.spinner('推論中...'):
                    if live_update:
                        res_df, topics, reco, pace_text, conf_text, _, _, _, err_log = run_real_prediction(target_race['id'], now.strftime('%Y-%m-%d'), bundle, skip_live_scrape=False)
                    else:
                        res_df, topics, reco, pace_text, conf_text, _, _, _, err_log = get_morning_prediction(target_race['id'], now.strftime('%Y-%m-%d'), bundle)
"""
old_call_block_2 = """            if st.button("🚀 予想開始", type="primary"):
                with st.spinner('推論中...'):
                    res_df, topics, reco, pace_text, conf_text, _, _, _, err_log = run_real_prediction(target_race['id'], now.strftime('%Y-%m-%d'), bundle)"""
text = text.replace(old_call_block_2, live_logic_640)

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(text)

print("Phase 2 complete")
