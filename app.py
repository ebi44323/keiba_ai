import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import requests
from bs4 import BeautifulSoup
import re
import os
import datetime
import pytz
import traceback
import time
import json
import random

from features_engine import NUM_FEATURES, CAT_FEATURES, TE_COLS, classify_style

# ── ロギング設定 ────────────────────────────────────────────────
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(funcName)s: %(message)s',
    handlers=[
        logging.FileHandler('keiba.log', encoding='utf-8'),
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger('keiba_ebye')

st.set_page_config(page_title="keiba-ebye 予測ダッシュボード", page_icon="🐴", layout="wide")
st.title("🐴 keiba-ebye 予測ダッシュボード")
st.markdown("えーびーあい (ebi × AI × Eye) が、極限まで高められた精度でお宝馬を暴き出すかも。。。。")

from src.features_engine import NUM_FEATURES, CAT_FEATURES, TE_COLS, classify_style
from src.utils import VENUE_MAWARI, VENUE_CHIKEI, TRACK_CONDITION_MAP, classify_race_class, resolve_name, get_headers
from src.core_model import prepare_model_and_data, _HF_TOKEN, _HF_REPO_ID
from src.scraper import get_todays_races, get_weekend_dates, get_payouts, get_all_payouts, get_odds_from_soup, fetch_horse_last_race
from src.reports import generate_pdf_report, generate_txt_report
from src.discord_utils import _push_discord_queue, send_discord_prediction, send_discord_review, _test_discord_webhook, _DISCORD_WEBHOOK_URL, _DISCORD_REVIEW_WEBHOOK_URL
from src.inference import run_real_prediction

@st.cache_data(ttl=3600*12, show_spinner=False)
def get_morning_prediction(race_id, race_date_str, _bundle):
    # 朝版（直前スクレイピングなし）
    return run_real_prediction(race_id, race_date_str, _bundle, skip_live_scrape=True)



_hub_available = bool(_HF_TOKEN and _HF_REPO_ID)
_hub_label = "HF Hub" if _hub_available else "ローカル学習"

with st.spinner(f'AIエンジン起動中... ({_hub_label}からロード試行)'):
    bundle = prepare_model_and_data()
    (model, model_win, model_reg, features, cat_features, num_features, cat_categories_dict,
     latest_horse_data, horse_course_dict, ped_dict,
     known_jockeys, known_trainers, te_dicts, global_mean, recent_return_rate, ensemble_weight,
     auc_win, auc_place) = bundle

# ==========================================
# 4. メインUI構成
# ==========================================
st.sidebar.markdown("## 🕹️ keiba-ebye メニュー")

# ── Free / Pro 認証 ─────────────────────────────────────────
_PRO_PASSWORD = os.environ.get("PRO_PASSWORD", "")  # HuggingFace Secrets に設定

if _PRO_PASSWORD:
    _is_pro = st.session_state.get('is_pro', False)

    # ── オーナー自動ログイン ─────────────────────────────────
    # OWNER_KEY Secret を設定しておくと、URLパラメータ ?key=<OWNER_KEY> で
    # パスワード入力なしに自動ログインできる
    _OWNER_KEY = os.environ.get("OWNER_KEY", "")
    if not _is_pro and _OWNER_KEY:
        _url_key = st.query_params.get("key", "")
        if _url_key == _OWNER_KEY:
            st.session_state['is_pro'] = True
            st.session_state['is_owner'] = True
            _is_pro = True

    if not _is_pro:
        _pw_input = st.sidebar.text_input("🔑 Pro アクセスコード", type="password",
                                           placeholder="Proメンバーはここに入力")
        if _pw_input == _PRO_PASSWORD:
            st.session_state['is_pro'] = True
            st.rerun()
        st.sidebar.caption("アクセスコードなしでも「次のレースを予想」は無料でご利用いただけます。")
    else:
        _is_owner = st.session_state.get('is_owner', False)
        if _is_owner:
            st.sidebar.success("👑 オーナーログイン中")
        else:
            st.sidebar.success("✅ Pro メンバー")
            if st.sidebar.button("ログアウト", key="logout"):
                st.session_state['is_pro'] = False
                st.session_state['is_owner'] = False
                st.rerun()
else:
    _is_pro = True  # PRO_PASSWORD未設定 = 全機能開放（開発時・移行期）
# Proのみのメニュー項目
_PRO_ACTIONS = [
    "📅 今週末の全レース予想",
    "🔍 レースを指定して予想",
    "📝 1日の振り返り (答え合わせ)",
    "🧪 性能試験 (バックテスト)",
    "📈 長期成績分析",
    "📊 AIチューニング & バックテスト",
    "🏇 騎手・調教師フォーム分析",
    "📝 馬券メモ管理",
    "🐴 愛馬の成長記録",
]
_FREE_ACTIONS = ["⏩ 次のレースを予想"]
_ALL_ACTIONS = _FREE_ACTIONS + _PRO_ACTIONS

action = st.sidebar.radio(
    "機能を選択",
    _FREE_ACTIONS if not _is_pro else _ALL_ACTIONS
)

# Pro機能に直接アクセスしようとした場合のガード
if action in _PRO_ACTIONS and not _is_pro:
    st.warning("この機能はProメンバー限定です。サイドバーにアクセスコードを入力してください。")
    st.stop()

# ── Discord 設定 ─────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("### 💬 Discord 連携")

if not _DISCORD_WEBHOOK_URL:
    _discord_enabled = False
    st.sidebar.caption("💡 HuggingFace Secrets に以下を設定してください")
    st.sidebar.code("DISCORD_WEBHOOK_URL        # 直前予想チャンネル\nDISCORD_REVIEW_WEBHOOK_URL # 振り返りチャンネル")
else:
    # チャンネル設定状況表示
    _pred_ok   = bool(_DISCORD_WEBHOOK_URL)
    _review_ok = bool(_DISCORD_REVIEW_WEBHOOK_URL and
                      _DISCORD_REVIEW_WEBHOOK_URL != _DISCORD_WEBHOOK_URL)
    st.sidebar.caption(f"📢 直前予想: {'✅ 専用ch' if _pred_ok else '❌ 未設定'}")
    st.sidebar.caption(f"📊 振り返り: {'✅ 専用ch' if _review_ok else '⚠️ 予想chと共用'}")

    # 自動投稿 ON/OFF
    _discord_enabled = st.sidebar.checkbox(
        "📤 直前予想 自動投稿",
        value=True,
        help="発走15分前にDiscord通知（GitHub Actions経由）、5分前に画面を最新オッズで更新します"
    )

    # 接続テストボタン
    _test_col1, _test_col2 = st.sidebar.columns(2)
    with _test_col1:
        if st.button("🔌 予想ch テスト", key="discord_test_pred", use_container_width=True):
            _ok, _msg = _test_discord_webhook(_DISCORD_WEBHOOK_URL, "直前予想")
            if _ok: st.sidebar.success(_msg)
            else:   st.sidebar.error(_msg)
    with _test_col2:
        if st.button("🔌 振返ch テスト", key="discord_test_review", use_container_width=True):
            _ok, _msg = _test_discord_webhook(_DISCORD_REVIEW_WEBHOOK_URL, "振り返り")
            if _ok: st.sidebar.success(_msg)
            else:   st.sidebar.error(_msg)

st.sidebar.markdown("---")
st.sidebar.markdown("### 💰 軍資金シミュレーター")
sim_budget     = st.sidebar.number_input("軍資金 (円)", 1000, 500000, 30000, 1000,
                   help="1日の総予算。ケリー基準でここから各レースに配分します。")
sim_ev_filter  = st.sidebar.slider("購入する期待値の下限", 1.0, 3.0, 1.2, 0.1,
                   help="この期待値以上の馬だけを買います。高いほど厳選。")
sim_kelly_frac = st.sidebar.slider("ケリー係数", 0.1, 1.0, 0.25, 0.05,
                   help="1.0=フルケリー(高リスク) 0.25=推奨(安定)")
sim_max_per_race = st.sidebar.slider("1レース最大投資額 (軍資金の%)", 5, 40, 20, 5,
                   help="1レースに軍資金の何%まで使うか上限を設定します。") / 100

# ── モデル管理 (Pro + HF Hubが設定済みの場合のみ表示) ─────
if _is_pro and _hub_available:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ モデル管理")
    st.sidebar.caption(f"HF Hub: `{_HF_REPO_ID}`")
    st.sidebar.caption(f"🎚 アンサンブル重み: 複勝={ensemble_weight:.1f} / 1着={1-ensemble_weight:.1f}")
    # AUC: 特徴量計算にリークが含まれるため表示は参考値扱い
    if auc_win > 0:
        st.sidebar.caption(f"⚠️ AUC(参考): 1着={auc_win:.4f} / 複勝={auc_place:.4f}")
        st.sidebar.caption("　└ 特徴量リークにより過大評価の可能性あり")
    if st.sidebar.button("🔄 強制再学習 & Hub更新", help="データが更新された際に手動で再学習してHubにアップロードします"):
        st.cache_resource.clear()
        with st.spinner("再学習中... (数分かかります)"):
            (model, model_win, model_reg, features, cat_features, num_features, cat_categories_dict,
             latest_horse_data, horse_course_dict, ped_dict,
             known_jockeys, known_trainers, te_dicts, global_mean, recent_return_rate, ensemble_weight,
             auc_win, auc_place) = prepare_model_and_data(force_retrain=True)
        st.sidebar.success("✅ 再学習完了・Hubにアップロードしました")
        st.rerun()

tokyo_tz = pytz.timezone('Asia/Tokyo')
now = datetime.datetime.now(tokyo_tz)

def display_error_log(err_log):
    st.error("⚠️ 予想データまたは結果の取得に失敗しました。")
    with st.expander("🔍 エラー解析ログを見る (デバッグ用)", expanded=True):
        if not err_log:
            st.write("（エラーログなし: ネットワーク接続またはサイト側の問題の可能性があります）")
        for log in err_log:
            st.code(log, language=None)

def display_result(df_res, topics, reco, pace_text, confidence_text, show_change_table=True):
    tab1, tab2, tab3, tab4 = st.tabs(["📊 予想一覧", "💡 展開・買い目", "🔍 性能詳細", "🎰 複合馬券EV"])

    with tab1:
        if "鉄板" in confidence_text: st.success(confidence_text)
        elif "波乱" in confidence_text: st.error(confidence_text)
        else: st.info(confidence_text)

        # ── 軍資金シミュレーター ─────────────────────────────
        def calc_kelly_sim(p_raw, odds_raw):
            if "見送り" in confidence_text: return 0
            try:
                p = float(str(p_raw).replace('%',''))/100 if '%' in str(p_raw) else float(p_raw)
                b = float(odds_raw) - 1.0
            except: return 0
            if b <= 0: return 0
            if p * float(odds_raw) < sim_ev_filter: return 0
            f_star = p - (1.0 - p) / b
            if f_star <= 0: return 0
            bet = int(min(f_star * sim_kelly_frac * sim_budget, sim_budget * sim_max_per_race) / 100) * 100
            return max(0, bet)

        # メモ読み込み
        memo_horses = st.session_state.get('memo_horses', [])
        all_memos = {}
        if memo_horses and os.path.exists("horse_memos.json"):
            try:
                with open("horse_memos.json", encoding="utf-8") as _mf:
                    all_memos = json.load(_mf)
            except Exception as _e:
                logger.debug(f'horse_memos.json読み込みスキップ: {_e}')

        # ── 🏇 馬柱UI ─────────────────────────────────────────
        import altair as alt
        st.markdown("#### 🏇 出走馬 AI予想")
        total_bet = 0

        # ── 馬柱テーブル（上段: 印付きリスト）─────────────────
        bets = []
        for _, row in df_res.iterrows():
            bet = calc_kelly_sim(row['勝率(AI予測)'], row['単勝オッズ'])
            bets.append(f"¥{bet:,}" if bet > 0 else "見送り")
            if bet > 0: total_bet += bet

        show_df = df_res[['印','枠番','馬番','馬名','脚質カテゴリ','単勝オッズ','勝率(AI予測)','複勝率(AI予測)','期待値']].copy()
        show_df = show_df.rename(columns={'勝率(AI予測)':'勝率','複勝率(AI予測)':'複勝率','単勝オッズ':'オッズ','脚質カテゴリ':'脚質'})
        show_df['💰推奨'] = bets
        show_df['勝率']   = (show_df['勝率'] * 100).map('{:.1f}%'.format)
        show_df['複勝率'] = (show_df['複勝率'] * 100).map('{:.1f}%'.format)

        if total_bet > 0:
            st.caption(f"💰 推奨投資合計: **¥{total_bet:,}** / 軍資金¥{sim_budget:,}の {total_bet/sim_budget*100:.1f}%")

        # メモあり馬のメッセージ
        for _, row in df_res.iterrows():
            hname = row['馬名']
            if hname in all_memos:
                latest = sorted(all_memos[hname], key=lambda x: x["日付"], reverse=True)[0]
                st.info(f"📝 **{hname}** ({row['印']}) にメモあり: {latest['タグ']} {latest['日付']} — {latest['メモ'] or '(内容なし)'}")

        def highlight_row(row):
            horse_name = row.get('馬名', '')
            bet_str    = row.get('💰推奨', '見送り')
            try:
                ev = float(row.get('期待値', 0) or 0)
            except (ValueError, TypeError):
                ev = 0.0
            if horse_name in memo_horses:
                return ['border-left:3px solid #9B59B6;background:rgba(155,89,182,0.08)'] * len(row)
            # EV1.5以上は常に赤ハイライト
            if ev >= 1.5:
                return ['background-color:rgba(255,75,75,0.18)'] * len(row)
            if bet_str != '見送り':
                return ['background-color:rgba(255,200,0,0.10)'] * len(row)
            return [''] * len(row)

        st.dataframe(
            show_df.style.apply(highlight_row, axis=1)
                   .format({'期待値':'{:.2f}','オッズ':'{:.1f}','枠番':'{:.0f}','馬番':'{:.0f}'}),
            use_container_width=True, hide_index=True
        )

        # ── リアルタイム勝率バー（下段: グラフ）────────────────
        st.markdown("---")
        bar_df = df_res[['馬番','馬名','印','勝率(AI予測)','期待値','単勝オッズ']].copy()
        bar_df['勝率%'] = (bar_df['勝率(AI予測)'] * 100).round(1)
        bar_df['label'] = bar_df['印'] + ' ' + bar_df['馬名']
        bar_df['color'] = bar_df['期待値'].apply(
            lambda v: '#FF4B4B' if v >= 1.5 else '#4B8BFF'
        )
        bar_chart = alt.Chart(bar_df).mark_bar(cornerRadiusEnd=4).encode(
            y=alt.Y('label:N', sort=list(bar_df['label']), title='', axis=alt.Axis(labelFontSize=12)),
            x=alt.X('勝率%:Q', title='AI勝率 (%)', scale=alt.Scale(domain=[0, bar_df['勝率%'].max()*1.15])),
            color=alt.Color('color:N', scale=None, legend=None),
            tooltip=['馬名', '勝率%', '期待値', '単勝オッズ']
        ).properties(height=max(200, len(bar_df) * 32))
        text_chart = alt.Chart(bar_df).mark_text(align='left', dx=4, fontSize=11).encode(
            y=alt.Y('label:N', sort=list(bar_df['label'])),
            x=alt.X('勝率%:Q'),
            text=alt.Text('勝率%:Q', format='.1f'),
            color=alt.value('var(--color-text-secondary)')
        )
        st.altair_chart((bar_chart + text_chart).configure_view(strokeWidth=0), use_container_width=True)
        st.caption("赤バー = EV1.5以上の注目馬 / 青バー = 通常")

    with tab2:
        st.info(f"**🏇 展開予想:**\n{pace_text}")
        ev_horses = df_res[(df_res.index < 5) & (df_res['期待値'] >= sim_ev_filter)]
        if not ev_horses.empty:
            st.error(f"💰 **【期待値レーダー発動】** {', '.join(ev_horses['馬名'].tolist())} に妙味あり！")
        if topics: st.warning("**📝 要注目トピック馬:**\n\n" + "\n".join(topics))

    with tab3:
        import altair as alt

        # 見方ガイド
        with st.expander("📖 指標の見方・解説 (クリックで開く)", expanded=False):
            guide_data = {
                "指標": ["地力(中央値)", "最高ポテンシャル", "上昇度",
                          "コース適性", "位置取り変化", "近3走安定度",
                          "休養日数", "騎手変化", "馬場変化", "距離変化"],
                "見方・ポイント": [
                    "近5走スピード指数の中央値。50が平均、高いほど強い。最も信頼できる実力値",
                    "近5走の最高値。地力との差が大きい馬は条件次第で爆発力がある",
                    "前走指数-近5走中央値。+2以上=上昇中(赤強調)、-2以下=下降気味(グレー)",
                    "このコースでの過去着順パーセント。0に近いほどコースが得意(緑強調)",
                    "今回予想コーナー順位-前走。マイナス=前に行きそう、プラス=後退しそう",
                    "直近3走の着順パーセント平均。0.3以下=安定(緑)、0.6以上=不安定",
                    "前走からの間隔日数。14日以内=超短期、56日以上=休み明けで状態要確認",
                    "前走と今回の騎手比較。格上騎手への乗り替わりは要注目",
                    "芝/ダートの変更。初芝・初ダートは過去実績との乖離リスクあり",
                    "距離の変更。距離延長/短縮で得意不得意が変わる。適性距離の確認推奨",
                ],
            }
            st.dataframe(pd.DataFrame(guide_data), use_container_width=True, hide_index=True)

        # スコアテーブル
        st.markdown("#### 📐 AI評価スコア詳細")
        score_cols_map = {
            '近5走_中央値スピード指数': '地力(中央値)',
            '近5走_最高スピード指数':   '最高ポテンシャル',
            '上昇度_スピード指数':       '上昇度',
            'コース適性_着順パーセント': 'コース適性',
            '位置取りショック':          '位置取り変化',
            '直近3走着順パーセント':     '近3走安定度',
        }
        avail_s = {k: v for k, v in score_cols_map.items() if k in df_res.columns}
        score_df = df_res[['馬番', '馬名', '脚質カテゴリ'] + list(avail_s.keys())].copy()
        score_df = score_df.rename(columns=avail_s)

        def highlight_score(row):
            styles = [''] * len(row)
            cols = list(row.index)
            for col, fn in [
                ('上昇度', lambda v: 'color:#FF4B4B;font-weight:bold' if v >= 2.0
                           else ('color:#888888' if v <= -2.0 else '')),
                ('地力(中央値)', lambda v: 'color:#4B8BFF;font-weight:bold' if v >= 55 else ''),
                ('コース適性', lambda v: 'color:#22AA22;font-weight:bold' if v <= 0.2 else ''),
                ('近3走安定度', lambda v: 'color:#22AA22;font-weight:bold' if v <= 0.3
                              else ('color:#888888' if v >= 0.6 else '')),
            ]:
                if col in cols:
                    try:
                        styles[cols.index(col)] = fn(float(row[col]))
                    except: pass
            return styles

        fmt_s = {v: '{:.2f}' for v in avail_s.values()}
        st.dataframe(
            score_df.style.apply(highlight_score, axis=1).format(fmt_s, na_rep='不明'),
            use_container_width=True, hide_index=True
        )
        st.caption("赤=上昇度+2以上 / 青=地力55以上 / 緑=コース適性0.2以下 or 安定度0.3以下")

        # 前走比較テーブル（バックテスト・振り返りではshow_change_table=Falseで非表示）
        if show_change_table:
            st.markdown("#### 🔄 前走との変化点チェック")

            def fmt_kyuyo(days):
                try:
                    v = float(days)
                    if np.isnan(v) or v <= 0: return '不明'
                    d = int(v)
                    if d <= 14:  return f'[超短]{d}日'
                    if d <= 28:  return f'[中2週]{d}日'
                    if d <= 56:  return f'[中3-7週]{d}日'
                    return f'[休み明け]{d}日'
                except: return '不明'

            change_rows = []
            for _, row in df_res.iterrows():
                flag_j  = int(row.get('乗り替わりフラグ', 0) or 0)
                prev_j  = str(row['_前走騎手']) if '_前走騎手' in df_res.columns and not pd.isna(row.get('_前走騎手')) else '不明'
                now_j   = str(row.get('騎手', ''))
                jockey_str = f"変更:{prev_j}->{now_j}" if flag_j == 1 and prev_j not in ('不明', '', 'nan') else '変化なし'

                flag_s  = int(row.get('馬場替わりフラグ', 0) or 0)
                prev_s  = str(row['_前走馬場']) if '_前走馬場' in df_res.columns and not pd.isna(row.get('_前走馬場')) else '不明'
                now_s   = str(row.get('芝/ダート', ''))
                surf_str = f"変更:{prev_s}->{now_s}" if flag_s == 1 else now_s

                flag_d  = int(row.get('距離変更フラグ', 0) or 0)
                prev_d  = row.get('_前走距離') if '_前走距離' in df_res.columns else np.nan
                now_d   = row.get('距離', np.nan)
                if flag_d == 1 and not pd.isna(prev_d):
                    diff = int(float(now_d)) - int(float(prev_d))
                    dist_str = f"変更:{int(float(prev_d))}m->{int(float(now_d))}m({'+' if diff>0 else ''}{diff}m)"
                else:
                    try:    dist_str = f"{int(float(now_d))}m"
                    except: dist_str = '不明'

                change_rows.append({
                    '馬番':     row['馬番'],
                    '馬名':     row['馬名'],
                    '休養日数': fmt_kyuyo(row.get('休養日数', np.nan)),
                    '騎手変化': jockey_str,
                    '馬場変化': surf_str,
                    '距離変化': dist_str,
                })
            change_df = pd.DataFrame(change_rows)

            def highlight_change(row):
                styles = [''] * len(row)
                cols = list(row.index)
                for c in ['騎手変化', '馬場変化', '距離変化']:
                    if c in cols and '変更:' in str(row.get(c, '')):
                        styles[cols.index(c)] = 'color:#FFA500;font-weight:bold'
                if '休養日数' in cols:
                    v = str(row.get('休養日数', ''))
                    if '休み明け' in v:  styles[cols.index('休養日数')] = 'color:#888888'
                    elif '超短' in v:    styles[cols.index('休養日数')] = 'color:#FF6666'
                return styles

            st.dataframe(change_df.style.apply(highlight_change, axis=1), use_container_width=True, hide_index=True)
            st.caption("オレンジ=変化あり / [超短]=14日以内 / [休み明け]=56日以上")

        # スピード指数バーチャート
        if '近5走_中央値スピード指数' in df_res.columns:
            st.markdown("#### 📊 地力比較チャート")
            chart_data = df_res[['馬名', '近5走_中央値スピード指数']].copy()
            if '近5走_最高スピード指数' in df_res.columns:
                chart_data['近5走_最高スピード指数'] = df_res['近5走_最高スピード指数']
            chart_data = chart_data.dropna(subset=['近5走_中央値スピード指数'])
            chart_data = chart_data.sort_values('近5走_中央値スピード指数', ascending=False).head(12)
            base = alt.Chart(chart_data).encode(y=alt.Y('馬名:N', sort='-x', title=''))
            bar  = base.mark_bar(color='#4B8BFF', opacity=0.75).encode(
                x=alt.X('近5走_中央値スピード指数:Q', title='スピード指数'),
                tooltip=['馬名', '近5走_中央値スピード指数']
            )
            layer = bar
            if '近5走_最高スピード指数' in chart_data.columns:
                tick = base.mark_tick(color='#FF4B4B', thickness=2, size=15).encode(
                    x='近5走_最高スピード指数:Q', tooltip=['馬名', '近5走_最高スピード指数']
                )
                layer = bar + tick
            st.altair_chart(layer.properties(height=max(220, len(chart_data)*30)), use_container_width=True)
            st.caption("青バー=近5走中央値(地力) / 赤ティック=近5走最高値(ポテンシャル)")

    with tab4:
        st.markdown("AI勝率から計算した複合馬券の理論期待値です。**1.0以上**が購入検討ライン。")
        probs = df_res['勝率(AI予測)'].values
        odds_list = df_res['単勝オッズ'].values
        names = df_res['馬名'].values
        nums = df_res['馬番'].values

        # 複勝率（近似）
        fukusho_probs = np.clip(probs * 2.8, 0, 0.99)

        # 馬連・ワイドの期待値（上位5頭の組み合わせ）
        umaren_rows, wide_rows = [], []
        for a in range(min(5, len(probs))):
            for b in range(a+1, min(7, len(probs))):
                # 馬連: aかbが1着でもう一方が2着
                p_umaren = probs[a]*fukusho_probs[b] + probs[b]*fukusho_probs[a]
                p_umaren = min(p_umaren, 0.99)
                # ワイド: 両馬が3着以内
                p_wide = fukusho_probs[a] * fukusho_probs[b] * 1.5  # 相関補正
                p_wide = min(p_wide, 0.99)
                # 単勝オッズから馬連オッズを推定（簡易モデル: 単勝の積÷0.7）
                est_umaren_odds = (odds_list[a] * odds_list[b]) / 8.0
                est_wide_odds   = (odds_list[a] * odds_list[b]) / 20.0
                ev_umaren = p_umaren * est_umaren_odds
                ev_wide   = p_wide   * est_wide_odds
                umaren_rows.append({'組合せ': f'{nums[a]}-{nums[b]}', '馬名': f'{names[a]} - {names[b]}',
                                    '推定EV': round(ev_umaren, 2), '理論的中率': f'{p_umaren*100:.1f}%',
                                    '推定オッズ': f'{est_umaren_odds:.1f}倍'})
                wide_rows.append({'組合せ': f'{nums[a]}-{nums[b]}', '馬名': f'{names[a]} - {names[b]}',
                                  '推定EV': round(ev_wide, 2), '理論的中率': f'{p_wide*100:.1f}%',
                                  '推定オッズ': f'{est_wide_odds:.1f}倍'})

        # 3連複（上位3頭固定）
        sanrenpuku_rows = []
        for a in range(min(4, len(probs))):
            for b in range(a+1, min(5, len(probs))):
                for c in range(b+1, min(6, len(probs))):
                    p3 = fukusho_probs[a] * fukusho_probs[b] * fukusho_probs[c] * 3.0
                    p3 = min(p3, 0.99)
                    est_odds3 = (odds_list[a] * odds_list[b] * odds_list[c]) / 20.0
                    ev3 = p3 * est_odds3
                    sanrenpuku_rows.append({'組合せ': f'{nums[a]}-{nums[b]}-{nums[c]}',
                                            '推定EV': round(ev3, 2), '理論的中率': f'{p3*100:.1f}%',
                                            '推定オッズ': f'{est_odds3:.0f}倍'})

        def color_ev(val):
            if isinstance(val, float):
                if val >= 1.5: return 'color:#FF4B4B; font-weight:bold'
                if val >= 1.0: return 'color:#FFA500; font-weight:bold'
            return ''

        sub1, sub2, sub3 = st.tabs(["馬連", "ワイド", "3連複"])
        with sub1:
            st.caption("※ オッズは単勝オッズから推定した理論値です。実際のオッズとは異なります。")
            df_uma = pd.DataFrame(umaren_rows).sort_values('推定EV', ascending=False)
            st.dataframe(df_uma.style.applymap(color_ev, subset=['推定EV']).format({'推定EV': '{:.2f}'}), use_container_width=True, hide_index=True)
        with sub2:
            df_wid = pd.DataFrame(wide_rows).sort_values('推定EV', ascending=False)
            st.dataframe(df_wid.style.applymap(color_ev, subset=['推定EV']).format({'推定EV': '{:.2f}'}), use_container_width=True, hide_index=True)
        with sub3:
            df_san = pd.DataFrame(sanrenpuku_rows).sort_values('推定EV', ascending=False).head(10)
            st.dataframe(df_san.style.applymap(color_ev, subset=['推定EV']).format({'推定EV': '{:.2f}'}), use_container_width=True, hide_index=True)


if action in ["⏩ 次のレースを予想", "🔍 レースを指定して予想"]:
    todays_races = get_todays_races()
    if not todays_races: st.warning(f"本日 ({now.strftime('%Y/%m/%d')}) はJRAのレースが開催されていません。")
    else:
        if action == "⏩ 次のレースを予想":
            st.subheader("🕒 まもなく出走するレース")
            races_sorted_by_time = sorted(todays_races, key=lambda x: x['time'])
            next_race = next((r for r in races_sorted_by_time if r['time'] > now), None)

            if next_race:
                mins_left = int((next_race['time'] - now).total_seconds() / 60)
                st.info(f"👉 **{next_race['place']} {next_race['num']}R** 「{next_race['title']}」 (あと **{mins_left}** 分)")

                # ── オッズ自動再取得 ─────────────────────────────────
                # Discord通知: 発走15分前（GitHub Actionsの遅延を考慮して余裕を持たせる）
                # 画面更新:    発走5分前（最新オッズで予想を更新）
                auto_refresh = st.checkbox(
                    "🔄 発走前に自動でオッズ再取得・Discord通知する",
                    value=False,
                    help="15分前にDiscord通知 → 5分前に画面の予想を最新オッズで更新します"
                )
                col_btn1, col_btn2 = st.columns([2, 1])
                with col_btn1:
                    manual_run = st.button("🚀 keiba-ebye 予想起動！", type="primary")
                with col_btn2:
                    force_refresh = st.button("🔄 オッズ再取得して更新", help="最新オッズで予想を再実行します")

                # 自動トリガー判定
                # discord_triggered: 15分前に一度だけ → Discordに通知
                # auto_triggered:     5分前に一度だけ → 画面の予想を更新
                discord_triggered = False
                auto_triggered = False

                if auto_refresh:
                    _last_discord_key = f'last_discord_{next_race["id"]}'
                    _last_refresh_key = f'last_auto_{next_race["id"]}'

                    # Discord通知: 発走25〜40分前の間に一度だけ発火
                    # GitHub Actionsのcronは最大30分遅れることがある。
                    # 締め切り2分前（発走2〜3分前）に届くには40分前にキューに入れる必要がある。
                    # 25〜40分前の窓で発火させることで「遅くとも発走10分前には届く」を保証する。
                    if 25 <= mins_left <= 40:
                        if not st.session_state.get(_last_discord_key, False):
                            discord_triggered = True
                            st.session_state[_last_discord_key] = True
                            st.info(f"📤 発走{mins_left}分前！Discord通知をキューに追加します（GitHub Actions経由で数分〜30分以内に届きます）")

                    # 画面更新: 発走0〜6分前に一度だけ発火（最新オッズ取得）
                    if 0 <= mins_left <= 6:
                        last_auto = st.session_state.get(_last_refresh_key, 0)
                        if time.time() - last_auto > 300:
                            auto_triggered = True
                            st.session_state[_last_refresh_key] = time.time()
                            st.warning(f"⚡ 発走{mins_left}分前！最新オッズで予想を更新します...")


                live_update = st.button("🔄 直前オッズ・馬体重で最新情報を取得し再予測", use_container_width=True)
                if manual_run or force_refresh or auto_triggered or discord_triggered or live_update:
                    with st.spinner('AIが推論中（最新オッズ取得含む）...'):
                        res_df, topics, reco, pace_text, conf_text, _, _, _, err_log = run_real_prediction(next_race['id'], now.strftime('%Y-%m-%d'), bundle, skip_live_scrape=False)
                else:
                    res_df, topics, reco, pace_text, conf_text, _, _, _, err_log = get_morning_prediction(next_race['id'], now.strftime('%Y-%m-%d'), bundle)

                if res_df is not None:
                    display_result(res_df, topics, reco, pace_text, conf_text)
                    if force_refresh or auto_triggered:
                        st.success("✅ オッズを再取得して予想を更新しました")

                    # ── Discord自動投稿（discord_triggered: 15分前）──────
                    if discord_triggered and _discord_enabled and _DISCORD_WEBHOOK_URL:
                        _race_info = {
                            'place':   next_race['place'],
                            'num':     next_race['num'],
                            'title':   next_race['title'],
                            'mins_left': mins_left,
                            'race_id': next_race['id'],   # 重複防止キー
                        }
                        _sent_key = f'discord_sent_{next_race["id"]}'
                        if not st.session_state.get(_sent_key, False):
                            _ok = send_discord_prediction(
                                res_df, topics, reco, pace_text, conf_text,
                                _race_info, _DISCORD_WEBHOOK_URL
                            )
                            if _ok:
                                st.session_state[_sent_key] = True
                                st.success("📤 Discordに予想を投稿しました！")
                            else:
                                st.warning("⚠️ Discord投稿に失敗しました（ログを確認してください）")

                    # ── 手動Discord投稿ボタン ─────────────────────────
                    if _DISCORD_WEBHOOK_URL:
                        _discord_btn_key = f'discord_btn_{next_race["id"]}'
                        if st.button("📤 Discordに投稿", key=_discord_btn_key,
                                     help="この予想をDiscordに手動で投稿します"):
                            _race_info = {
                                'place':   next_race['place'],
                                'num':     next_race['num'],
                                'title':   next_race['title'],
                                'mins_left': mins_left,
                                'race_id': f"manual_{next_race['id']}",  # 手動は別キー
                            }
                            _ok = send_discord_prediction(
                                res_df, topics, reco, pace_text, conf_text,
                                _race_info, _DISCORD_WEBHOOK_URL
                            )
                            if _ok:
                                st.success("📤 Discordに投稿しました！")
                            else:
                                st.error("❌ Discord投稿失敗（Webhook URLを確認してください）")
                else: display_error_log(err_log)

                # 自動更新チェック用ページ再読み込み
                if auto_refresh and mins_left > 6:
                    if mins_left > 20:
                        st.caption(f"発走{mins_left}分前 — 15〜20分前になるとDiscord通知、5分前に予想を更新します")
                    elif mins_left > 6:
                        st.caption(f"発走{mins_left}分前 — Discord通知済み。5分前に最新オッズで予想を更新します")
            else:
                st.success("🏁 本日の全レースは終了しました。")
            
        elif action == "🔍 レースを指定して予想":
            options = [f"{r['place']} {r['num']}R - {r['title']}" for r in todays_races]
            selected = st.selectbox("レースを選んでください", options)
            target_race = todays_races[options.index(selected)]

            live_update = st.button("🔄 直前オッズ・馬体重で最新情報を取得し再推論", use_container_width=True)
            if st.button("🚀 朝版 予想開始", type="primary") or live_update:
                with st.spinner('推論中...'):
                    if live_update:
                        res_df, topics, reco, pace_text, conf_text, _, _, _, err_log = run_real_prediction(target_race['id'], now.strftime('%Y-%m-%d'), bundle, skip_live_scrape=False)
                    else:
                        res_df, topics, reco, pace_text, conf_text, _, _, _, err_log = get_morning_prediction(target_race['id'], now.strftime('%Y-%m-%d'), bundle)

                    if res_df is not None:
                        display_result(res_df, topics, reco, pace_text, conf_text)
                        # 手動Discord投稿ボタン
                        if _DISCORD_WEBHOOK_URL:
                            if st.button("📤 Discordに投稿", key=f"discord_spec_{target_race['id']}"):
                                _race_info = {
                                    'place':   target_race['place'],
                                    'num':     target_race['num'],
                                    'title':   target_race['title'],
                                    'mins_left': 0,
                                    'race_id': f"manual_{target_race['id']}",
                                }
                                _ok = send_discord_prediction(
                                    res_df, topics, reco, pace_text, conf_text,
                                    _race_info, _DISCORD_WEBHOOK_URL
                                )
                                st.success("📤 投稿しました！") if _ok else st.error("❌ 投稿失敗")
                    else: display_error_log(err_log)

elif action == "📅 今週末の全レース予想":
    st.subheader("📅 今週末 (土・日) の先取り予想")
    sat_str, sun_str = get_weekend_dates()
    col1, col2 = st.columns(2)
    with col1: run_sat = st.button(f"🚀 土曜日 ({sat_str[4:6]}/{sat_str[6:]}) の予想", type="primary")
    with col2: run_sun = st.button(f"🚀 日曜日 ({sun_str[4:6]}/{sun_str[6:]}) の予想", type="primary")

    if run_sat or run_sun:
        _td = sat_str if run_sat else sun_str
        st.session_state["weekend_date"] = _td
        st.session_state["weekend_results"] = []
        with st.spinner("出馬表を収集中..."):
            _races = get_todays_races(_td)
        if not _races:
            st.error("出馬表が未発表です。")
        else:
            _bar = st.progress(0, text="推論中...")
            _results = []
            for _i, _r in enumerate(_races):
                # レースごとにリアルタイム表示
                with st.expander(f"🏁 {_r['place']} {_r['num']}R", expanded=True):
                    _res_df, _topics, _reco, _pace, _conf, _track, _place, _dist, _elog = run_real_prediction(
                        _r["id"], f"{_td[:4]}-{_td[4:6]}-{_td[6:]}", bundle)
                    _ww = [e for e in (_elog or []) if "枠順未確定" in e]
                    if _ww: st.warning(_ww[0])
                    if _res_df is not None:
                        display_result(_res_df, _topics, _reco, _pace, _conf)
                        _results.append({
                            "date": f"{_td[:4]}年{_td[4:6]}月{_td[6:]}日",
                            "place": _place or _r["place"], "num": _r["num"],
                            "track": _track, "dist": _dist,
                            "pace": _pace, "confidence": _conf,
                            "df": _res_df, "topics": _topics, "reco": _reco,
                        })
                    else:
                        display_error_log(_elog)
                time.sleep(1.0)
                _bar.progress((_i + 1) / len(_races))
            # 完了後にsession_stateへ保存
            st.session_state["weekend_results"] = _results

    # ダウンロードボタンはsession_stateから（再実行でも消えない）
    _cached = st.session_state.get("weekend_results", [])
    _td2    = st.session_state.get("weekend_date", "")
    _valid  = [r for r in _cached if r.get("df") is not None]
    if _valid and _td2:
        st.markdown("---")
        _c1, _c2 = st.columns(2)
        _c1.download_button(
            f"📥 {_td2[4:6]}/{_td2[6:]} 予想レポート(.txt)",
            data=generate_txt_report(_valid),
            file_name=f"keiba_weekend_{_td2}.txt",
            mime="text/plain",
            key="dl_txt_weekend",
        )
        _hw = generate_pdf_report(_valid)
        if _hw:
            _c2.download_button(
                f"🌐 {_td2[4:6]}/{_td2[6:]} 予想レポート(.html)",
                data=_hw,
                file_name=f"keiba_weekend_{_td2}.html",
                mime="text/html",
                key="dl_html_weekend",
            )

elif action == "📝 1日の振り返り (答え合わせ)":
    st.subheader("📝 1日のレース結果とAI予想の答え合わせ")
    target_date = st.date_input("振り返りたい日付を選択", datetime.date.today() - datetime.timedelta(days=1))

    if st.button("🚀 振り返り実行！", type="primary"):
        with st.spinner(f'{target_date.strftime("%Y/%m/%d")} のレースデータと結果を取得・集計中...'):
            races = get_todays_races(target_date.strftime('%Y%m%d'))
            if not races:
                st.error("指定した日付のレースが見つかりません。")
            else:
                my_bar = st.progress(0, text="集計中...")

                stats = {
                    'honmei_races': 0, 'honmei_tan_hits': 0, 'honmei_tan_return': 0,
                    'honmei_fuku_hits': 0, 'honmei_fuku_return': 0,
                    'umaren_races': 0, 'umaren_invest': 0, 'umaren_hits': 0, 'umaren_return': 0,
                    'wide_ana_races': 0, 'wide_ana_invest': 0, 'wide_ana_hits': 0, 'wide_ana_return': 0,
                    'ev_invest': 0, 'ev_tan_hits': 0, 'ev_tan_return': 0, 'ev_fuku_hits': 0, 'ev_fuku_return': 0,
                    'shiba_races': 0, 'shiba_return': 0, 'dart_races': 0, 'dart_return': 0,
                    'exp_races': 0, 'exp_return': 0, 'new_races': 0, 'new_return': 0,
                }

                for i, r in enumerate(races):
                    res_df, topics, reco, pace_text, conf_text, track_type, place, dist, err_log = run_real_prediction(r['id'], target_date.strftime('%Y-%m-%d'), bundle, skip_live_scrape=True)
                    payouts = get_all_payouts(r['id'])

                    # =========================================================
                    # レースごとの予想を expander で表示（★追加）
                    # =========================================================
                    honmei_name = res_df.iloc[0]['馬名'] if res_df is not None else "不明"
                    honmei_num  = res_df.iloc[0]['馬番'] if res_df is not None else "-"
                    tan_pay = payouts['tansho'].get(honmei_num, 0) if res_df is not None else 0
                    hit_icon = "✅" if tan_pay > 0 else ("❌" if res_df is not None and payouts['tansho'] else "⚠️")
                    expander_label = f"{hit_icon} {r['place']} {r['num']}R  ◎{honmei_num}番 {honmei_name}"
                    if tan_pay > 0:
                        expander_label += f"  → 単勝 {tan_pay/100:.1f}倍 的中！"

                    with st.expander(expander_label, expanded=False):
                        if res_df is not None:
                            display_result(res_df, topics, reco, pace_text, conf_text, show_change_table=False)
                            # 払い戻し結果を表示
                            if payouts['tansho']:
                                st.markdown("##### 📋 払い戻し結果")
                                result_rows = []
                                for rank_i, row in res_df.iterrows():
                                    uma = row['馬番']
                                    tan = payouts['tansho'].get(uma, 0)
                                    fuku = payouts['fukusho'].get(uma, 0)
                                    if rank_i < 5 or tan > 0 or fuku > 0:
                                        result_rows.append({
                                            '印': row['印'],
                                            '馬番': uma,
                                            '馬名': row['馬名'],
                                            'AI勝率': f"{row['勝率(AI予測)']*100:.1f}%",
                                            '単勝払戻': f"¥{tan:,}" if tan > 0 else '-',
                                            '複勝払戻': f"¥{fuku:,}" if fuku > 0 else '-',
                                        })
                                if result_rows:
                                    st.dataframe(pd.DataFrame(result_rows), use_container_width=True, hide_index=True)
                            else:
                                st.warning("払い戻しデータが取得できませんでした")
                        else:
                            display_error_log(err_log)

                    # =========================================================
                    # 集計処理（従来通り）
                    # =========================================================
                    if res_df is not None and payouts['tansho']:
                        honmei = res_df.iloc[0]['馬番']
                        has_unraced = ('新馬' in r['title']) or ('未出走' in r['title'])

                        stats['honmei_races'] += 1
                        if track_type == "芝": stats['shiba_races'] += 1
                        elif track_type == "ダート": stats['dart_races'] += 1
                        if has_unraced: stats['new_races'] += 1
                        else: stats['exp_races'] += 1

                        if honmei in payouts['tansho']:
                            stats['honmei_tan_hits'] += 1
                            stats['honmei_tan_return'] += payouts['tansho'][honmei]
                            if track_type == "芝": stats['shiba_return'] += payouts['tansho'][honmei]
                            elif track_type == "ダート": stats['dart_return'] += payouts['tansho'][honmei]
                            if has_unraced: stats['new_return'] += payouts['tansho'][honmei]
                            else: stats['exp_return'] += payouts['tansho'][honmei]

                        if honmei in payouts['fukusho']:
                            stats['honmei_fuku_hits'] += 1
                            stats['honmei_fuku_return'] += payouts['fukusho'][honmei]

                        if len(res_df) >= 5:
                            himo_list = res_df.iloc[1:5]['馬番'].tolist()
                            stats['umaren_races'] += 1
                            stats['umaren_invest'] += len(himo_list) * 100
                            for himo in himo_list:
                                key = tuple(sorted([honmei, himo]))
                                if key in payouts['umaren']:
                                    stats['umaren_hits'] += 1
                                    stats['umaren_return'] += payouts['umaren'][key]

                        ana_list = res_df[(res_df.index >= 4) & (res_df['期待値'] >= 1.5)]['馬番'].tolist()
                        if ana_list:
                            stats['wide_ana_races'] += 1
                            stats['wide_ana_invest'] += len(ana_list) * 100
                            for ana in ana_list:
                                key = tuple(sorted([honmei, ana]))
                                if key in payouts['wide']:
                                    stats['wide_ana_hits'] += 1
                                    stats['wide_ana_return'] += payouts['wide'][key]

                        ev_list = res_df[(res_df.index < 5) & (res_df['期待値'] >= 1.5)]['馬番'].tolist()
                        if ev_list:
                            stats['ev_invest'] += len(ev_list) * 100
                            for ev in ev_list:
                                if ev in payouts['tansho']:
                                    stats['ev_tan_hits'] += 1
                                    stats['ev_tan_return'] += payouts['tansho'][ev]
                                if ev in payouts['fukusho']:
                                    stats['ev_fuku_hits'] += 1
                                    stats['ev_fuku_return'] += payouts['fukusho'][ev]

                    time.sleep(0.5)
                    my_bar.progress((i + 1) / len(races))
                
                # 計算
                tan_rate = (stats['honmei_tan_return'] / (stats['honmei_races'] * 100) * 100) if stats['honmei_races'] > 0 else 0
                fuku_rate = (stats['honmei_fuku_return'] / (stats['honmei_races'] * 100) * 100) if stats['honmei_races'] > 0 else 0
                uma_rate = (stats['umaren_return'] / stats['umaren_invest'] * 100) if stats['umaren_invest'] > 0 else 0
                wide_rate = (stats['wide_ana_return'] / stats['wide_ana_invest'] * 100) if stats['wide_ana_invest'] > 0 else 0
                ev_tan_rate = (stats['ev_tan_return'] / stats['ev_invest'] * 100) if stats['ev_invest'] > 0 else 0
                ev_fuku_rate = (stats['ev_fuku_return'] / stats['ev_invest'] * 100) if stats['ev_invest'] > 0 else 0
                shiba_rate = (stats['shiba_return'] / (stats['shiba_races'] * 100) * 100) if stats['shiba_races'] > 0 else 0
                dart_rate = (stats['dart_return'] / (stats['dart_races'] * 100) * 100) if stats['dart_races'] > 0 else 0
                exp_rate = (stats['exp_return'] / (stats['exp_races'] * 100) * 100) if stats['exp_races'] > 0 else 0
                new_rate = (stats['new_return'] / (stats['new_races'] * 100) * 100) if stats['new_races'] > 0 else 0

                # CSVセーブ
                csv_file = "ai_daily_history.csv"
                daily_data = pd.DataFrame([{
                    '日付': target_date.strftime('%Y/%m/%d'),
                    '本命単勝回収率': round(tan_rate, 1),
                    '本命複勝回収率': round(fuku_rate, 1),
                    '穴馬単勝回収率': round(ev_tan_rate, 1),
                    '穴馬複勝回収率': round(ev_fuku_rate, 1)
                }])
                if os.path.exists(csv_file):
                    existing_df = pd.read_csv(csv_file)
                    for col in ['本命単勝回収率', '本命複勝回収率', '穴馬単勝回収率', '穴馬複勝回収率']:
                        if col not in existing_df.columns: existing_df[col] = 0.0
                    existing_df = existing_df[existing_df['日付'] != target_date.strftime('%Y/%m/%d')]
                    updated_df = pd.concat([existing_df, daily_data])
                    updated_df.to_csv(csv_file, index=False)
                else: daily_data.to_csv(csv_file, index=False)

                # HF Dataset Hubにai_daily_history.csvを保存（再起動対策）
                if _HF_TOKEN and _HF_REPO_ID:
                    try:
                        from huggingface_hub import HfApi as _HfApi
                        _api = _HfApi(token=_HF_TOKEN)
                        _api.upload_file(
                            path_or_fileobj=csv_file,
                            path_in_repo="ai_daily_history.csv",
                            repo_id=_HF_REPO_ID,
                            repo_type="dataset",
                            commit_message=f"成績履歴更新 {target_date.strftime('%Y-%m-%d')}",
                            token=_HF_TOKEN,
                        )
                    except Exception: pass

                st.markdown("---")
                st.markdown(f"### 🏆 {target_date.strftime('%Y/%m/%d')} レース振り返りレポート")
                st.markdown(f"**対象レース数: {stats['honmei_races']} レース**")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.success("🎯 【本命(◎) 単勝・複勝成績】")
                    st.write(f"- **単勝 的中率**: {(stats['honmei_tan_hits'] / stats['honmei_races'] * 100):.1f}% ({stats['honmei_tan_hits']}R)")
                    st.write(f"- **単勝 回収率**: **{tan_rate:.1f}%**")
                    st.write(f"- **複勝 的中率**: {(stats['honmei_fuku_hits'] / stats['honmei_races'] * 100):.1f}% ({stats['honmei_fuku_hits']}R)")
                    st.write(f"- **複勝 回収率**: **{fuku_rate:.1f}%**")
                    st.markdown("---")
                    st.write(f"🌱 **芝** 回収率: {shiba_rate:.1f}% ({stats['shiba_races']}R)")
                    st.write(f"🏜️ **ダート** 回収率: {dart_rate:.1f}% ({stats['dart_races']}R)")
                    st.markdown("---")
                    st.write(f"📚 **既走馬のみ** 回収率: **{exp_rate:.1f}%** ({stats['exp_races']}R)")
                    st.write(f"🔰 **未出走混在** 回収率: **{new_rate:.1f}%** ({stats['new_races']}R)")
                    
                with col2:
                    st.info("🔗 【馬券シミュレーション】")
                    st.write(f"- **馬連流し (◎ → 2〜5番手へ4点)**")
                    st.write(f"  投資: ¥{stats['umaren_invest']:,} / 回収率: **{uma_rate:.1f}%** (的中 {stats['umaren_hits']}R)")
                    st.write(f"- **穴馬ワイド (◎ → 期待値特大の穴馬へ)**")
                    st.write(f"  該当: {stats['wide_ana_races']}R / 回収率: **{wide_rate:.1f}%** (的中 {stats['wide_ana_hits']}回)")
                    st.markdown("---")
                    st.warning("🔥 【上位5頭内 期待値1.5以上馬 ベタ買い】")
                    st.write(f"- 該当数: {int(stats['ev_invest']/100)} 頭")
                    st.write(f"- **単勝 回収率**: **{ev_tan_rate:.1f}%** (的中 {stats['ev_tan_hits']}頭)")
                    st.write(f"- **複勝 回収率**: **{ev_fuku_rate:.1f}%** (的中 {stats['ev_fuku_hits']}頭)")

                # ── Discord 振り返り投稿エリア ─────────────────────────────
                if _DISCORD_REVIEW_WEBHOOK_URL:
                    st.markdown("---")
                    _review_rates = {
                        'tan_rate': tan_rate, 'fuku_rate': fuku_rate,
                        'ev_tan_rate': ev_tan_rate, 'ev_fuku_rate': ev_fuku_rate,
                        'uma_rate': uma_rate, 'shiba_rate': shiba_rate, 'dart_rate': dart_rate,
                    }
                    _review_date_str = target_date.strftime('%Y/%m/%d')

                    # 自動送信チェック: 当日のレース開催日 & 17時以降 & 未送信
                    _today_jst = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))
                    _is_race_day = target_date == _today_jst.date()
                    _after_17h   = _today_jst.hour >= 17
                    _auto_sent_key = f'review_auto_sent_{target_date}'
                    _already_sent  = st.session_state.get(_auto_sent_key, False)

                    if _is_race_day and _after_17h and not _already_sent:
                        _ok = send_discord_review(stats, _review_rates,
                                                  _review_date_str, _DISCORD_REVIEW_WEBHOOK_URL)
                        if _ok:
                            st.session_state[_auto_sent_key] = True
                            st.success("📤 振り返り結果をDiscordに自動投稿しました！（17時以降の初回実行）")
                        else:
                            st.warning("⚠️ Discord自動投稿に失敗しました")

                    # 手動送信ボタン
                    _dc1, _dc2 = st.columns([3, 1])
                    with _dc1:
                        st.caption("📊 振り返り結果をDiscordに投稿できます"
                                   "（当日17時以降は振り返り実行時に自動送信）")
                    with _dc2:
                        if st.button("📤 Discordに投稿", key=f"review_discord_{target_date}",
                                     type="primary"):
                            _ok = send_discord_review(stats, _review_rates,
                                                      _review_date_str, _DISCORD_REVIEW_WEBHOOK_URL)
                            if _ok:
                                st.success("📤 振り返り結果をDiscordに投稿しました！")
                            else:
                                st.error("❌ 投稿失敗（Webhook URLを確認してください）")

elif action == "📈 長期成績分析":
    st.subheader("📈 長期成績分析 & 回収率ダッシュボード")
    import altair as alt
    csv_file = "ai_daily_history.csv"
    # ダッシュボード設定
    db_col1, db_col2 = st.columns([3,1])
    with db_col2:
        target_rate = st.number_input("目標回収率 (%)", 80, 200, 100, 5,
            help="このラインをグラフに表示します。損益分岐点=100%")
        focus_col = st.selectbox("重点分析指標",
            ['本命単勝回収率','本命複勝回収率','穴馬単勝回収率','穴馬複勝回収率'],
            help="累積損益グラフで比較する主指標")

    # 起動時にGitHubからai_daily_history.csvを取得（再起動でリセット防止）
    if not os.path.exists(csv_file) and _HF_TOKEN and _HF_REPO_ID:
        try:
            from huggingface_hub import hf_hub_download
            _hist_path = hf_hub_download(
                repo_id=_HF_REPO_ID, filename="ai_daily_history.csv",
                repo_type="dataset", token=_HF_TOKEN, cache_dir="/tmp/hf_cache"
            )
            import shutil
            shutil.copy(_hist_path, csv_file)
        except Exception: pass

    if not os.path.exists(csv_file):
        st.info("まだデータがありません。「1日の振り返り」を実行するとここにデータが蓄積されます。")
    else:
        history_df = pd.read_csv(csv_file)
        for col in ['本命単勝回収率','本命複勝回収率','穴馬単勝回収率','穴馬複勝回収率']:
            if col not in history_df.columns: history_df[col] = 0.0
        history_df['日付'] = pd.to_datetime(history_df['日付'], errors='coerce')
        history_df = history_df.dropna(subset=['日付']).sort_values('日付').reset_index(drop=True)

        if len(history_df) == 0:
            st.warning("有効なデータがありません。")
        else:
            # ── KPI サマリー ─────────────────────────────────
            n  = len(history_df)
            avg_tan  = history_df['本命単勝回収率'].mean()
            avg_fuku = history_df['本命複勝回収率'].mean()
            avg_ana  = history_df['穴馬単勝回収率'].mean()
            over100_tan  = (history_df['本命単勝回収率'] >= 100).sum()
            over100_fuku = (history_df['本命複勝回収率'] >= 100).sum()

            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("📅 集計日数",     f"{n}日")
            k2.metric("📈 本命単勝 平均", f"{avg_tan:.1f}%",
                      delta=f"{avg_tan-100:+.1f}%", delta_color="normal")
            k3.metric("📊 本命複勝 平均", f"{avg_fuku:.1f}%",
                      delta=f"{avg_fuku-100:+.1f}%", delta_color="normal")
            k4.metric("🔥 穴馬単勝 平均", f"{avg_ana:.1f}%",
                      delta=f"{avg_ana-100:+.1f}%", delta_color="normal")
            k5.metric("✅ 単勝100%超え日", f"{over100_tan}日 / {n}日",
                      f"{over100_tan/n*100:.0f}%")

            st.markdown("---")

            # ── 移動平均オプション ────────────────────────────
            if n <= 2:
                ma_window = 1
                if n == 2:
                    st.caption("データが2件のため移動平均は適用されません（3件以上で有効）。")
                else:
                    st.caption("データが1件のため移動平均は適用されません。")
            else:
                _ma_max = min(10, n - 1)          # 最大: min(10, n-1)、n>=3なら最低2
                _ma_def = min(3, _ma_max)          # default: _ma_maxを超えない
                ma_window = st.slider("移動平均ウィンドウ (日)", 1, _ma_max, _ma_def, 1)

            # ── 折れ線グラフ ─────────────────────────────────
            plot_cols = ['本命単勝回収率','本命複勝回収率','穴馬単勝回収率','穴馬複勝回収率']
            history_df['日付_str'] = history_df['日付'].dt.strftime('%Y/%m/%d')

            # 移動平均を計算
            for col in plot_cols:
                history_df[f'{col}_MA'] = history_df[col].rolling(ma_window, min_periods=1).mean()

            melted = history_df.melt(
                '日付_str',
                value_vars=[f'{c}_MA' for c in plot_cols],
                var_name='指標', value_name='回収率(%)'
            )
            melted['指標'] = melted['指標'].str.replace('_MA','')

            rule100 = alt.Chart(pd.DataFrame({'y':[100]})).mark_rule(
                color='gray', strokeDash=[4,4], opacity=0.6
            ).encode(y='y:Q')

            line = alt.Chart(melted).mark_line(point=True).encode(
                x=alt.X('日付_str:N', sort=None, title='日付'),
                y=alt.Y('回収率(%):Q', title='回収率 (%)'),
                color=alt.Color('指標:N', legend=alt.Legend(orient='bottom')),
                tooltip=['日付_str','指標','回収率(%)']
            ).properties(height=320)
            st.altair_chart(line + rule100, use_container_width=True)
            st.caption(f"灰色破線 = 損益分岐点 / {ma_window}日移動平均を表示中 / 目標: {target_rate}%")

            st.markdown("---")
            st.markdown("#### 📋 日別詳細テーブル")

            # 値でセルを色付け
            def color_rate(val):
                try:
                    v = float(val)
                    if v >= 120: return 'background-color:rgba(255,75,75,0.2);color:#c00;font-weight:bold'
                    if v >= 100: return 'background-color:rgba(255,165,0,0.15);color:#a60'
                    if v < 70:  return 'background-color:rgba(100,100,100,0.08);color:#999'
                except: pass
                return ''

            show_table = history_df[['日付_str'] + plot_cols].copy()
            show_table = show_table.rename(columns={'日付_str':'日付'}).sort_values('日付', ascending=False)
            st.dataframe(
                show_table.style.applymap(color_rate, subset=plot_cols)
                          .format({c:'{:.1f}%' for c in plot_cols}),
                use_container_width=True, hide_index=True
            )

            # ── 月別集計 ─────────────────────────────────────
            if len(history_df) >= 2:
                st.markdown("#### 📅 月別集計")
                history_df['年月'] = history_df['日付'].dt.to_period('M').astype(str)
                monthly = history_df.groupby('年月')[plot_cols].mean().round(1)
                monthly['対象日数'] = history_df.groupby('年月').size()
                st.dataframe(
                    monthly.style.applymap(color_rate, subset=plot_cols)
                           .format({c:'{:.1f}%' for c in plot_cols}),
                    use_container_width=True
                )

            # ── 累積損益シミュレーション ────────────────────
            st.markdown("---")
            st.markdown("#### 📈 累積損益シミュレーション (4戦略比較)")
            sim_unit = st.number_input("1レースあたりの賭け金 (円)", 100, 10000, 100, 100, key="sim_unit_hist")
            for col in plot_cols:
                history_df[f'損益_{col}'] = (history_df[col] - 100) / 100 * sim_unit
                history_df[f'累積_{col}'] = history_df[f'損益_{col}'].cumsum()
            history_df['損益']   = history_df[f'損益_{focus_col}']
            history_df['累積損益'] = history_df[f'累積_{focus_col}']
            history_df['日付_str2'] = history_df['日付'].dt.strftime('%m/%d')

            cum_chart = alt.Chart(history_df).mark_line(
                color='#4B8BFF', strokeWidth=2, point=alt.OverlayMarkDef(color='#4B8BFF', size=50)
            ).encode(
                x=alt.X('日付_str2:N', sort=None, title='日付', axis=alt.Axis(labelAngle=-45)),
                y=alt.Y('累積損益:Q', title='累積損益 (円)'),
                tooltip=['日付_str2', alt.Tooltip('累積損益:Q', format=',')]
            ).properties(height=220)
            # 目標回収率ライン（損益分岐点 + ユーザー設定ライン）
            rule_data = pd.DataFrame({'y': [0], 'label': ['損益0円']})
            zero_rule = alt.Chart(rule_data).mark_rule(
                color='gray', strokeDash=[4,4], opacity=0.5
            ).encode(y='y:Q')
            st.altair_chart(cum_chart + zero_rule, use_container_width=True)

            total_profit  = history_df['累積損益'].iloc[-1] if len(history_df) > 0 else 0
            total_invest_h = len(history_df) * sim_unit
            cp1, cp2, cp3 = st.columns(3)
            cp1.metric("総投資額",   f"¥{total_invest_h:,.0f}")
            cp2.metric("累積損益",   f"¥{total_profit:,.0f}",
                       delta=f"{'+' if total_profit>=0 else ''}{total_profit:.0f}円")
            cp3.metric("総合回収率", f"{(total_profit/total_invest_h+1)*100:.1f}%" if total_invest_h>0 else "N/A")

            # ── 条件別回収率ヒートマップ ────────────────────
            if len(history_df) >= 3:
                st.markdown("---")
                st.markdown("#### 🗓️ 週別回収率ヒートマップ (曜日×週)")
                history_df['曜日'] = history_df['日付'].dt.day_name().map({
                    'Monday':'月','Tuesday':'火','Wednesday':'水','Thursday':'木',
                    'Friday':'金','Saturday':'土','Sunday':'日'
                })
                history_df['週'] = history_df['日付'].dt.strftime('%m/%d週')
                hm_data = history_df.groupby(['週','曜日'])['本命単勝回収率'].mean().reset_index()
                hm_data.columns = ['週','曜日','回収率']
                heat = alt.Chart(hm_data).mark_rect(
                    cornerRadius=3, stroke='white', strokeWidth=1
                ).encode(
                    x=alt.X('週:N', title=''),
                    y=alt.Y('曜日:N', sort=['月','火','水','木','金','土','日'], title=''),
                    color=alt.Color('回収率:Q',
                        scale=alt.Scale(domain=[50,150], scheme='redblue'),
                        legend=alt.Legend(title='回収率(%)')),
                    tooltip=['週','曜日', alt.Tooltip('回収率:Q', format='.1f')]
                ).properties(height=180)
                st.altair_chart(heat, use_container_width=True)
                st.caption("赤=高回収率 / 青=低回収率。開催日(主に土日)のみ反映")

# ==========================================
# 🌟 性能試験 (バックテスト) 機能
# ==========================================
elif action == "🧪 性能試験 (バックテスト)":
    st.subheader("🧪 性能試験 (バックテスト)")
    with st.expander("ℹ️ バックテスト精度について（必読）", expanded=False):
        st.info("""**バックテスト結果の見方**

過去2週間以内のバックテストは信頼性が高いです。それより古い日付は参考値として扱ってください。

モデルは「学習データの最終日」までの全期間で学習されています。
古い日付でのバックテストでは、直近2週間より前の検証で一部の馬情報（コース適性・直近成績）に
「その時点では知り得なかった未来情報」が混入する場合があります。

今回のバージョンからリーク防止処理を追加済みです（最新_日付がバックテスト日以降の馬は前走情報をNaNマスク）。
それでも学習モデル自体は全期間データで学習されているため、完全な隔離ではありません。""")

    # ── 設定エリア ────────────────────────────────────────
    col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
    with col_cfg1:
        test_date = st.date_input("テストする日付", datetime.date.today() - datetime.timedelta(days=3))
    with col_cfg2:
        ev_threshold = st.slider("期待値フィルター", 1.0, 3.0, 1.5, 0.1,
                                  help="この値以上の期待値の馬だけをベット対象にします")
    with col_cfg3:
        bet_unit = st.number_input("1点あたりの賭け金 (円)", 100, 10000, 100, 100)

    if st.button("🔥 バックテスト実行！", type="primary"):
        with st.spinner(f'全レースを推論・集計中...'):
            test_races = get_todays_races(test_date.strftime('%Y%m%d'))
            if not test_races:
                st.error("レースが見つかりません。")
            else:
                my_bar = st.progress(0, text="集計中...")
                results_for_txt = []
                analysis_records = []  # レースごとの詳細記録

                for i, r in enumerate(test_races):
                    with st.expander(f"🏁 {r['place']} {r['num']}R"):
                        res_df, topics, reco, pace_text, conf_text, track_type, place, dist, err_log = run_real_prediction(r['id'], test_date.strftime('%Y-%m-%d'), bundle, skip_live_scrape=True)
                        t_dict, f_dict = get_payouts(r['id'])

                        if res_df is not None:
                            display_result(res_df, topics, reco, pace_text, conf_text, show_change_table=False)
                            results_for_txt.append({'date': test_date.strftime('%Y年%m月%d日'), 'place': place, 'num': r['num'], 'track': track_type, 'dist': dist, 'pace': pace_text, 'confidence': conf_text, 'df': res_df, 'topics': topics, 'reco': reco})

                            if not t_dict:
                                st.warning("⚠️ 払い戻しデータが取得できませんでした（予想は表示済み）")
                            else:
                                try:
                                    d = int(dist)
                                    if d <= 1400: d_cat = "短距離(〜1400m)"
                                    elif d <= 1600: d_cat = "マイル(1600m)"
                                    elif d <= 2200: d_cat = "中距離(1800〜2200m)"
                                    else: d_cat = "長距離(2400m〜)"
                                except: d_cat = "不明"

                                honmei = res_df.iloc[0]['馬番']
                                honmei_tan = t_dict.get(honmei, 0)
                                honmei_fuku = f_dict.get(honmei, 0)

                                ev_targets = res_df[(res_df.index < 5) & (res_df['期待値'] >= ev_threshold)]
                                for _, horse in ev_targets.iterrows():
                                    ret_t = t_dict.get(horse['馬番'], 0)
                                    ret_f = f_dict.get(horse['馬番'], 0)
                                    analysis_records.append({
                                        'レース': f"{place}{r['num']}R",
                                        '競馬場': place,
                                        '芝/ダート': track_type,
                                        '距離帯': d_cat,
                                        '馬名': horse['馬名'],
                                        '印': horse['印'],
                                        'AI勝率': horse['勝率(AI予測)'],
                                        '期待値': horse['期待値'],
                                        '単勝オッズ': horse['単勝オッズ'],
                                        '投資額': bet_unit,
                                        '単勝回収': ret_t * bet_unit // 100,
                                        '複勝回収': ret_f * bet_unit // 100,
                                        '本命単勝払戻': honmei_tan,
                                        '本命複勝払戻': honmei_fuku,
                                    })
                        else:
                            if err_log: display_error_log(err_log)
                            else: st.warning(f"⚠️ {r['place']} {r['num']}R: 取得失敗")
                    time.sleep(1.0)
                    my_bar.progress((i + 1) / len(test_races))

                # ── 集計レポート ─────────────────────────────────────
                st.markdown("---")
                st.markdown(f"### 🏆 {test_date.strftime('%Y/%m/%d')} バックテスト集計レポート")

                if not analysis_records:
                    st.warning("期待値フィルターに合致する馬がいませんでした。フィルター値を下げてみてください。")
                else:
                    import altair as alt
                    df_ana = pd.DataFrame(analysis_records)
                    total_invest   = df_ana['投資額'].sum()
                    total_tan_ret  = df_ana['単勝回収'].sum()
                    total_fuku_ret = df_ana['複勝回収'].sum()
                    tan_hits  = (df_ana['単勝回収'] > 0).sum()
                    fuku_hits = (df_ana['複勝回収'] > 0).sum()
                    tan_rate  = total_tan_ret  / total_invest * 100 if total_invest > 0 else 0
                    fuku_rate = total_fuku_ret / total_invest * 100 if total_invest > 0 else 0

                    # KPIカード
                    k1, k2, k3, k4, k5 = st.columns(5)
                    k1.metric("🎯 対象ベット数", f"{len(df_ana)}件",
                              help=f"期待値{ev_threshold}以上 × 上位5頭以内")
                    k2.metric("💰 総投資額", f"¥{total_invest:,}")
                    k3.metric("📈 単勝回収率",
                              f"{tan_rate:.1f}%",
                              f"{tan_rate-100:+.1f}%",
                              delta_color="normal")
                    k4.metric("📊 複勝回収率",
                              f"{fuku_rate:.1f}%",
                              f"{fuku_rate-100:+.1f}%",
                              delta_color="normal")
                    k5.metric("✅ 的中数",
                              f"単:{tan_hits} / 複:{fuku_hits}",
                              f"的中率 {tan_hits/len(df_ana)*100:.0f}% / {fuku_hits/len(df_ana)*100:.0f}%")

                    st.markdown("---")

                    # 損益推移グラフ
                    df_ana['損益(単)']  = df_ana['単勝回収'] - df_ana['投資額']
                    df_ana['損益(複)']  = df_ana['複勝回収'] - df_ana['投資額']
                    df_ana['累計損益(単)'] = df_ana['損益(単)'].cumsum()
                    df_ana['累計損益(複)'] = df_ana['損益(複)'].cumsum()
                    df_ana['番号'] = range(1, len(df_ana)+1)

                    st.markdown("#### 📈 累積損益推移")
                    melted = df_ana.melt('番号', value_vars=['累計損益(単)','累計損益(複)'], var_name='戦略', value_name='累計損益')
                    rule0 = alt.Chart(pd.DataFrame({'y':[0]})).mark_rule(color='gray', strokeDash=[4,4]).encode(y='y:Q')
                    line = alt.Chart(melted).mark_line(point=True).encode(
                        x=alt.X('番号:Q', title='ベット番号'),
                        y=alt.Y('累計損益:Q', title='累計損益 (円)'),
                        color='戦略:N',
                        tooltip=['番号','戦略','累計損益']
                    ).properties(height=250)
                    st.altair_chart(line + rule0, use_container_width=True)

                    st.markdown("#### 🔍 条件別成績")

                    def make_seg(df, col):
                        g = df.groupby(col).agg(
                            件数=('投資額','count'),
                            投資=('投資額','sum'),
                            単勝回収=('単勝回収','sum'),
                            複勝回収=('複勝回収','sum'),
                        ).reset_index()
                        g['単勝回収率(%)'] = (g['単勝回収']/g['投資']*100).round(1)
                        g['複勝回収率(%)'] = (g['複勝回収']/g['投資']*100).round(1)
                        g['単勝損益']=g['単勝回収']-g['投資']
                        return g[[col,'件数','投資','単勝回収率(%)','複勝回収率(%)','単勝損益']].sort_values('単勝回収率(%)',ascending=False)

                    def style_seg(df):
                        def color_row(row):
                            if row['単勝回収率(%)'] >= 120: return ['background-color:rgba(255,75,75,0.15)']*len(row)
                            if row['単勝回収率(%)'] >= 100: return ['background-color:rgba(255,165,0,0.1)']*len(row)
                            return ['']*len(row)
                        return df.style.apply(color_row,axis=1).format({'単勝回収率(%)':'{}%','複勝回収率(%)':'{}%','投資':'¥{:,}','単勝損益':'¥{:,}'})

                    bt1, bt2, bt3, bt4 = st.tabs(["⛰️ 芝/ダート", "🏟️ 競馬場", "📏 距離帯", "📋 全ベット一覧"])
                    with bt1: st.dataframe(style_seg(make_seg(df_ana,'芝/ダート')), use_container_width=True, hide_index=True)
                    with bt2: st.dataframe(style_seg(make_seg(df_ana,'競馬場')), use_container_width=True, hide_index=True)
                    with bt3:
                        sort_order = ["短距離(〜1400m)","マイル(1600m)","中距離(1800〜2200m)","長距離(2400m〜)","不明"]
                        df_d = make_seg(df_ana,'距離帯')
                        df_d = df_d.set_index('距離帯').reindex([x for x in sort_order if x in df_d['距離帯'].values]).reset_index()
                        st.dataframe(style_seg(df_d), use_container_width=True, hide_index=True)
                    with bt4:
                        show_detail = df_ana[['レース','印','馬名','AI勝率','期待値','単勝オッズ','投資額','単勝回収','複勝回収']].copy()
                        show_detail['AI勝率'] = (show_detail['AI勝率']*100).round(1).astype(str)+'%'
                        show_detail['期待値'] = show_detail['期待値'].round(2)
                        show_detail['結果'] = show_detail['単勝回収'].apply(lambda x: '✅ 的中' if x>0 else '❌')
                        def color_result(row):
                            if row['単勝回収'] > 0: return ['background-color:rgba(75,255,75,0.1)']*len(row)
                            return ['']*len(row)
                        st.dataframe(show_detail.style.apply(color_result,axis=1)
                                     .format({'期待値':'{:.2f}','単勝オッズ':'{:.1f}','投資額':'¥{:,}','単勝回収':'¥{:,}','複勝回収':'¥{:,}'}),
                                     use_container_width=True, hide_index=True)

                if results_for_txt:
                    _db1, _db2 = st.columns(2)
                    _db1.download_button("📥 バックテスト結果 (.txt)", data=generate_txt_report(results_for_txt), file_name=f"keiba_backtest_{test_date.strftime('%Y%m%d')}.txt", mime="text/plain")
                    _html_b = generate_pdf_report(results_for_txt)
                    if _html_b: _db2.download_button("🌐 バックテスト結果 (.html)", data=_html_b, file_name=f"keiba_backtest_{test_date.strftime('%Y%m%d')}.html", mime="text/html")

# 🌟 新機能: 一口馬主・推し馬向け 成長記録グラフ

# ==========================================
# ② 新・モデル検証＆AIチューニング (Phase 3実装)
# ==========================================
elif action == "📊 AIチューニング & バックテスト":
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
                    from src.features_engine import TE_COLS, create_features
                    df_bt, _ = create_features(df_bt, te_dicts)
                    ret_rate, res_df = run_timeseries_backtest(df_bt, features, cat_features, list(TE_COLS), n_splits=bt_splits, test_days=bt_days)

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
                    for col in ['着順', '単勝', '人気']: df_op[col] = pd.to_numeric(df_op[col], errors='coerce')

                    from src.optuna_tuner import run_optuna_tuning
                    from src.features_engine import TE_COLS, create_features
                    df_op, _ = create_features(df_op, te_dicts)
                    # 戻り値は (best_params, summary_str, cv_results_df) の3つ
                    best_p, msg, cv_df = run_optuna_tuning(
                        df_op, features, cat_features, list(TE_COLS), n_trials=n_trials
                    )

                    st.success(msg)
                    if best_p:
                        st.subheader("最適パラメータ")
                        st.json(best_p)
                    if cv_df is not None and not cv_df.empty:
                        st.subheader("試行結果 (上位10件)")
                        st.dataframe(cv_df.head(10), use_container_width=True)
                    st.info(
                        "✅ **適用方法**: 上記パラメータを `src/core_model.py` の "
                        "`model_win = lgb.LGBMRanker(...)` に設定して再学習してください。\n\n"
                        "※ CV回収率が100〜130%程度なら信頼できる最適化結果です。"
                    )
                except Exception as e:
                    import traceback
                    st.error(f"Optunaチューニングエラー: {e}")
                    st.code(traceback.format_exc())

elif action == "🏇 騎手・調教師フォーム分析":

    st.subheader("🏇 騎手・調教師 近況フォーム分析")
    st.info("学習データから直近の騎手・調教師の好調/不調を分析します。")

    try:
        df_form = pd.read_csv('learning_data_perfect_tier.zip', compression='zip', dtype=str)
        df_form['日付'] = pd.to_datetime(df_form['日付'], format='mixed', errors='coerce')
        df_form = df_form.dropna(subset=['日付'])
        df_form['着順'] = pd.to_numeric(df_form['着順'], errors='coerce')
        df_form['単勝'] = pd.to_numeric(df_form['単勝'], errors='coerce')
        df_form['人気'] = pd.to_numeric(df_form['人気'], errors='coerce')

        max_dt = df_form['日付'].max()
        period_days = st.slider("分析期間 (日)", 30, 180, 90, 30)
        since_dt = max_dt - pd.Timedelta(days=period_days)
        df_recent = df_form[df_form['日付'] >= since_dt].copy()
        df_recent['勝ち'] = (df_recent['着順'] == 1).astype(int)
        df_recent['複勝'] = (df_recent['着順'] <= 3).astype(int)
        df_recent['人気馬逃げ'] = ((df_recent['人気'] <= 3) & (df_recent['着順'] > 5)).astype(int)
        df_recent['穴馬激走'] = ((df_recent['人気'] >= 7) & (df_recent['着順'] <= 3)).astype(int)

        top_n = st.slider("表示件数", 10, 50, 20, 5)

        tab_j, tab_t = st.tabs(["🏅 騎手", "🏠 調教師"])

        def build_form_df(df, col):
            g = df.groupby(col).agg(
                出走数=('着順', 'count'),
                勝利数=('勝ち', 'sum'),
                複勝数=('複勝', 'sum'),
                人気馬逃げ数=('人気馬逃げ', 'sum'),
                穴馬激走数=('穴馬激走', 'sum'),
                単勝回収額=('単勝', lambda x: (df.loc[x.index][df.loc[x.index]['勝ち']==1]['単勝'] * 100).sum()),
            ).reset_index()
            g = g[g['出走数'] >= 10]
            g['勝率(%)']   = (g['勝利数']  / g['出走数'] * 100).round(1)
            g['複勝率(%)'] = (g['複勝数']  / g['出走数'] * 100).round(1)
            g['単勝回収率(%)'] = (g['単勝回収額'] / (g['出走数'] * 100) * 100).round(1)
            g['フォームスコア'] = (g['勝率(%)'] * 2 + g['複勝率(%)'] + g['単勝回収率(%)'] / 10).round(1)
            return g.sort_values('フォームスコア', ascending=False).head(top_n)

        def style_form(df):
            def color_row(row):
                if row['単勝回収率(%)'] >= 120:
                    return ['background-color: rgba(255,75,75,0.15)'] * len(row)
                elif row['単勝回収率(%)'] >= 100:
                    return ['background-color: rgba(255,165,0,0.1)'] * len(row)
                return [''] * len(row)
            return df.style.apply(color_row, axis=1).format({
                '勝率(%)': '{:.1f}%', '複勝率(%)': '{:.1f}%',
                '単勝回収率(%)': '{:.1f}%', 'フォームスコア': '{:.1f}'
            })

        with tab_j:
            if '騎手' in df_recent.columns:
                df_j = build_form_df(df_recent, '騎手')
                st.caption(f"直近{period_days}日の成績（出走10回以上、フォームスコア順）赤＝回収率120%超、橙＝100%超")
                st.dataframe(style_form(df_j), use_container_width=True, hide_index=True)
            else:
                st.warning("騎手データが見つかりません")

        with tab_t:
            if '調教師' in df_recent.columns:
                df_t = build_form_df(df_recent, '調教師')
                st.caption(f"直近{period_days}日の成績（出走10回以上、フォームスコア順）赤＝回収率120%超、橙＝100%超")
                st.dataframe(style_form(df_t), use_container_width=True, hide_index=True)
            else:
                st.warning("調教師データが見つかりません")

    except Exception as e:
        st.error(f"フォーム分析エラー: {e}")


# ==========================================
# 馬券メモ管理
# ==========================================
elif action == "📝 馬券メモ管理":
    st.subheader("📝 馬券メモ管理")
    st.caption("馬に紐づいたメモを記録します。次回出走時にハイライト表示されます。")

    MEMO_FILE = "horse_memos.json"

    def load_memos():
        # 1. まずローカルファイルを確認
        if os.path.exists(MEMO_FILE):
            try:
                with open(MEMO_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as _e:
                logger.warning(f'load_memos ファイル読み込み失敗: {_e}')
        # 2. GitHubから取得を試みる（ローカルにない場合）
        gh_token = os.environ.get("GITHUB_TOKEN", "")
        gh_repo  = os.environ.get("GITHUB_REPO", "")   # 例: "username/keiba-ebye"
        if gh_token and gh_repo:
            try:
                api_url = f"https://api.github.com/repos/{gh_repo}/contents/{MEMO_FILE}"
                resp = requests.get(api_url,
                    headers={"Authorization": f"token {gh_token}", "Accept": "application/vnd.github.v3+json"},
                    timeout=5)
                if resp.status_code == 200:
                    import base64
                    data = json.loads(base64.b64decode(resp.json()["content"]).decode("utf-8"))
                    # ローカルにキャッシュ
                    with open(MEMO_FILE, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                    return data
            except: pass
        return {}

    def save_memos(memos):
        # ローカル保存
        with open(MEMO_FILE, "w", encoding="utf-8") as f:
            json.dump(memos, f, ensure_ascii=False, indent=2)
        # GitHub自動コミット（Secrets設定がある場合）
        gh_token = os.environ.get("GITHUB_TOKEN", "")
        gh_repo  = os.environ.get("GITHUB_REPO", "")
        if not gh_token or not gh_repo:
            return  # Secrets未設定なら無視
        try:
            import base64
            api_url  = f"https://api.github.com/repos/{gh_repo}/contents/{MEMO_FILE}"
            headers  = {"Authorization": f"token {gh_token}", "Accept": "application/vnd.github.v3+json"}
            # 既存ファイルのSHAを取得（更新時に必要）
            get_resp = requests.get(api_url, headers=headers, timeout=5)
            sha = get_resp.json().get("sha", "") if get_resp.status_code == 200 else ""
            content_b64 = base64.b64encode(
                json.dumps(memos, ensure_ascii=False, indent=2).encode("utf-8")
            ).decode("utf-8")
            payload = {
                "message": f"📝 馬券メモ更新 {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
                "content": content_b64,
            }
            if sha: payload["sha"] = sha
            put_resp = requests.put(api_url, headers=headers, json=payload, timeout=8)
            if put_resp.status_code in (200, 201):
                st.toast("✅ GitHubにメモを自動保存しました", icon="💾")
            else:
                st.toast(f"⚠️ GitHub保存失敗(status={put_resp.status_code})", icon="⚠️")
        except Exception as _e:
            st.toast(f"⚠️ GitHub自動コミットエラー: {_e}", icon="⚠️")

    memos = load_memos()

    col_l, col_r = st.columns([2, 1])

    with col_r:
        st.markdown("#### ✏️ メモを追加・編集")
        memo_horse = st.text_input("馬名", placeholder="例: ドウデュース")
        memo_tag = st.selectbox("タグ", [
            "🔴 出遅れ", "🟠 不利あり", "🟡 内有利で損",
            "🟢 好走の手応え", "🔵 距離が長かった", "⚫ 馬場が合わなかった",
            "⚪ その他",
        ])
        memo_text = st.text_area("メモ内容", placeholder="例: 3角で大外に張られ位置取り悪化。次走は巻き返し期待", height=100)
        memo_date = st.date_input("レース日", datetime.date.today())

        if st.button("💾 メモを保存", type="primary") and memo_horse.strip():
            if memo_horse not in memos:
                memos[memo_horse] = []
            memos[memo_horse].append({
                "日付": memo_date.strftime("%Y/%m/%d"),
                "タグ": memo_tag,
                "メモ": memo_text.strip(),
            })
            save_memos(memos)
            st.success(f"✅ {memo_horse} のメモを保存しました！")
            st.rerun()

    with col_l:
        st.markdown("#### 📋 登録済みメモ一覧")
        if not memos:
            st.info("まだメモがありません。右の欄からメモを追加してください。")
        else:
            search_horse = st.text_input("🔍 馬名で検索", placeholder="馬名を入力...")
            display_memos = {
                k: v for k, v in memos.items()
                if not search_horse or search_horse.lower() in k.lower()
            }
            if not display_memos:
                st.warning(f"'{search_horse}' に一致する馬のメモはありません。")
            else:
                for horse_name, memo_list in sorted(display_memos.items()):
                    with st.expander(f"🐴 {horse_name}  ({len(memo_list)}件)", expanded=False):
                        for i, memo in enumerate(sorted(memo_list, key=lambda x: x["日付"], reverse=True)):
                            tag_color = {
                                "🔴": "rgba(255,75,75,0.1)",
                                "🟠": "rgba(255,165,0,0.1)",
                                "🟡": "rgba(255,230,0,0.1)",
                                "🟢": "rgba(75,200,75,0.1)",
                                "🔵": "rgba(75,75,255,0.1)",
                                "⚫": "rgba(100,100,100,0.1)",
                            }.get(memo["タグ"][0], "rgba(200,200,200,0.05)")
                            st.markdown(
                                f'<div style="padding:8px;margin:4px 0;border-radius:6px;'
                                f'background:{tag_color};border-left:3px solid #ccc;">'
                                f'<b>{memo["日付"]}</b> {memo["タグ"]}<br>'
                                f'{memo["メモ"] or "(メモなし)"}</div>',
                                unsafe_allow_html=True
                            )
                            col_del = st.columns([4, 1])[1]
                            if col_del.button("🗑️", key=f"del_{horse_name}_{i}", help="削除"):
                                memos[horse_name].pop(i)
                                if not memos[horse_name]:
                                    del memos[horse_name]
                                save_memos(memos)
                                st.rerun()

    # 次回出走時ハイライト用: メモ対象馬名リストをセッションに保存
    if memos:
        st.session_state['memo_horses'] = list(memos.keys())

elif action == "🐴 愛馬の成長記録":
    st.subheader("🐴 愛馬の成長記録 & AIスカウティング")
    import altair as alt

    col_inp1, col_inp2 = st.columns([3, 1])
    with col_inp1:
        horse_name = st.text_input("🔍 馬名を入力", placeholder="例: ドウデュース、リバティアイランド")
    with col_inp2:
        show_n = st.number_input("最近N戦を表示", 5, 50, 20, 5)

    if st.button("📊 成長記録を表示", type="primary") and horse_name:
        with st.spinner(f"{horse_name} のデータを検索中..."):
            try:
                data_file = 'learning_data_perfect_tier.zip'
                if not os.path.exists(data_file):
                    st.error(f"データベースファイル ({data_file}) が見つかりません。")
                else:
                    df_hist = pd.read_csv(data_file, compression='zip', dtype=str)
                    df_hist['日付'] = pd.to_datetime(df_hist['日付'], format='mixed', errors='coerce')
                    df_horse = df_hist[df_hist['馬名'] == horse_name].copy()

                    if df_horse.empty:
                        # 部分一致検索
                        candidates = df_hist[df_hist['馬名'].str.contains(horse_name, na=False)]['馬名'].unique()
                        if len(candidates) > 0:
                            st.warning(f"「{horse_name}」は見つかりませんでした。似た名前: {', '.join(candidates[:5])}")
                        else:
                            st.warning(f"「{horse_name}」は見つかりませんでした。")
                    else:
                        df_horse = df_horse.sort_values('日付').dropna(subset=['日付'])
                        df_horse = df_horse.tail(show_n).copy()

                        # 数値変換
                        for col in ['着順','人気','単勝','上り','当日馬体重','馬体重増減',
                                    '補正タイム偏差','タイム差','距離','斤量','枠番','馬番']:
                            if col in df_horse.columns:
                                df_horse[col] = pd.to_numeric(df_horse[col], errors='coerce')

                        # タイム指数計算
                        if '補正タイム偏差' in df_horse.columns:
                            df_horse['タイム指数'] = (50 - df_horse['補正タイム偏差'] * 10).round(1)

                        df_horse['日付_str'] = df_horse['日付'].dt.strftime('%Y/%m/%d')

                        st.success(f"✅ {len(df_horse)}戦分のデータ（直近{show_n}戦）を表示中")

                        # ── サマリーカード ──────────────────────────────
                        total_n = len(df_horse)
                        wins    = (df_horse['着順'] == 1).sum()
                        top3    = (df_horse['着順'] <= 3).sum()
                        avg_pop = df_horse['人気'].mean()
                        avg_idx = df_horse['タイム指数'].mean() if 'タイム指数' in df_horse.columns else None
                        best_idx = df_horse['タイム指数'].max() if 'タイム指数' in df_horse.columns else None

                        k1,k2,k3,k4,k5 = st.columns(5)
                        k1.metric("🏅 成績",     f"{wins}勝/{total_n}戦",
                                  f"複勝 {top3}/{total_n}")
                        k2.metric("🎯 勝率",      f"{wins/total_n*100:.1f}%",
                                  f"複勝率 {top3/total_n*100:.1f}%")
                        k3.metric("👥 平均人気", f"{avg_pop:.1f}番人気")
                        if avg_idx: k4.metric("📊 平均タイム指数", f"{avg_idx:.1f}")
                        if best_idx: k5.metric("🚀 最高タイム指数", f"{best_idx:.1f}")

                        st.markdown("---")

                        # ── タブ構成 ───────────────────────────────────
                        tab_idx, tab_agari, tab_weight, tab_rank, tab_table = st.tabs([
                            "📈 タイム指数", "💨 末脚・タイム差", "⚖️ 馬体重", "👑 着順・人気", "📋 詳細テーブル"
                        ])

                        def make_line(data, x, y, color='#4B8BFF', title='', reverse_y=False, zero_line=False):
                            if data.empty: return None
                            scale = alt.Scale(reverse=reverse_y)
                            chart = alt.Chart(data).mark_line(point=True, color=color).encode(
                                x=alt.X(f'{x}:N', sort=None, title='日付'),
                                y=alt.Y(f'{y}:Q', scale=scale, title=title),
                                tooltip=[f'{x}:N', f'{y}:Q']
                            ).interactive()
                            if zero_line:
                                rule = alt.Chart(pd.DataFrame({'y':[0]})).mark_rule(
                                    color='gray', strokeDash=[4,4]).encode(y='y:Q')
                                return chart + rule
                            return chart

                        with tab_idx:
                            st.caption("数値が高いほど優秀なパフォーマンスです。")
                            if 'タイム指数' in df_horse.columns:
                                d = df_horse[['日付_str','タイム指数','競馬場','芝/ダート','距離','着順']].dropna(subset=['タイム指数'])
                                if not d.empty:
                                    chart = alt.Chart(d).mark_line(point=True).encode(
                                        x=alt.X('日付_str:N', sort=None, title=''),
                                        y=alt.Y('タイム指数:Q', title='タイム指数'),
                                        color=alt.condition(
                                            alt.datum['着順'] == 1,
                                            alt.value('#FF4B4B'),
                                            alt.value('#4B8BFF')
                                        ),
                                        tooltip=['日付_str','タイム指数','競馬場','芝/ダート','距離','着順']
                                    ).interactive().properties(height=280)
                                    st.altair_chart(chart, use_container_width=True)
                                    st.caption("赤点 = 1着")
                            else:
                                st.info("タイム指数データがありません（補正タイム偏差列が必要）")

                        with tab_agari:
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.caption("上がり3F (低いほど末脚切れる)")
                                if '上り' in df_horse.columns:
                                    d = df_horse[['日付_str','上り']].dropna(subset=['上り'])
                                    if not d.empty:
                                        c = make_line(d,'日付_str','上り','#FFA500','上がり3F(秒)',reverse_y=True)
                                        if c: st.altair_chart(c, use_container_width=True)
                            with col_b:
                                st.caption("1着タイム差 (0.0=1着)")
                                if 'タイム差' in df_horse.columns:
                                    d = df_horse[['日付_str','タイム差']].dropna(subset=['タイム差'])
                                    if not d.empty:
                                        c = make_line(d,'日付_str','タイム差','#FF4B4B','タイム差(秒)',reverse_y=True)
                                        if c: st.altair_chart(c, use_container_width=True)

                        with tab_weight:
                            col_w1, col_w2 = st.columns(2)
                            wt_col = '当日馬体重' if '当日馬体重' in df_horse.columns else None
                            with col_w1:
                                st.caption("馬体重推移 (kg)")
                                if wt_col:
                                    d = df_horse[['日付_str', wt_col]].replace(0, np.nan).dropna(subset=[wt_col])
                                    if not d.empty:
                                        c = make_line(d,'日付_str', wt_col,'#4BFF8B','馬体重(kg)')
                                        if c: st.altair_chart(c, use_container_width=True)
                                else:
                                    st.info("馬体重データなし")
                            with col_w2:
                                st.caption("馬体重増減 (前走比)")
                                if '馬体重増減' in df_horse.columns:
                                    d = df_horse[['日付_str','馬体重増減']].dropna(subset=['馬体重増減'])
                                    if not d.empty:
                                        c = make_line(d,'日付_str','馬体重増減','#888','増減(kg)',zero_line=True)
                                        if c: st.altair_chart(c, use_container_width=True)

                        with tab_rank:
                            st.caption("数値が低い（1位に近い）ほど上位。Y軸反転。")
                            rank_d = df_horse[['日付_str','着順','人気']].dropna(subset=['着順'])
                            if not rank_d.empty:
                                melted = rank_d.melt('日付_str', value_vars=['着順','人気'],
                                                     var_name='項目', value_name='順位')
                                max_v = max(18, rank_d[['着順','人気']].max().max() + 1)
                                cr = alt.Chart(melted).mark_line(point=True).encode(
                                    x=alt.X('日付_str:N', sort=None, title=''),
                                    y=alt.Y('順位:Q', scale=alt.Scale(domain=[1, max_v], reverse=True)),
                                    color=alt.Color('項目:N'),
                                    tooltip=['日付_str','項目','順位']
                                ).interactive().properties(height=250)
                                st.altair_chart(cr, use_container_width=True)

                                # 人気 vs 着順の散布図（人気より走った/凡走した分析）
                                st.markdown("##### 📊 人気 vs 着順（左下 = 人気通り勝ち / 右上 = 人気負け）")
                                sc_d = df_horse[['日付_str','人気','着順','競馬場','レース名']].dropna(subset=['人気','着順'])
                                if not sc_d.empty:
                                    sc_d['超過率'] = sc_d['着順'] - sc_d['人気']
                                    sc = alt.Chart(sc_d).mark_circle(size=100).encode(
                                        x=alt.X('人気:Q', title='人気', scale=alt.Scale(domain=[1,18])),
                                        y=alt.Y('着順:Q', title='着順', scale=alt.Scale(domain=[1,18], reverse=True)),
                                        color=alt.Color('超過率:Q',
                                            scale=alt.Scale(scheme='redblue', domain=[-8,8]),
                                            legend=alt.Legend(title='着順-人気')),
                                        tooltip=['日付_str','人気','着順','競馬場','レース名']
                                    ).properties(height=250).interactive()
                                    # 対角線 (人気=着順の線)
                                    diag_data = pd.DataFrame({'x':range(1,19),'y':range(1,19)})
                                    diag = alt.Chart(diag_data).mark_line(color='gray',strokeDash=[3,3],opacity=0.5).encode(
                                        x='x:Q', y=alt.Y('y:Q', scale=alt.Scale(reverse=True)))
                                    st.altair_chart(sc + diag, use_container_width=True)
                                    st.caption("青 = 人気より上の着順（激走）/ 赤 = 人気を下回る着順（凡走）/ 灰点線 = 人気通り")

                        with tab_table:
                            disp_cols = [c for c in [
                                '日付_str','競馬場','芝/ダート','距離','馬場','着順','人気',
                                '騎手','斤量','当日馬体重','馬体重増減','上り','タイム差','タイム指数','単勝'
                            ] if c in df_horse.columns]
                            show = df_horse[disp_cols].copy().sort_values('日付_str', ascending=False)
                            show = show.rename(columns={'日付_str':'日付'})

                            def color_rank(row):
                                try:
                                    r = int(row['着順'])
                                    if r == 1: return ['background-color:rgba(255,215,0,0.2)']*len(row)
                                    if r <= 3: return ['background-color:rgba(192,192,192,0.15)']*len(row)
                                except: pass
                                return ['']*len(row)

                            fmt = {}
                            for c in ['上り','タイム差','タイム指数','単勝']:
                                if c in show.columns: fmt[c] = '{:.1f}'
                            if '馬体重増減' in show.columns: fmt['馬体重増減'] = '{:+.0f}'
                            st.dataframe(
                                show.style.apply(color_rank, axis=1).format(fmt),
                                use_container_width=True, hide_index=True
                            )

            except Exception as e:
                import traceback
                st.error(f"データの読み込み中にエラーが発生しました: {e}")
                with st.expander("詳細エラーログ"):
                    st.code(traceback.format_exc())
