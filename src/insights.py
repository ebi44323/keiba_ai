"""
insights.py — レース固有の「根拠」と「注意フラグ」の生成

シミュレーターの絵と、なぜその馬なのかを一本につなぐための materials。
**すべて res_df に既にある値から literal に作る**（新しい推定はしない）ので、
書かれていることは必ずデータで裏が取れる。再学習も不要。

  race_reasons(df, pace_label, venue, track, distance) … ◎の根拠（2〜4行）
  horse_flags(row)                                      … 1頭ぶんの注意フラグ
  race_grade_label(title)                               … G1/G2/G3/OP などの格
"""

import re
import math

# ── レース格（朝刊/アプリの色分け用）───────────────────────────────────
_GRADE_PATTERNS = [
    (re.compile(r'\bG\s*1\b|ＧI(?!I)|GI(?!I)|Ｇ１'), 'G1',   '#b8860b'),
    (re.compile(r'\bG\s*2\b|ＧII(?!I)|GII(?!I)|Ｇ２'), 'G2',   '#7d3c98'),
    (re.compile(r'\bG\s*3\b|ＧIII|GIII|Ｇ３'),        'G3',   '#2471a3'),
    (re.compile(r'リステッド|\(L\)|Ｌ'),               'L',    '#1e8449'),
    (re.compile(r'オープン|ＯＰ|OP'),                  'OP',   '#616b7b'),
]


def race_grade_label(title: str) -> tuple:
    """(格ラベル, 色) を返す。該当しなければ ('', '')。"""
    t = str(title or '')
    for pat, label, color in _GRADE_PATTERNS:
        if pat.search(t):
            return label, color
    return '', ''


def _f(row, key, default=None):
    try:
        v = row.get(key)
        if v is None:
            return default
        v = float(v)
        return default if math.isnan(v) else v
    except (TypeError, ValueError):
        return default


# ── 1頭ぶんの注意フラグ ────────────────────────────────────────────────
def horse_flags(row) -> list:
    """見落としやすい条件を拾って [(絵文字, 短文, 重要度)] で返す。

    重要度: 'warn'（危険寄り）/ 'note'（中立の注意）/ 'good'（好材料）
    """
    out = []
    w = _f(row, '馬体重増減')
    if w is not None and abs(w) >= 20:
        out.append(('⚖️', f'馬体重{w:+.0f}kg の大幅増減', 'warn'))
    elif w is not None and abs(w) >= 12:
        out.append(('⚖️', f'馬体重{w:+.0f}kg', 'note'))

    rest = _f(row, '休養日数')
    if rest is not None:
        if rest >= 180:
            out.append(('🛌', f'長期休養明け（{int(rest)}日ぶり）', 'warn'))
        elif rest >= 90:
            out.append(('🛌', f'休み明け（{int(rest)}日ぶり）', 'note'))
        elif rest <= 8:
            out.append(('🔁', f'連闘（中{int(rest)}日）', 'note'))

    if _f(row, '新馬フラグ', 0) == 1:
        out.append(('🌱', '初出走（過去データなし）', 'warn'))
    if _f(row, '乗り替わりフラグ', 0) == 1:
        out.append(('🔄', '乗り替わり', 'note'))
    if _f(row, '距離変更フラグ', 0) == 1:
        out.append(('📏', '距離変更', 'note'))
    if _f(row, '馬場替わりフラグ', 0) == 1:
        out.append(('🔀', '芝⇄ダート替わり', 'note'))
    if _f(row, 'レース格上挑戦フラグ', 0) == 1:
        out.append(('⬆️', '格上挑戦', 'note'))
    if _f(row, 'コース初挑戦フラグ', 0) == 1:
        out.append(('🆕', 'このコース初出走', 'note'))
    if str(row.get('穴馬マーク', '')) == '🎯':
        out.append(('🎯', 'モデルDの穴馬候補', 'good'))
    return out


# ── ◎の根拠（そのレース固有）──────────────────────────────────────────
_PACE_STYLE_FIT = {
    # (ペース, ゾーン) → (評価, 一言)
    ('ハイ', '逃げ'):  ('warn', '先行争いが激しく、前は苦しくなりやすい'),
    ('ハイ', '先行'):  ('warn', '前が速くなるぶん、粘りきれるかが鍵'),
    ('ハイ', '中団'):  ('good', 'ペースが上がれば差しが決まりやすい'),
    ('ハイ', '後方'):  ('good', 'ハイペースは後方からの台頭を後押しする'),
    ('スロー', '逃げ'): ('good', '楽に運べれば前残りの公算'),
    ('スロー', '先行'): ('good', '流れが落ち着けば前の位置が有利に働く'),
    ('スロー', '中団'): ('warn', '流れが遅いと差し届かないおそれ'),
    ('スロー', '後方'): ('warn', 'スローだと後方からでは届きにくい'),
}


def race_reasons(df, pace_label: str = '', venue: str = '', track: str = '',
                 distance=None, max_items: int = 4) -> list:
    """◎（先頭行）について、そのレース固有の根拠を返す。

    戻り値: [(絵文字, 文章, 重要度)]。すべて res_df の値から literal に作る。
    """
    if df is None or len(df) == 0:
        return []
    r = df.iloc[0]
    out = []

    # 1) 想定隊列とペースの相性
    zone = str(r.get('想定ゾーン', '') or '')
    pos = _f(r, '想定隊列順')
    if zone and pos:
        line = f'想定{int(pos)}番手（{zone}）'
        key = 'ハイ' if 'ハイ' in pace_label else 'スロー' if 'スロー' in pace_label else ''
        fit = _PACE_STYLE_FIT.get((key, zone))
        if fit:
            out.append(('🏇', f'{line}。{fit[1]}', fit[0]))
        else:
            out.append(('🏇', line, 'note'))

    # 2) 枠（このコースで外枠が前に行けるか）
    try:
        from src.pace_model import _draw_effect
        eff = _draw_effect(venue, track, distance)
        uma = _f(r, '馬番')
        n = len(df)
        if uma and n > 1 and abs(eff) >= 0.02:
            outer = (uma - 1) / (n - 1) > 0.5
            fwd = (eff < 0)      # 負 = 外枠ほど前
            if outer == fwd:
                out.append(('🚪', f'{int(uma)}番枠。このコースは'
                                  f'{"外" if fwd else "内"}枠が前を取りやすく、位置取りで有利', 'good'))
            else:
                out.append(('🚪', f'{int(uma)}番枠。このコースは'
                                  f'{"外" if fwd else "内"}枠が前を取りやすく、位置取りはやや不利', 'warn'))
    except Exception:
        pass

    # 3) 能力の裏づけ（近走のスピード指数）
    best = _f(r, 'ベスト3走_中央値スピード指数')
    recent = _f(r, '過去3走平均スピード指数')
    if best is not None and recent is not None:
        try:
            others = df['ベスト3走_中央値スピード指数'].astype(float)
            rank = int((others > best).sum()) + 1
            if rank <= 3:
                out.append(('⚡', f'ベスト3走のスピード指数はメンバー{rank}位', 'good'))
        except Exception:
            pass

    # 4) 妙味（EV）
    ev = _f(r, '期待値')
    odds = _f(r, '単勝オッズ')
    if ev is not None and odds is not None and ev >= 1.3:
        out.append(('💰', f'単勝{odds:.1f}倍で期待値{ev:.2f}（AI評価に対してオッズが甘い）', 'good'))

    # 5) 危険サイン
    for emo, txt, lv in horse_flags(r):
        if lv == 'warn':
            out.append((emo, f'◎にも注意点: {txt}', 'warn'))
            break
    return out[:max_items]


def danger_favorites(df, max_items: int = 2) -> list:
    """人気（低オッズ）なのにAI評価が低い馬 = 危険な人気馬。"""
    if df is None or len(df) == 0:
        return []
    out = []
    for i, (_, r) in enumerate(df.iterrows()):
        odds = _f(r, '単勝オッズ')
        win = _f(r, '勝率(AI予測)', 0) or 0
        if odds is not None and 0 < odds <= 4.0 and win < 0.10 and i >= 2:
            out.append((r.get('馬番'), r.get('馬名', ''), odds, win))
        if len(out) >= max_items:
            break
    return out
