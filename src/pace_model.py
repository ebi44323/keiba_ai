"""
pace_model.py — 展開予想（想定ペース・想定隊列）

学習データから作った統計テーブル（src/pace_tables.json）を lookup するだけなので、
**LightGBMの再学習は不要**。build_pace_tables.py でテーブルを作り直せる。

★このモジュールの設計方針（実測にもとづく・2026-09-25）
  精度のないものを精密に見せない、が原則。実測値は以下のとおり:
    ペース … 想定先行馬数との相関 -0.227（芝ダ両方で単調）。ただし時系列OOSの
             R² は 芝0.12 / ダート0.057 と弱い → **秒数を断言せず5段階ラベル**。
    隊列  … 前走位置率との相関 +0.441（R²0.194）。各馬のSDは 0.275 で、
             14頭立てなら ±3.6頭分ぶれる → **点ではなく帯（中央値±SD）**で返す。
    枠順  … 全体では相関 +0.007 とほぼ無効。芝は外枠ほど後方・ダートは外枠ほど前と
             **符号が逆で打ち消し合う**ため、(競馬場×芝ダ)別のテーブルで補正する。
             効果量は最大 0.07（1頭分弱）なので、あくまで微調整。

使い方:
    from src.pace_model import predict_pace, predict_formation
    pace = predict_pace(runners, venue='東京', track='芝', distance=1600)
    form = predict_formation(runners, venue='東京', track='芝')

    runners: 各馬 dict のリスト。使うキーは
      '馬番', '前走_前半コーナー率'（0=先頭〜1=最後方。無ければ '脚質カテゴリ' で代用）
"""

import os
import json
import logging

logger = logging.getLogger('keiba_ebye')

_TABLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pace_tables.json')
_TABLES = None

# 前走位置率が取れない馬のための、脚質カテゴリ→位置率の対応（おおよその中央値）
STYLE_TO_RATE = {'逃げ': 0.05, '先行': 0.28, '差し': 0.60, '追込': 0.85, 'マクリ': 0.55}

# ペースラベル: 予測偏差を予測値自身のSDで割った z で5段階に分ける。
# （予測の振れ幅は小さいので、実測SDではなく予測SDで割らないとラベルが偏る）
PACE_LABELS = [
    (-1.0, '超ハイペース', '🔥'),
    (-0.35, 'ハイペース',   '⚡'),
    (0.35, '平均ペース',    '➖'),
    (1.0,  'スローペース',  '🐢'),
    (99.0, '超スローペース', '💤'),
]


def _tables() -> dict:
    global _TABLES
    if _TABLES is None:
        try:
            with open(_TABLE_PATH, encoding='utf-8') as f:
                _TABLES = json.load(f)
        except Exception as e:
            logger.warning(f'pace_tables.json を読めません（展開予想を無効化）: {e}')
            _TABLES = {}
    return _TABLES


def available() -> bool:
    return bool(_tables().get('pace_base'))


def _runner_rate(r: dict):
    """その馬の前走位置率（0=先頭〜1=最後方）。取れなければ脚質から代用、それも無ければ None。"""
    v = r.get('前走_前半コーナー率')
    try:
        if v is not None and 0.0 <= float(v) <= 1.0:
            return float(v)
    except (TypeError, ValueError):
        pass
    style = str(r.get('脚質カテゴリ', '') or '').strip()
    return STYLE_TO_RATE.get(style)


def _pace_baseline(venue: str, track: str, distance) -> float:
    """コース条件のベースライン（秒/F）。細かい条件から順にフォールバックする。"""
    t = _tables()
    try:
        d = int(float(distance))
    except (TypeError, ValueError):
        d = 0
    for key, tbl in ((f'{venue}|{track}|{d}', 'pace_base'),
                     (f'{track}|{d}', 'pace_fb_track_dist'),
                     (f'{venue}|{track}', 'pace_fb_venue_track')):
        v = t.get(tbl, {}).get(key)
        if v is not None:
            return float(v)
    return float(t.get('pace_fb_track', {}).get(track, 12.4))


def predict_pace(runners: list, venue: str, track: str, distance) -> dict:
    """想定ペースを返す。

    戻り値:
      label      … '超ハイペース' 〜 '超スローペース'（5段階）
      emoji      … 表示用
      z          … 予測偏差 / 予測SD（負ほど速い）
      lead_n     … 想定先行馬数（前走3番手以内相当）
      text       … 画面にそのまま出せる一文
      confident  … 前走情報が足りているか（False なら参考値）
    """
    t = _tables()
    if not t.get('pace_coef') or not runners:
        return {'label': '不明', 'emoji': '', 'z': 0.0, 'lead_n': 0,
                'text': '展開予想は利用できません', 'confident': False}

    rates = [_runner_rate(r) for r in runners]
    known = [x for x in rates if x is not None]
    n = len(runners)
    # 前走情報がほとんど無いレース（新馬戦など）は「先行馬0頭＝超スロー」と
    # 断言してしまうので、そもそも判定しない。
    if len(known) < 5:
        return {'label': '不明', 'emoji': '❓', 'z': 0.0, 'lead_n': 0,
                'baseline': round(_pace_baseline(venue, track, distance), 3),
                'deviation': 0.0, 'confident': False,
                'text': '❓ 展開は読みにくい（前走データのある馬が少ない）'}
    # 前走で3番手以内 ≒ 位置率 0.2 以下を「先行候補」とみなす
    lead_n = sum(1 for x in known if x <= 0.2)
    # 前走情報が取れない馬は平均的な出現率で補完する
    if known and len(known) < n:
        lead_n += round((n - len(known)) * (lead_n / max(len(known), 1)))

    coef = t['pace_coef'].get(track) or t['pace_coef'].get('芝')
    dev = coef[0] + coef[1] * lead_n + coef[2] * n          # 秒/F（負=速い）
    sd = (t.get('pace_pred_sd', {}).get(track)
          or t.get('pace_pred_sd', {}).get('芝') or 0.05)
    z = dev / max(sd, 1e-6)

    label, emoji = PACE_LABELS[-1][1], PACE_LABELS[-1][2]
    for thr, lb, em in PACE_LABELS:
        if z < thr:
            label, emoji = lb, em
            break

    base = _pace_baseline(venue, track, distance)
    confident = len(known) >= max(5, n * 0.6)
    note = '' if confident else '（前走データが少なく参考値）'
    text = (f'{emoji} 想定{label}｜想定先行馬 {lead_n}頭/{n}頭{note}')
    return {'label': label, 'emoji': emoji, 'z': round(z, 3), 'lead_n': int(lead_n),
            'baseline': round(base, 3), 'deviation': round(dev, 4),
            'text': text, 'confident': confident}


def predict_formation(runners: list, venue: str = '', track: str = '') -> list:
    """想定隊列を「帯」で返す（1角付近の想定位置）。

    戻り値: runners と同じ並びの list。各要素:
      馬番, mid（想定位置率 0=先頭〜1=最後方）, lo/hi（±1SD の帯）, rank（想定隊列順）,
      zone（'逃げ'/'先行'/'中団'/'後方'）, known（前走データがあったか）

    ⚠️ SD は実測で 0.275（14頭立てなら±3.6頭分）。**点ではなく帯として扱うこと。**
    """
    t = _tables()
    if not runners or not t.get('pos_slope'):
        return []
    a, b = t.get('pos_intercept', 0.26), t.get('pos_slope', 0.44)
    sd = t.get('pos_sd', 0.275)
    draw_eff = t.get('draw_effect', {}).get(f'{venue}|{track}', 0.0)
    n = len(runners)

    out = []
    for r in runners:
        rate = _runner_rate(r)
        known = rate is not None
        mid = a + b * (rate if known else 0.5)
        # 枠補正（内0基準・外で +draw_eff）。効果は小さいのであくまで微調整。
        try:
            uma = float(r.get('馬番', 0) or 0)
            if n > 1 and uma >= 1:
                mid += draw_eff * ((uma - 1) / (n - 1) - 0.5)
        except (TypeError, ValueError):
            pass
        mid = min(max(mid, 0.0), 1.0)
        out.append({
            '馬番': r.get('馬番'), '馬名': r.get('馬名', ''),
            'mid': round(mid, 4),
            'lo': round(max(0.0, mid - sd), 4),
            'hi': round(min(1.0, mid + sd), 4),
            'known': known,
        })
    # ゾーンは mid の絶対値ではなく「そのレース内での順位」で決める。
    # 回帰は条件付き平均なので値が中央へ圧縮され（最速の馬でも mid≈0.26）、
    # 絶対値で切ると全馬が「先行〜中団」に潰れて隊列として使い物にならない。
    # 一方 *順序* は保たれているので、順位パーセンタイルで切るのが正しい。
    order = sorted(range(len(out)), key=lambda i: out[i]['mid'])
    for rank, i in enumerate(order, start=1):
        pct = (rank - 1) / max(n - 1, 1)     # 0=最前 1=最後方
        out[i]['rank'] = rank
        out[i]['pct'] = round(pct, 4)
        out[i]['zone'] = ('逃げ' if pct < 0.12 else '先行' if pct < 0.38
                          else '中団' if pct < 0.70 else '後方')
    return out
