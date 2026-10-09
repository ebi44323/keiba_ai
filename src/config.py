"""
config.py - 共通定数・ユーティリティ関数の集約モジュール
============================================================
PLACE_DICT, VENUE_MAWARI, VENUE_CHIKEI, TRACK_CONDITION_MAP,
get_headers(), safe_sleep() を一元管理する。
各モジュールからはここを import すること（重複定義を防ぐ）。
"""

import random
import time

PLACE_DICT = {
    '01': '札幌', '02': '函館', '03': '福島', '04': '新潟', '05': '東京',
    '06': '中山', '07': '中京', '08': '京都', '09': '阪神', '10': '小倉',
}

VENUE_MAWARI = {
    '札幌': '右回り', '函館': '右回り', '福島': '右回り', '新潟': '左回り', '東京': '左回り',
    '中山': '右回り', '中京': '左回り', '京都': '右回り', '阪神': '右回り', '小倉': '右回り',
}

VENUE_CHIKEI = {
    '札幌': '平坦', '函館': '平坦', '福島': '急坂', '新潟': '平坦', '東京': '急坂',
    '中山': '急坂', '中京': '急坂', '京都': '緩坂', '阪神': '急坂', '小倉': '平坦',
}

TRACK_CONDITION_MAP = {'良': 0, '稍重': 1, '重': 2, '不良': 3}

_UA_LIST = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
]


def get_headers():
    return {
        "User-Agent": random.choice(_UA_LIST),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "ja,en-US;q=0.7,en;q=0.3",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Referer": "https://www.netkeiba.com/",
        "Upgrade-Insecure-Requests": "1",
    }


def safe_sleep(base=1.5, jitter=1.0):
    time.sleep(base + random.uniform(0, jitter))


# ============================================================
# HTTP GET（リトライ＋指数バックオフ）
# ------------------------------------------------------------
# 2026-09-25 追加。netkeiba のスクレイプは一時的な 403/429/5xx や接続断で
# 簡単に失敗し、その結果「朝刊が0件」「振り返りが0件」といった全滅につながっていた
# （2026-04〜05 は実際に数週間データが止まった）。全スクレイプをこの関数に通して
# 短時間の失敗を吸収する。
#
# 挙動は requests.get のドロップイン置き換えになるようにしてある:
#   - 成功/リトライ不能なステータスは Response をそのまま返す
#   - 最後まで失敗したら最後の例外を送出する（＝従来どおり呼び出し側の except が拾う）
# 403 もリトライ対象にしているのは、get_headers() が毎回 UA を選び直すため
# 別の UA で通ることがあるから。
# ============================================================
RETRY_STATUS = (403, 429, 500, 502, 503, 504)


def http_get(url, headers=None, timeout=10, retries=2, backoff=1.6, **kwargs):
    """netkeiba 等への GET。一時エラーは指数バックオフで再試行する。

    retries: 追加の再試行回数（0なら1回だけ叩く）。
    """
    import requests  # 遅延 import（config を軽量に保つ）

    last_exc = None
    last_res = None
    for attempt in range(retries + 1):
        try:
            res = requests.get(url, headers=headers if headers is not None else get_headers(),
                               timeout=timeout, **kwargs)
            last_res = res
            if res.status_code not in RETRY_STATUS:
                return res
            last_exc = None
        except Exception as e:      # 接続断・タイムアウト・DNS など
            last_exc = e
        if attempt < retries:
            # 指数バックオフ＋ジッタ（同時実行のワークフローが同期しないように）
            time.sleep((backoff ** attempt) + random.uniform(0, 0.5))
    if last_res is not None:
        return last_res             # リトライ後も 403/5xx → 呼び出し側の判定に委ねる
    raise last_exc                  # 全試行が例外 → 従来どおり例外を投げる


def field_softmax_temperature(base_t, n_runners):
    """出走頭数に応じた softmax 温度（2026-08-16・小頭数の勝率膨張対策）。

    小頭数ほど softmax が一様化し、人気薄の勝率を実勢の10倍以上に膨らませていた
    （実データ: 本命勝率は ≤9頭で ~42% だが 15-18頭で ~31%。小頭数は分布が"尖る"べき）。
    そこで小頭数ほど温度を下げて softmax を尖らせ、本命を持ち上げ人気薄を圧縮する。
    ⚠️ 学習(core_model)と推論(inference)で必ず同一適用すること（キャリブレータ整合のため）。

    factor: N=6以下→0.75（25%シャープ化） / N=16以上→1.0（従来通り）の線形。
    """
    try:
        n = float(n_runners)
    except (TypeError, ValueError):
        return base_t
    factor = 0.75 + 0.25 * min(max((n - 6.0) / 10.0, 0.0), 1.0)
    return base_t * factor


# ── 事後温度補正（Phase 2a・2026-10-09）────────────────────────────────
# 本番OOS 324R（2026-08-30〜10-04）の診断で、◎の勝率が実勢の 0.58倍（市場比 0.69倍）と
# 過小評価だった。レース内で p^(1/T) 再正規化して分布を尖らせる。
# T* は 全データ/前半/後半 すべて 0.65 で一致（レースlogloss 1.5%改善・ECE 1.75→0.96pp）。
# 順位は変えない＝◎選定に影響しない・再学習不要（キャリブレータの後段に置く事後変換）。
# ⚠️ ◎のEV昇格と勝負/回避ラベルは「補正前の勝率」で判定する（inference.py の 勝率(補正前)）。
#    絶対閾値なので、補正後の勝率で判定すると勝負が 104→155R(48%) に急増するため。
#    ラベルの作り直しは Phase 2b で実ROIから行う。
# 1.0 にすると補正前と完全に同じ挙動に戻る（ロールバック用）。
POST_TEMPERATURE = 0.65


def apply_post_temperature(probs, t=None):
    """1レース分の勝率に p^(1/T) をかけて合計1に再正規化する（順位は不変）。"""
    import numpy as np
    t = POST_TEMPERATURE if t is None else float(t)
    p = np.clip(np.asarray(probs, dtype=float), 1e-9, None)
    if t == 1.0 or len(p) == 0:
        return p / p.sum() if len(p) else p
    q = p ** (1.0 / t)
    return q / q.sum()


def smooth_isotonic_predict(calibrator, x):
    """IsotonicRegression の段差を補間して滑らかにした予測（2026-10-10）。

    Isotonic は階段関数なので、近いスコアの馬が同じ勝率に潰れる（◎と2番手が同値になり、
    ◎は 1e-9 のタイブレークで決まるだけ・勝率差=0 でラベルも拮抗に寄る）。
    各段の「中心のx」と「段の値y」を結んだ折れ線で補間する（段の平均はほぼ保たれる）。
      - 最下段より下: 原点へ比例で下げる（y0 × x/x0）
      - 最上段より上: 最後の区間の傾きで延長（傾きの上限は原点比例 y/x・値の上限なし）
    単調非減少を保つので順位は変わらない。fit 済み属性が無ければ従来の predict を返す。
    """
    import numpy as np
    x = np.asarray(x, dtype=float)
    try:
        xt = np.asarray(calibrator.X_thresholds_, dtype=float)
        yt = np.asarray(calibrator.y_thresholds_, dtype=float)
    except AttributeError:
        return calibrator.predict(x)
    if len(xt) < 2:
        return calibrator.predict(x)
    uy, inv = np.unique(yt, return_inverse=True)
    cx = np.array([xt[inv == k].mean() for k in range(len(uy))])
    if len(cx) < 2:
        return calibrator.predict(x)
    out = np.interp(x, cx, uy)
    lo = x < cx[0]
    if lo.any() and cx[0] > 0:
        out[lo] = uy[0] * np.clip(x[lo], 0, None) / cx[0]
    hi = x > cx[-1]
    if hi.any():
        # 最後の区間の傾きで延長。ただし最上段はサンプルが少なく急になりがちなので、
        # 原点比例の傾き(y/x)を上限にする。上限値(0.95等)で切ると強い馬同士が同値に戻るため切らない
        # （呼び出し側はレース内で合計1に正規化するので、ここは相対的な重みとして使われる）。
        slope = (uy[-1] - uy[-2]) / max(cx[-1] - cx[-2], 1e-12)
        slope = min(slope, uy[-1] / max(cx[-1], 1e-12))
        out[hi] = uy[-1] + slope * (x[hi] - cx[-1])
    return out
