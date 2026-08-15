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
