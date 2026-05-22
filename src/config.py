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
