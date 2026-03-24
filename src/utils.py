import re
import pandas as pd
import logging

logger = logging.getLogger('keiba_ebye')

# 共通定数・ユーティリティは config.py で一元管理。ここは re-export のみ。
from src.config import (
    VENUE_MAWARI, VENUE_CHIKEI, TRACK_CONDITION_MAP,
    get_headers, _UA_LIST,
)


def classify_race_class(race_name: str) -> int:
    """レース名からクラスコードを返す（G1=9 〜 新馬=0）"""
    t = str(race_name)
    if 'G1' in t or 'GⅠ' in t or 'ＧＩ' in t: return 9
    if 'G2' in t or 'GⅡ' in t or 'ＧⅡ' in t: return 8
    if 'G3' in t or 'GⅢ' in t or 'ＧⅢ' in t: return 7
    if 'リステッド' in t or 'Listed' in t:        return 6
    if '新馬' in t:                               return 0
    if '未勝利' in t:                             return 1
    if '1勝' in t or '500万' in t:               return 2
    if '2勝' in t or '1000万' in t:              return 3
    if '3勝' in t or '1600万' in t:              return 4
    return 5  # オープン


def resolve_name(short_name, known_names):
    if pd.isna(short_name) or short_name == '不明': return '不明'
    clean_name = re.sub(r'[☆▲△◇★\n\s　]', '', str(short_name))
    clean_name = re.sub(r'\[[東西地外]\]', '', clean_name)
    clean_name = re.sub(r'(栗東|美浦)', '', clean_name)
    if not clean_name: return '不明'
    aliases = {"鮫島駿": "鮫島克駿", "鮫島良": "鮫島良太", "吉田隼": "吉田隼人",
               "武幸": "武幸四郎", "菅原明": "菅原明良"}
    if clean_name in aliases: clean_name = aliases[clean_name]
    normalized_dict = {}
    for kn in known_names:
        if pd.isna(kn): continue
        norm_kn = re.sub(r'[☆▲△◇★\n\s　]', '', str(kn))
        norm_kn = re.sub(r'\[[東西地外]\]', '', norm_kn)
        norm_kn = re.sub(r'(栗東|美浦)', '', norm_kn)
        if norm_kn not in normalized_dict: normalized_dict[norm_kn] = []
        normalized_dict[norm_kn].append(kn)
    if clean_name in normalized_dict: return sorted(normalized_dict[clean_name], key=len)[0]
    fwd = [n for nk, orig in normalized_dict.items() if nk.startswith(clean_name) for n in orig]
    if fwd: return sorted(fwd, key=len)[0]
    par = [n for nk, orig in normalized_dict.items() if clean_name in nk for n in orig]
    if par: return sorted(par, key=len)[0]
    return clean_name
