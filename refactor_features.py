import re

with open('src/core_model.py', 'r', encoding='utf-8') as f:
    core_text = f.read()

# We need to extract the exact block from df['日付'] = ... to else: df[col] = '不明'
start_idx = core_text.find("    df['日付'] = pd.to_datetime(df['日付']")
end_search = "        else: df[col] = '不明'\n"
end_idx = core_text.find(end_search, start_idx) + len(end_search)

features_code = core_text[start_idx:end_idx]

func_str = f"""
def create_features(df, te_dicts=None):
    import pandas as pd
    import numpy as np
    import re
    from src.utils import VENUE_MAWARI, VENUE_CHIKEI, TRACK_CONDITION_MAP, classify_race_class
    
{features_code}
    return df, te_dicts
"""

with open('src/features_engine.py', 'a', encoding='utf-8') as f:
    f.write(func_str)

replacement = """    from src.features_engine import create_features
    df, _ = create_features(df)
"""
new_core = core_text[:start_idx] + replacement + core_text[end_idx:]

with open('src/core_model.py', 'w', encoding='utf-8') as f:
    f.write(new_core)

print("Refactoring completed.")
