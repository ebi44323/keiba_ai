import re

with open('app_backup_before_refactor.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_app = []

# 1. Top section
new_app.extend(lines[0:33])

# 2. Imports from src
new_app.append("from src.features_engine import NUM_FEATURES, CAT_FEATURES, TE_COLS, classify_style\n")
new_app.append("from src.utils import VENUE_MAWARI, VENUE_CHIKEI, TRACK_CONDITION_MAP, classify_race_class, resolve_name, get_headers\n")
new_app.append("from src.core_model import prepare_model_and_data, _hub_label, _HF_TOKEN, _HF_REPO_ID\n")
new_app.append("from src.scraper import get_todays_races, get_weekend_dates, get_payouts, get_all_payouts, get_odds_from_soup, fetch_horse_last_race\n")
new_app.append("from src.reports import generate_pdf_report, generate_txt_report\n")
new_app.append("from src.discord_utils import _push_discord_queue, send_discord_prediction, send_discord_review, _test_discord_webhook, _DISCORD_WEBHOOK_URL\n")
new_app.append("from src.inference import run_real_prediction\n\n")

# 3. Model loading block (modified from 501-512)
model_load_block = """
_hub_available = bool(_HF_TOKEN and _HF_REPO_ID)

with st.spinner(f'AIエンジン起動中... ({_hub_label}からロード試行)'):
    bundle = prepare_model_and_data()
    (model, model_win, features, cat_features, num_features, cat_categories_dict,
     latest_horse_data, horse_course_dict, ped_dict,
     known_jockeys, known_trainers, te_dicts, global_mean, recent_return_rate, ensemble_weight,
     auc_win, auc_place) = bundle

"""
new_app.append(model_load_block)

# 4. Extract UI from line 1772 to end
ui_text = "".join(lines[1772:])

# 5. Regex replacements for run_real_prediction
# replace: run_real_prediction(arg1, arg2) -> run_real_prediction(arg1, arg2, bundle)
# replace: run_real_prediction(arg1, arg2, skip_live_scrape=True) -> run_real_prediction(arg1, arg2, bundle, skip_live_scrape=True)

# We use a custom string replacement just to handle the specific line calls safely
ui_text = re.sub(
    r'(run_real_prediction\s*\(\s*[^,]+,\s*[^,]+?)(\s*\))',
    r'\1, bundle\2',
    ui_text
)

ui_text = re.sub(
    r'(run_real_prediction\s*\(\s*[^,]+,\s*[^,]+?)(\s*,\s*skip_live_scrape=True\s*\))',
    r'\1, bundle\2',
    ui_text
)

new_app.append(ui_text)

with open('app.py', 'w', encoding='utf-8') as f:
    f.writelines(new_app)

print("app.py rewrite successful")
