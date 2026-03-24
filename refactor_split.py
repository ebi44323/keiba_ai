import os

os.makedirs('src', exist_ok=True)

with open('app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

def write_chunk(filename, start, end, header=""):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(header + "\n")
        f.write("".join(lines[start-1:end]))

# 1. utils.py
utils_header = """import random
import re
import pandas as pd
import logging

logger = logging.getLogger('keiba_ebye')
"""
write_chunk('src/utils.py', 34, 81, utils_header)

# 2. scraper.py
scraper_header = """import requests
from bs4 import BeautifulSoup
import re
import datetime
import pytz
import logging
import pandas as pd
import numpy as np
from src.utils import get_headers

logger = logging.getLogger('keiba_ebye')
"""
# lines 514-787: get_todays_races -> get_odds_from_soup
write_chunk('src/scraper.py', 514, 787, scraper_header)
# append fetch_horse_last_race and jockey globals (1175-1286)
with open('src/scraper.py', 'a', encoding='utf-8') as f:
    f.write("\n" + "".join(lines[1174:1286]))

# 3. reports.py
reports_header = """import pandas as pd
import logging
logger = logging.getLogger('keiba_ebye')
"""
write_chunk('src/reports.py', 789, 954, reports_header)

# 4. discord_utils.py
discord_header = """import os
import json
import datetime
import logging

logger = logging.getLogger('keiba_ebye')

_HF_TOKEN   = os.environ.get("HF_TOKEN", "")
_HF_REPO_ID = os.environ.get("HF_REPO_ID", "")
"""
write_chunk('src/discord_utils.py', 967, 1169, discord_header)

# 5. core_model.py (model and data_prep combined for now to avoid circular deps)
core_model_header = """import os
import json
import datetime
import pandas as pd
import numpy as np
import lightgbm as lgb
import logging
import pytz
import random
import time
import re
from src.features_engine import NUM_FEATURES, CAT_FEATURES, TE_COLS, classify_style
from src.utils import VENUE_MAWARI, VENUE_CHIKEI, TRACK_CONDITION_MAP, classify_race_class

logger = logging.getLogger('keiba_ebye')

_HF_TOKEN   = os.environ.get("HF_TOKEN", "")
_HF_REPO_ID = os.environ.get("HF_REPO_ID", "")   
_MODEL_FILE = "keiba_model.pkl"                    
_META_FILE  = "keiba_model_meta.json"             
"""
write_chunk('src/core_model.py', 95, 500, core_model_header)

# 6. inference.py
inference_header = """import pandas as pd
import numpy as np
import requests
import json
from bs4 import BeautifulSoup
import re
import datetime
import logging
from src.utils import get_headers, resolve_name
from src.scraper import fetch_horse_last_race
from src.features_engine import classify_style

logger = logging.getLogger('keiba_ebye')
"""
# Need _safe_col (1289) and run_real_prediction (1299-1771)
write_chunk('src/inference.py', 1289, 1771, inference_header)

# 7. Move features_engine.py to src/ !! Wait, let's keep features_engine.py at root, 
# but models in src/ will import it from parent or we move it to src/
import shutil
if os.path.exists('features_engine.py'):
    shutil.copy('features_engine.py', 'src/features_engine.py')

print("Split completed.")
