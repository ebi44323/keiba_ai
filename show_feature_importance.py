"""
学習済みモデルからfeature importanceを表示するスクリプト。

usage:
  # HF Hubからモデルを取得して実行（初回のみダウンロード）
  $env:HF_TOKEN="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"
  $env:HF_REPO_ID="ebi44323/keiba-ebye-models"
  python show_feature_importance.py [--type gain|split] [--top N]

  # キャッシュ済みpklを直接指定
  python show_feature_importance.py --pkl keiba_model.pkl
"""
import argparse
import io
import os
import joblib
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--type', choices=['gain', 'split'], default='gain')
parser.add_argument('--top', type=int, default=0, help='上位N件のみ表示（0=全件）')
parser.add_argument('--pkl', default='', help='ローカルpklパス（省略時はHF Hubからダウンロード）')
args = parser.parse_args()

# ── モデルロード ─────────────────────────────────────────────────
if args.pkl:
    bundle = joblib.load(args.pkl)
else:
    HF_TOKEN   = os.environ.get('HF_TOKEN')
    HF_REPO_ID = os.environ.get('HF_REPO_ID', 'ebi44323/keiba-ebye-models')
    if not HF_TOKEN:
        raise SystemExit('HF_TOKEN が未設定です。$env:HF_TOKEN="hf_xxxx" をセットしてください。')
    from huggingface_hub import hf_hub_download
    print(f'HF Hubからダウンロード中: {HF_REPO_ID}/keiba_model.pkl ...')
    local_path = hf_hub_download(repo_id=HF_REPO_ID, filename='keiba_model.pkl',
                                 repo_type='dataset', token=HF_TOKEN,
                                 local_dir='.')
    bundle = joblib.load(local_path)

# ── bundleがタプル形式（core_model.py の返値）か確認 ──────────────
if isinstance(bundle, dict):
    raise SystemExit('このpklはHF Hub用bundleではありません（trained_assets.pkl等を指定していませんか？）')

model_a   = bundle[0]   # LGBMRanker  3着内
model_b   = bundle[1]   # LGBMRanker  1着
model_c   = bundle[2]   # LGBMRegressor 着順%
num_features = bundle[5]

feature_names = model_b.booster_.feature_name()

def get_importance(model, label):
    imp = model.booster_.feature_importance(importance_type=args.type)
    return pd.Series(imp.astype(float), index=feature_names, name=label)

df = pd.DataFrame({
    'A(3着内 W=0.058)': get_importance(model_a, 'A'),
    'B(1着   W=0.816)': get_importance(model_b, 'B'),
    'C(着順% W=0.126)': get_importance(model_c, 'C'),
})

weights = np.array([0.0581, 0.8159, 0.1261])
df['加重スコア'] = df.values @ weights
df = df.sort_values('加重スコア', ascending=False)
df['寄与率%'] = (df['加重スコア'] / df['加重スコア'].sum() * 100).round(2)

top = args.top if args.top > 0 else len(df)
pd.set_option('display.max_rows', top + 5)
pd.set_option('display.width', 130)
pd.set_option('display.float_format', '{:.1f}'.format)

print(f"\n=== Feature Importance ({args.type}) — アンサンブル加重順 ===\n")
out = df[['寄与率%', 'A(3着内 W=0.058)', 'B(1着   W=0.816)', 'C(着順% W=0.126)']].head(top)
print(out.to_string())
print(f"\n合計 {len(df)} 特徴量 | 重み: A×0.058 + B×0.816 + C×0.126")
