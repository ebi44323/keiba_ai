"""
check_market_rate_auc.py - 市場勝率の AUC 寄与検証スクリプト
==============================================================
「市場勝率あり」と「市場勝率なし」の2条件でモデルBを学習し
AUC を比較することで、オッズ情報への依存度を確認します。

【使い方】
  cd C:/Users/t-tsuchiya/Documents/keiba_ai
  python check_market_rate_auc.py

【所要時間目安】
  学習データ 約20万行 × 2回学習 → 約2〜5分
==============================================================
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

from src.features_engine import NUM_FEATURES, CAT_FEATURES, TE_COLS, create_features
from src.utils import TRACK_CONDITION_MAP

# ── 1. データ読み込み ─────────────────────────────────────────────
print("=" * 60)
print("市場勝率 AUC 寄与検証")
print("=" * 60)

print("\n📊 学習データ読み込み中...")
for path, kw in [('learning_data_perfect_tier.zip', {'compression': 'zip'}),
                 ('learning_data_perfect_tier.csv', {})]:
    if os.path.exists(path):
        df_raw = pd.read_csv(path, dtype=str, **kw)
        print(f"  → {path}: {len(df_raw):,}行 読み込み完了")
        break
else:
    print("❌ 学習データが見つかりません。スクリプトをプロジェクトルートで実行してください。")
    sys.exit(1)

# ── 2. 特徴量生成 ─────────────────────────────────────────────────
print("\n⚙️  特徴量生成中...")
df, _ = create_features(df_raw)

# ── 3. 学習/テスト分割（直近30日をテスト）────────────────────────
df_valid = df.dropna(subset=['着順', '単勝']).copy()
df_valid['馬券内']  = (df_valid['着順'] <= 3).astype(int)
df_valid['win_label'] = (df_valid['着順'] == 1).astype(int)
df_valid = df_valid.sort_values(['レースID', '馬番'])

max_date = df_valid['日付'].max()
test_start = max_date - pd.Timedelta(days=30)
train_df = df_valid[df_valid['日付'] < test_start].copy()
test_df  = df_valid[df_valid['日付'] >= test_start].copy()

print(f"  訓練: {len(train_df):,}行  テスト: {len(test_df):,}行")
print(f"  テスト期間: {test_start.date()} 〜 {max_date.date()}")

# ── 4. Target Encoding (trainのみから) ──────────────────────────
num_features = list(NUM_FEATURES)
cat_features = list(CAT_FEATURES)

global_mean = train_df['馬券内'].mean()
for col in TE_COLS:
    if col not in train_df.columns: continue
    te = train_df.groupby(col)['馬券内'].mean().to_dict()
    train_df[f'{col}_TE'] = train_df[col].map(te).fillna(global_mean)
    test_df[f'{col}_TE']  = test_df[col].map(te).fillna(global_mean)
    if f'{col}_TE' not in num_features:
        num_features.append(f'{col}_TE')

for col in num_features:
    for df_ in [train_df, test_df]:
        if col not in df_.columns:
            df_[col] = np.nan
        else:
            df_[col] = pd.to_numeric(df_[col], errors='coerce')

for col in cat_features:
    for df_ in [train_df, test_df]:
        if col not in df_.columns:
            df_[col] = '不明'
        df_[col] = df_[col].fillna('不明').astype('category')
    # testのカテゴリをtrainに合わせる
    cats = list(train_df[col].cat.categories)
    if '不明' not in cats: cats.append('不明')
    test_df[col] = pd.Categorical(test_df[col].astype(str), categories=cats)

features_with    = [f for f in (cat_features + num_features) if f in train_df.columns]
features_without = [f for f in features_with if f != '市場勝率']

train_groups = train_df.groupby('レースID', sort=False).size().values
test_groups  = test_df.groupby('レースID', sort=False).size().values

def train_and_eval(feat_list, label):
    """指定特徴量リストでモデルBを学習しAUCを返す"""
    cat_feats = [f for f in cat_features if f in feat_list]
    m = lgb.LGBMRanker(
        n_estimators=477, learning_rate=0.020536,
        num_leaves=19, max_bin=197, cat_smooth=36.63,
        colsample_bytree=0.6777, subsample=0.6289,
        min_child_samples=20, random_state=123,
        verbose=-1,
    )
    m.fit(
        train_df[feat_list], train_df['win_label'],
        group=train_groups,
        categorical_feature=cat_feats,
        eval_set=[(test_df[feat_list], test_df['win_label'])],
        eval_group=[test_groups],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
    )
    preds = m.predict(test_df[feat_list])
    y_true = test_df['win_label']
    auc = roc_auc_score(y_true, preds)
    print(f"  [{label}]  特徴量数: {len(feat_list):3d}  AUC: {auc:.4f}")
    return auc, m

# ── 5. 2条件で学習・評価 ─────────────────────────────────────────
print("\n🏃 モデルB（1着予測ランカー）を2条件で学習中...")
print()

auc_with,    model_with    = train_and_eval(features_with,    "市場勝率あり")
auc_without, model_without = train_and_eval(features_without, "市場勝率なし")

diff = auc_with - auc_without

print()
print("=" * 60)
print("📊 結果サマリー")
print("=" * 60)
print(f"  市場勝率あり  AUC: {auc_with:.4f}")
print(f"  市場勝率なし  AUC: {auc_without:.4f}")
print(f"  差分 (寄与):      {diff:+.4f}")
print()

if diff >= 0.05:
    print("⚠️  市場勝率への依存度が高い（差分 >= 0.05）")
    print("   Optunaは「市場勝率なし」の特徴量セットで回すことを強く推奨します。")
    print("   真のモデル力は AUC {:.4f} 程度です。".format(auc_without))
elif diff >= 0.02:
    print("💡 市場勝率の寄与はやや大きい（差分 0.02〜0.05）")
    print("   Optunaは「市場勝率あり」のまま回してもよいですが、")
    print("   実運用では過剰な信頼を避けてください。")
else:
    print("✅ 市場勝率の寄与は小さい（差分 < 0.02）")
    print("   モデルが独立した予測力を持っています。")
    print("   Optunaはそのまま「市場勝率あり」で回して問題ありません。")

# ── 6. 特徴量重要度（上位15）─────────────────────────────────────
print()
print("=" * 60)
print("📈 特徴量重要度 TOP 15（市場勝率あり）")
print("=" * 60)
imp_with = pd.Series(model_with.feature_importances_, index=features_with)
for i, (feat, val) in enumerate(imp_with.sort_values(ascending=False).head(15).items(), 1):
    marker = "  ← ⚠️ オッズ情報" if feat == '市場勝率' else ""
    print(f"  {i:2d}. {feat:<30s} {val:>8.0f}{marker}")

print()
print("=" * 60)
print("📈 特徴量重要度 TOP 15（市場勝率なし）")
print("=" * 60)
imp_without = pd.Series(model_without.feature_importances_, index=features_without)
for i, (feat, val) in enumerate(imp_without.sort_values(ascending=False).head(15).items(), 1):
    print(f"  {i:2d}. {feat:<30s} {val:>8.0f}")

print()
print("✅ 検証完了")
