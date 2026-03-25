"""
ローカル再学習 → HF Hubへモデル保存スクリプト
使い方:
  python train_and_push.py
  ※ HF_TOKEN と HF_REPO_ID を環境変数にセットしてから実行
    例: $env:HF_TOKEN="hf_xxx"; $env:HF_REPO_ID="username/keiba-ebye"; python train_and_push.py
"""

import os, sys, datetime

HF_TOKEN   = os.environ.get("HF_TOKEN", "")
HF_REPO_ID = os.environ.get("HF_REPO_ID", "")

if not HF_TOKEN or not HF_REPO_ID:
    print("=" * 60)
    print("ERROR: HF_TOKEN と HF_REPO_ID が未設定です。")
    print()
    print("PowerShell で以下を実行してから再度このスクリプトを実行:")
    print('  $env:HF_TOKEN="hf_xxxxxxxxxxxx"')
    print('  $env:HF_REPO_ID="あなたのユーザー名/keiba-ebye"')
    print("=" * 60)
    sys.exit(1)

print(f"[{datetime.datetime.now():%H:%M:%S}] HF_TOKEN OK, HF_REPO_ID={HF_REPO_ID}")
print(f"[{datetime.datetime.now():%H:%M:%S}] 学習データ読み込み中...")

# Streamlitのキャッシュデコレータを無効化してインポート
import unittest.mock as mock

def _passthrough(func=None, **kw):
    """@st.cache_resource / @st.cache_data の無効化mock
    - @decorator      → decorator(func) → func をそのまま返す
    - @decorator(...) → decorator(...) returns lambda f: f
    """
    if callable(func):
        return func           # @st.cache_resource (引数なし)
    return lambda f: f        # @st.cache_resource(ttl=...) (引数あり)

with mock.patch("streamlit.cache_resource", _passthrough), \
     mock.patch("streamlit.cache_data",     _passthrough), \
     mock.patch("streamlit.spinner",        lambda *a, **kw: mock.MagicMock()):
    from src.core_model import prepare_model_and_data, _save_model_to_hub

print(f"[{datetime.datetime.now():%H:%M:%S}] 学習開始（force_retrain=True）...")
print("  ※ 197k行 × 4モデル。完了まで10〜30分かかります。")

bundle = prepare_model_and_data(force_retrain=True)

print(f"[{datetime.datetime.now():%H:%M:%S}] 学習完了。HF Hubへ保存中...")
ok = _save_model_to_hub(bundle)

if ok:
    print(f"[{datetime.datetime.now():%H:%M:%S}] ✅ HF Hubへ保存完了！")
    print("  → HuggingFace Spaceを再起動すると正常に動作するはずです。")
else:
    print(f"[{datetime.datetime.now():%H:%M:%S}] ❌ HF Hub保存失敗。トークン・リポジトリIDを確認してください。")
