"""
recover_april.py - 欠落している振り返りデータを遡って再計算するスクリプト
=================================================================
auto_review.py を過去の日付で繰り返し実行し、ai_daily_history.csv に追記する。

使い方:
  python recover_april.py              # 4月・5月初旬の欠落日を全て実行
  python recover_april.py 20260405     # 特定日だけ実行

必要な環境変数:
  HF_TOKEN   - HuggingFace API トークン（read/write 権限）
  HF_REPO_ID - モデル保存先 Dataset リポジトリ ID

注意:
  現在のモデルで予測し直すため「当時の予想」とは異なる可能性がある。
  あくまで「現在のモデルが当時の条件で予測した場合の成績」として記録される。
"""

import os, sys, datetime, time, subprocess

# 欠落している土日（ai_daily_history.csv に記録がない日）
MISSING_DATES = [
    "20260404", "20260405",
    "20260411", "20260412",
    "20260418", "20260419",
    "20260425", "20260426",
    "20260502", "20260503",
]

def check_env():
    missing = [k for k in ["HF_TOKEN", "HF_REPO_ID"] if not os.environ.get(k)]
    if missing:
        print(f"エラー: 環境変数が未設定です: {', '.join(missing)}")
        print("設定方法 (PowerShell):")
        print('  $env:HF_TOKEN="hf_xxxxxxxxxx"')
        print('  $env:HF_REPO_ID="ebi44323/keiba-ebye-models"')
        sys.exit(1)

def run_one_date(date_str: str) -> bool:
    print(f"\n{'='*55}")
    print(f"  振り返り実行: {date_str[:4]}/{date_str[4:6]}/{date_str[6:]}")
    print(f"{'='*55}")
    result = subprocess.run(
        [sys.executable, "auto_review.py", "--date", date_str],
        capture_output=False,
        text=True,
    )
    if result.returncode == 0:
        print(f"  完了: {date_str}")
        return True
    else:
        print(f"  失敗 (returncode={result.returncode}): {date_str}")
        return False

def main():
    check_env()

    if len(sys.argv) > 1:
        dates = [sys.argv[1]]
        print(f"指定日モード: {dates}")
    else:
        dates = MISSING_DATES
        print(f"欠落日一括モード: {len(dates)}日分")

    ok, ng = 0, 0
    for i, d in enumerate(dates):
        success = run_one_date(d)
        if success: ok += 1
        else:       ng += 1
        if i < len(dates) - 1:
            print("  次の日まで30秒待機中...")
            time.sleep(30)

    print(f"\n{'='*55}")
    print(f"完了: 成功 {ok}日 / 失敗 {ng}日")
    print(f"{'='*55}")
    print("HuggingFace Hub の ai_daily_history.csv を確認してください。")
    print("アプリで振り返りタブを開くと追加されたデータが表示されます。")

if __name__ == "__main__":
    main()
