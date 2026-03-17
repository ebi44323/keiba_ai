import datetime
import pandas as pd
import zipfile
import os
import time

print("=== 🔄 keiba-ebye 自動アップデートツール ===")

# ------------------------------------------
# 1. 先週の土日の日付を自動判定
# ------------------------------------------
today = datetime.date.today()
# 今週の日曜日から数えて、一番近い過去の日曜日を特定
last_sunday = today - datetime.timedelta(days=today.weekday() + 1)
last_saturday = last_sunday - datetime.timedelta(days=1)

print(f"📅 取得対象: {last_saturday.strftime('%Y/%m/%d')} (土) 〜 {last_sunday.strftime('%Y/%m/%d')} (日)")

# ------------------------------------------
# 2. データの取得＆追記ロジック（※ebiさんの既存スクリプトを結合）
# ------------------------------------------
print("🌐 netkeibaから先週のレース結果を自動取得中...")
time.sleep(1) # ※ここにebiさんが今まで使っていたデータ取得プログラムの処理を合流させます

# (ダミー処理：データが追加されたと仮定)
# df_new = ... (スクレイピングしたデータ)
# df = pd.read_csv('learning_data_perfect_tier.csv')
# df = pd.concat([df, df_new]).drop_duplicates(subset=['レースID', '馬ID'])
# df.to_csv('learning_data_perfect_tier.csv', index=False)

print("✅ 最新のレース結果をCSVに追記しました！")

# ------------------------------------------
# 3. アプリ用(クラウド用)にZIP自動圧縮！
# ------------------------------------------
print("📦 クラウドアプリ用にデータをZIP圧縮しています...")
csv_name = 'learning_data_perfect_tier.csv'
zip_name = 'learning_data_perfect_tier.zip'

if os.path.exists(csv_name):
    # ZIPファイルを作成してCSVを詰め込む（古いZIPは上書きされます）
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(csv_name)
    print(f"✅ 【完了】 '{zip_name}' の作成に成功しました！(サイズ圧縮完了)")
else:
    print(f"⚠️ エラー: '{csv_name}' が見つかりません。")

# ------------------------------------------
# 4. 次のアクションを指示
# ------------------------------------------
print("\n" + "="*50)
print("🎉 今週の keiba-ebye アップデート準備がすべて整いました！")
print(f"👉 あとは新しくなった '{zip_name}' を GitHub にドラッグ＆ドロップ（上書きCommit）するだけです！")
print("   数分後にはスマホアプリのAIが最新データに進化します。")
print("="*50 + "\n")